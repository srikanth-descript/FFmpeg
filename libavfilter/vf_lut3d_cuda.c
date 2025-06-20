/*
 * This file is part of FFmpeg.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <float.h>
#include <stdio.h>

#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/mem.h"
#include "libavutil/file_open.h"
#include "libavutil/intfloat.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"

#include "avfilter.h"
#include "filters.h"
#include "video.h"
#include "lut3d.h"

#include "cuda/load_helper.h"
#include "vf_lut3d_cuda.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_0RGB32,
    AV_PIX_FMT_0BGR32,
    AV_PIX_FMT_RGB32,
    AV_PIX_FMT_BGR32,
    AV_PIX_FMT_RGB48,
    AV_PIX_FMT_BGR48,
    AV_PIX_FMT_RGBA64,
    AV_PIX_FMT_BGRA64,
};

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct LUT3DCudaContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;

    enum AVPixelFormat in_fmt, out_fmt;
    const AVPixFmtDescriptor *in_desc, *out_desc;

    AVBufferRef *frames_ctx;
    AVFrame *frame;
    AVFrame *tmp_frame;

    CUcontext cu_ctx;
    CUmodule cu_module;
    CUfunction cu_func;
    CUstream cu_stream;
    CUdeviceptr cu_lut;

    struct rgbvec *lut;
    int lutsize;
    int lutsize2;
    struct rgbvec scale;
    int interpolation;
    char *file;

    int bit_depth;
    int passthrough;
} LUT3DCudaContext;

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static int get_bit_depth(enum AVPixelFormat fmt)
{
    switch (fmt) {
    case AV_PIX_FMT_0RGB32:
    case AV_PIX_FMT_0BGR32:
    case AV_PIX_FMT_RGB32:
    case AV_PIX_FMT_BGR32:
        return 8;
    case AV_PIX_FMT_RGB48:
    case AV_PIX_FMT_BGR48:
        return 12;
    case AV_PIX_FMT_RGBA64:
    case AV_PIX_FMT_BGRA64:
        return 10;
    default:
        return 8;
    }
}

static av_cold int lut3d_cuda_init(AVFilterContext * ctx)
{
    LUT3DCudaContext *s = ctx->priv;

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void lut3d_cuda_uninit(AVFilterContext * ctx)
{
    LUT3DCudaContext *s = ctx->priv;

    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        if (s->cu_lut) {
            CHECK_CU(cu->cuMemFree(s->cu_lut));
            s->cu_lut = 0;
        }
        CHECK_CU(cu->cuModuleUnload(s->cu_module));
        s->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_freep(&s->lut);
    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static av_cold int init_hwframe_ctx(LUT3DCudaContext * s,
                                    AVBufferRef * device_ctx, int width,
                                    int height)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext *) out_ref->data;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width = FFALIGN(width, 32);
    out_ctx->height = FFALIGN(height, 32);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->frame);
    ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
    if (ret < 0)
        goto fail;

    s->frame->width = width;
    s->frame->height = height;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
  fail:
    av_buffer_unref(&out_ref);
    return ret;
}

#define MAX_LINE_SIZE 512

static int skip_line(const char *p)
{
    while (*p && av_isspace(*p))
        p++;
    return !*p || *p == '#';
}

#define NEXT_LINE(loop_cond) do {                           \
    if (!fgets(line, sizeof(line), f)) {                    \
        av_log(ctx, AV_LOG_ERROR, "Unexpected EOF\n");      \
        return AVERROR_INVALIDDATA;                         \
    }                                                       \
} while (loop_cond)

static int parse_cube(AVFilterContext * ctx, FILE * f)
{
    LUT3DCudaContext *lut3d = ctx->priv;
    char line[MAX_LINE_SIZE];
    float min[3] = { 0.0, 0.0, 0.0 };
    float max[3] = { 1.0, 1.0, 1.0 };
    int size = 0, size2, i, j, k;

    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "LUT_3D_SIZE", 11)) {
            size = strtol(line + 12, NULL, 0);
            if (size < 2 || size > MAX_LEVEL) {
                av_log(ctx, AV_LOG_ERROR,
                       "Too large or invalid 3D LUT size\n");
                return AVERROR(EINVAL);
            }
            lut3d->lutsize = size;
            lut3d->lutsize2 = size * size;
            break;
        }
    }

    if (!size) {
        av_log(ctx, AV_LOG_ERROR, "3D LUT size not found\n");
        return AVERROR(EINVAL);
    }

    if (!
        (lut3d->lut =
         av_malloc_array(size * size * size, sizeof(*lut3d->lut))))
        return AVERROR(ENOMEM);

    size2 = size * size;
    rewind(f);

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            for (k = 0; k < size; k++) {
                struct rgbvec *vec = &lut3d->lut[i * size2 + j * size + k];

                do {
                  try_again:
                    NEXT_LINE(0);
                    if (!strncmp(line, "DOMAIN_", 7)) {
                        float *vals = NULL;
                        if (!strncmp(line + 7, "MIN ", 4))
                            vals = min;
                        else if (!strncmp(line + 7, "MAX ", 4))
                            vals = max;
                        if (!vals)
                            return AVERROR_INVALIDDATA;
                        if (av_sscanf
                            (line + 11, "%f %f %f", vals, vals + 1,
                             vals + 2) != 3)
                            return AVERROR_INVALIDDATA;
                        av_log(ctx, AV_LOG_DEBUG,
                               "min: %f %f %f | max: %f %f %f\n", min[0],
                               min[1], min[2], max[0], max[1], max[2]);
                        goto try_again;
                    } else if (!strncmp(line, "TITLE", 5)) {
                        goto try_again;
                    }
                } while (skip_line(line));
                if (av_sscanf(line, "%f %f %f", &vec->r, &vec->g, &vec->b)
                    != 3)
                    return AVERROR_INVALIDDATA;
                vec->r = av_clipf(vec->r, 0.f, 1.f);
                vec->g = av_clipf(vec->g, 0.f, 1.f);
                vec->b = av_clipf(vec->b, 0.f, 1.f);
            }
        }
    }

    lut3d->scale.r =
        av_clipf(1. / (max[0] - min[0]), 0.f, 1.f) * (lut3d->lutsize - 1);
    lut3d->scale.g =
        av_clipf(1. / (max[1] - min[1]), 0.f, 1.f) * (lut3d->lutsize - 1);
    lut3d->scale.b =
        av_clipf(1. / (max[2] - min[2]), 0.f, 1.f) * (lut3d->lutsize - 1);

    return 0;
}

static av_cold int lut3d_cuda_load_cube(AVFilterContext * ctx,
                                        const char *fname)
{
    FILE *f;
    int ret;

    f = avpriv_fopen_utf8(fname, "r");
    if (!f) {
        ret = AVERROR_INVALIDDATA;
        av_log(ctx, AV_LOG_ERROR, "Cannot read file '%s': %s\n", fname,
               av_err2str(ret));
        return ret;
    }

    ret = parse_cube(ctx, f);
    fclose(f);
    return ret;
}

static av_cold int lut3d_cuda_load_functions(AVFilterContext * ctx)
{
    LUT3DCudaContext *s = ctx->priv;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    char function_name[128];
    int ret;

    extern const unsigned char ff_vf_lut3d_cuda_ptx_data[];
    extern const unsigned int ff_vf_lut3d_cuda_ptx_len;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    ret = ff_cuda_load_module(ctx, s->hwctx, &s->cu_module,
                              ff_vf_lut3d_cuda_ptx_data,
                              ff_vf_lut3d_cuda_ptx_len);
    if (ret < 0)
        goto fail;

    switch (s->interpolation) {
    case INTERPOLATE_NEAREST:
        snprintf(function_name, sizeof(function_name),
                 "lut3d_interp_%d_nearest", s->bit_depth);
        break;
    case INTERPOLATE_TRILINEAR:
        snprintf(function_name, sizeof(function_name),
                 "lut3d_interp_%d_trilinear", s->bit_depth);
        break;
    case INTERPOLATE_TETRAHEDRAL:
    default:
        snprintf(function_name, sizeof(function_name),
                 "lut3d_interp_%d_tetrahedral", s->bit_depth);
        break;
    }

    ret =
        CHECK_CU(cu->
                 cuModuleGetFunction(&s->cu_func, s->cu_module,
                                     function_name));
    if (ret < 0) {
        av_log(ctx, AV_LOG_FATAL, "Failed to load CUDA function %s\n",
               function_name);
        goto fail;
    }

    ret =
        CHECK_CU(cu->
                 cuMemAlloc(&s->cu_lut,
                            s->lutsize * s->lutsize * s->lutsize * 3 *
                            sizeof(float)));
    if (ret < 0)
        goto fail;

    ret =
        CHECK_CU(cu->
                 cuMemcpyHtoD(s->cu_lut, s->lut,
                              s->lutsize * s->lutsize * s->lutsize *
                              sizeof(struct rgbvec)));
    if (ret < 0)
        goto fail;

  fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    return ret;
}

static av_cold int init_processing_chain(AVFilterContext * ctx,
                                         int in_width, int in_height,
                                         int out_width, int out_height)
{
    LUT3DCudaContext *s = ctx->priv;
    FilterLink *inl = ff_filter_link(ctx->inputs[0]);
    FilterLink *outl = ff_filter_link(ctx->outputs[0]);

    AVHWFramesContext *in_frames_ctx;
    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    int ret;

    if (!inl->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext *) inl->hw_frames_ctx->data;
    in_format = in_frames_ctx->sw_format;
    out_format = in_format;

    if (!format_is_supported(in_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(in_format));
        return AVERROR(ENOSYS);
    }
    if (!format_is_supported(out_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(out_format));
        return AVERROR(ENOSYS);
    }

    s->in_fmt = in_format;
    s->out_fmt = out_format;
    s->in_desc = av_pix_fmt_desc_get(s->in_fmt);
    s->out_desc = av_pix_fmt_desc_get(s->out_fmt);
    s->bit_depth = get_bit_depth(s->in_fmt);

    if (in_width == out_width && in_height == out_height
        && in_format == out_format) {
        s->frames_ctx = av_buffer_ref(inl->hw_frames_ctx);
        if (!s->frames_ctx)
            return AVERROR(ENOMEM);
    } else {
        s->passthrough = 0;

        ret =
            init_hwframe_ctx(s, in_frames_ctx->device_ref, out_width,
                             out_height);
        if (ret < 0)
            return ret;
    }

    outl->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!outl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int lut3d_cuda_config_props(AVFilterLink * outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    FilterLink *inl = ff_filter_link(inlink);
    LUT3DCudaContext *s = ctx->priv;
    AVHWFramesContext *frames_ctx;
    AVCUDADeviceContext *device_hwctx;
    int ret;

    if (s->file) {
        ret = lut3d_cuda_load_cube(ctx, s->file);
        if (ret < 0)
            return ret;
    }

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    ret =
        init_processing_chain(ctx, inlink->w, inlink->h, outlink->w,
                              outlink->h);
    if (ret < 0)
        return ret;

    frames_ctx = (AVHWFramesContext *) inl->hw_frames_ctx->data;
    device_hwctx = frames_ctx->device_ctx->hwctx;

    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s -> w:%d h:%d fmt:%s%s\n",
           inlink->w, inlink->h, av_get_pix_fmt_name(s->in_fmt),
           outlink->w, outlink->h, av_get_pix_fmt_name(s->out_fmt),
           s->passthrough ? " (passthrough)" : "");

    ret = lut3d_cuda_load_functions(ctx);
    if (ret < 0)
        return ret;

    return 0;
}

static int call_lut3d_kernel(AVFilterContext * ctx, AVFrame * out,
                             AVFrame * in)
{
    LUT3DCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    CUtexObject tex = 0;
    int ret;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format =
            (s->bit_depth >
             8) ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT8,
        .res.pitch2D.numChannels = 4,
        .res.pitch2D.width = in->width,
        .res.pitch2D.height = in->height,
        .res.pitch2D.pitchInBytes = in->linesize[0],
        .res.pitch2D.devPtr = (CUdeviceptr) in->data[0],
    };

    ret =
        CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    void *args[] = {
        &tex,
        &out->data[0],
        &out->width, &out->height, &out->linesize[0],
        &s->cu_lut, &s->lutsize,
        &s->scale.r, &s->scale.g, &s->scale.b
    };

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func,
                                      DIV_UP(out->width, BLOCKX),
                                      DIV_UP(out->height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream,
                                      args, NULL));

  exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    return ret;
}

static int lut3d_cuda_filter_frame(AVFilterLink * link, AVFrame * in)
{
    AVFilterContext *ctx = link->dst;
    LUT3DCudaContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    AVFrame *out = NULL;
    CUcontext dummy;
    int ret = 0;

    if (s->passthrough)
        return ff_filter_frame(outlink, in);

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = av_hwframe_get_buffer(s->frames_ctx, out, 0);
    if (ret < 0)
        goto fail;

    ret = call_lut3d_kernel(ctx, out, in);
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        goto fail;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

  fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(LUT3DCudaContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
#define TFLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_RUNTIME_PARAM)

static const AVOption lut3d_cuda_options[] = {
    { "file", "set 3D LUT file name", OFFSET(file), AV_OPT_TYPE_STRING,
     {.str = NULL},.flags = FLAGS },
    { "interp", "select interpolation mode", OFFSET(interpolation),
     AV_OPT_TYPE_INT, {.i64 =
                       INTERPOLATE_TETRAHEDRAL}, 0, NB_INTERP_MODE - 1,
     TFLAGS,.unit = "interp_mode" },
    { "nearest", "use values from the nearest defined points", 0,
     AV_OPT_TYPE_CONST, {.i64 = INTERPOLATE_NEAREST}, 0, 0, TFLAGS,.unit =
     "interp_mode" },
    { "trilinear", "interpolate values using the 8 points defining a cube",
     0, AV_OPT_TYPE_CONST, {.i64 =
                            INTERPOLATE_TRILINEAR}, 0, 0, TFLAGS,.unit =
     "interp_mode" },
    { "tetrahedral", "interpolate values using a tetrahedron", 0,
     AV_OPT_TYPE_CONST, {.i64 =
                         INTERPOLATE_TETRAHEDRAL}, 0, 0, TFLAGS,.unit =
     "interp_mode" },
    { "passthrough", "Do not process frames at all if parameters match",
     OFFSET(passthrough), AV_OPT_TYPE_BOOL, {.i64 = 1}, 0, 1, FLAGS },
    { NULL },
};

static const AVClass lut3d_cuda_class = {
    .class_name = "lut3d_cuda",
    .item_name = av_default_item_name,
    .option = lut3d_cuda_options,
    .version = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad lut3d_cuda_inputs[] = {
    {
     .name = "default",
     .type = AVMEDIA_TYPE_VIDEO,
     .filter_frame = lut3d_cuda_filter_frame,
      },
};

static const AVFilterPad lut3d_cuda_outputs[] = {
    {
     .name = "default",
     .type = AVMEDIA_TYPE_VIDEO,
     .config_props = lut3d_cuda_config_props,
      },
};

const FFFilter ff_vf_lut3d_cuda = {
    .p.name = "lut3d_cuda",
    .p.description = NULL_IF_CONFIG_SMALL("GPU accelerated 3D LUT filter"),

    .p.priv_class = &lut3d_cuda_class,

    .init = lut3d_cuda_init,
    .uninit = lut3d_cuda_uninit,

    .priv_size = sizeof(LUT3DCudaContext),

    FILTER_INPUTS(lut3d_cuda_inputs),
    FILTER_OUTPUTS(lut3d_cuda_outputs),

    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
