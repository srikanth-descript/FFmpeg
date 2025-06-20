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
#include "libavutil/csp.h"

#include "avfilter.h"
#include "filters.h"
#include "video.h"

#include "cuda/load_helper.h"
#include "vf_tonemap_cuda.h"
#include "cuda_async_queue.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_GBRPF32,
    AV_PIX_FMT_RGB48,
    AV_PIX_FMT_RGBA64,
    AV_PIX_FMT_P010LE,
    AV_PIX_FMT_P016LE,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV420P10LE,
    AV_PIX_FMT_YUV420P16LE,
    AV_PIX_FMT_NV12,
};

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

enum TonemapAlgorithm {
    TONEMAP_NONE,
    TONEMAP_LINEAR,
    TONEMAP_GAMMA,
    TONEMAP_CLIP,
    TONEMAP_REINHARD,
    TONEMAP_HABLE,
    TONEMAP_MOBIUS,
    TONEMAP_MAX,
};

typedef struct TonemapCudaContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;

    enum AVPixelFormat in_fmt, out_fmt;
    const AVPixFmtDescriptor *in_desc, *out_desc;

    AVBufferRef *frames_ctx;
    AVFrame     *frame;
    AVFrame *tmp_frame;

    CUcontext   cu_ctx;
    CUmodule    cu_module;
    CUfunction  cu_func;
    CUstream    cu_stream;

    // Async processing queue
    CudaAsyncQueue async_queue;
    int async_depth;
    int async_streams;
    int async_enabled;

    enum TonemapAlgorithm tonemap;
    double param;
    double desat;
    double peak;

    const AVLumaCoefficients *coeffs;
    int passthrough;
} TonemapCudaContext;

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static av_cold int tonemap_cuda_init(AVFilterContext *ctx)
{
    TonemapCudaContext *s = ctx->priv;

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    // Set async processing defaults
    if (s->async_depth <= 0)
        s->async_depth = DEFAULT_FRAME_QUEUE_SIZE;
    if (s->async_streams <= 0)
        s->async_streams = DEFAULT_CUDA_STREAMS;
    
    s->async_enabled = (s->async_depth > 1 || s->async_streams > 1);

    switch(s->tonemap) {
    case TONEMAP_GAMMA:
        if (isnan(s->param))
            s->param = 1.8f;
        break;
    case TONEMAP_REINHARD:
        if (!isnan(s->param))
            s->param = (1.0f - s->param) / s->param;
        break;
    case TONEMAP_MOBIUS:
        if (isnan(s->param))
            s->param = 0.3f;
        break;
    }

    if (isnan(s->param))
        s->param = 1.0f;

    return 0;
}

static av_cold void tonemap_cuda_uninit(AVFilterContext *ctx)
{
    TonemapCudaContext *s = ctx->priv;

    // Uninitialize async queue first
    if (s->async_enabled)
        cuda_async_queue_uninit(&s->async_queue);

    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        CHECK_CU(cu->cuModuleUnload(s->cu_module));
        s->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static av_cold int init_hwframe_ctx(TonemapCudaContext *s, AVBufferRef *device_ctx, int width, int height)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width     = FFALIGN(width,  32);
    out_ctx->height    = FFALIGN(height, 32);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->frame);
    ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
    if (ret < 0)
        goto fail;

    s->frame->width  = width;
    s->frame->height = height;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static av_cold int tonemap_cuda_load_functions(AVFilterContext *ctx)
{
    TonemapCudaContext *s = ctx->priv;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    char function_name[128];
    int ret;

    extern const unsigned char ff_vf_tonemap_cuda_ptx_data[];
    extern const unsigned int ff_vf_tonemap_cuda_ptx_len;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    ret = ff_cuda_load_module(ctx, s->hwctx, &s->cu_module,
                              ff_vf_tonemap_cuda_ptx_data, ff_vf_tonemap_cuda_ptx_len);
    if (ret < 0)
        goto fail;

    if (s->in_fmt == AV_PIX_FMT_GBRPF32) {
        snprintf(function_name, sizeof(function_name), "tonemap_cuda_planar_float");
    } else if ((s->in_fmt == AV_PIX_FMT_P010LE || s->in_fmt == AV_PIX_FMT_P016LE) && s->out_fmt == AV_PIX_FMT_NV12) {
        snprintf(function_name, sizeof(function_name), "tonemap_cuda_p016_to_nv12");
    } else if (s->in_fmt == AV_PIX_FMT_YUV420P10LE || s->in_fmt == AV_PIX_FMT_YUV420P16LE) {
        snprintf(function_name, sizeof(function_name), "tonemap_cuda_yuv420p10");
    } else {
        snprintf(function_name, sizeof(function_name), "tonemap_cuda_16bit");
    }

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func, s->cu_module, function_name));
    if (ret < 0) {
        av_log(ctx, AV_LOG_FATAL, "Failed to load CUDA function %s\n", function_name);
        goto fail;
    }

fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    return ret;
}

static av_cold int init_processing_chain(AVFilterContext *ctx, int in_width, int in_height,
                                         int out_width, int out_height)
{
    TonemapCudaContext *s = ctx->priv;
    FilterLink     *inl = ff_filter_link(ctx->inputs[0]);
    FilterLink    *outl = ff_filter_link(ctx->outputs[0]);

    AVHWFramesContext *in_frames_ctx;
    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    int ret;

    if (!inl->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    // Convert HDR to SDR format for encoder compatibility
    if (in_format == AV_PIX_FMT_P010LE || in_format == AV_PIX_FMT_P016LE) {
        out_format = AV_PIX_FMT_NV12;
    } else if (in_format == AV_PIX_FMT_YUV420P10LE || in_format == AV_PIX_FMT_YUV420P16LE) {
        out_format = AV_PIX_FMT_YUV420P;
    } else {
        out_format = in_format;
    }

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
    s->in_desc  = av_pix_fmt_desc_get(s->in_fmt);
    s->out_desc = av_pix_fmt_desc_get(s->out_fmt);

    if (in_width == out_width && in_height == out_height && in_format == out_format) {
        s->frames_ctx = av_buffer_ref(inl->hw_frames_ctx);
        if (!s->frames_ctx)
            return AVERROR(ENOMEM);
    } else {
        s->passthrough = 0;

        ret = init_hwframe_ctx(s, in_frames_ctx->device_ref, out_width, out_height);
        if (ret < 0)
            return ret;
    }

    outl->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!outl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int tonemap_cuda_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    FilterLink      *inl = ff_filter_link(inlink);
    TonemapCudaContext *s  = ctx->priv;
    AVHWFramesContext     *frames_ctx;
    AVCUDADeviceContext *device_hwctx;
    int ret;

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    ret = init_processing_chain(ctx, inlink->w, inlink->h, outlink->w, outlink->h);
    if (ret < 0)
        return ret;

    frames_ctx   = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    device_hwctx = frames_ctx->device_ctx->hwctx;

    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    s->coeffs = av_csp_luma_coeffs_from_avcsp(inlink->colorspace);

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s -> w:%d h:%d fmt:%s%s\n",
           inlink->w, inlink->h, av_get_pix_fmt_name(s->in_fmt),
           outlink->w, outlink->h, av_get_pix_fmt_name(s->out_fmt),
           s->passthrough ? " (passthrough)" : "");

    ret = tonemap_cuda_load_functions(ctx);
    if (ret < 0)
        return ret;

    return 0;
}

static int call_tonemap_kernel_async(AVFilterContext *ctx, AVFrame *out, AVFrame *in, CUstream stream)
{
    TonemapCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    CUtexObject tex = 0;
    int ret;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    if (s->in_fmt == AV_PIX_FMT_GBRPF32) {
        // Planar float format
        float coeff_r = av_q2d(s->coeffs->cr);
        float coeff_g = av_q2d(s->coeffs->cg);
        float coeff_b = av_q2d(s->coeffs->cb);
        void *args[] = {
            &in->data[0], &in->data[1], &in->data[2],
            &out->data[0], &out->data[1], &out->data[2],
            &out->width, &out->height, &in->linesize[0], &out->linesize[0],
            &s->tonemap, &s->param, &s->desat, &s->peak,
            &coeff_r, &coeff_g, &coeff_b
        };

        ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func,
                                           DIV_UP(out->width, BLOCKX), DIV_UP(out->height, BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, stream, args, NULL));
    } else if ((s->in_fmt == AV_PIX_FMT_P010LE || s->in_fmt == AV_PIX_FMT_P016LE) && s->out_fmt == AV_PIX_FMT_NV12) {
        // P016 to NV12 conversion
        CUDA_TEXTURE_DESC tex_desc_y = {
            .filterMode = CU_TR_FILTER_MODE_POINT,
            .flags = CU_TRSF_READ_AS_INTEGER,
        };
        
        CUDA_TEXTURE_DESC tex_desc_uv = {
            .filterMode = CU_TR_FILTER_MODE_POINT,
            .flags = CU_TRSF_READ_AS_INTEGER,
        };

        CUDA_RESOURCE_DESC res_desc_y = {
            .resType = CU_RESOURCE_TYPE_PITCH2D,
            .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT16,
            .res.pitch2D.numChannels = 1,
            .res.pitch2D.width = in->width,
            .res.pitch2D.height = in->height,
            .res.pitch2D.pitchInBytes = in->linesize[0],
            .res.pitch2D.devPtr = (CUdeviceptr)in->data[0],
        };

        CUDA_RESOURCE_DESC res_desc_uv = {
            .resType = CU_RESOURCE_TYPE_PITCH2D,
            .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT16,
            .res.pitch2D.numChannels = 2,
            .res.pitch2D.width = in->width / 2,
            .res.pitch2D.height = in->height / 2,
            .res.pitch2D.pitchInBytes = in->linesize[1],
            .res.pitch2D.devPtr = (CUdeviceptr)in->data[1],
        };

        CUtexObject tex_y = 0, tex_uv = 0;
        ret = CHECK_CU(cu->cuTexObjectCreate(&tex_y, &res_desc_y, &tex_desc_y, NULL));
        if (ret < 0) goto exit;
        
        ret = CHECK_CU(cu->cuTexObjectCreate(&tex_uv, &res_desc_uv, &tex_desc_uv, NULL));
        if (ret < 0) goto exit;

        float coeff_r = av_q2d(s->coeffs->cr);
        float coeff_g = av_q2d(s->coeffs->cg);
        float coeff_b = av_q2d(s->coeffs->cb);

        void *args[] = {
            &tex_y, &tex_uv,
            &out->data[0], &out->data[1],
            &out->width, &out->height, &out->linesize[0], &out->linesize[1],
            &s->tonemap, &s->param, &s->desat, &s->peak,
            &coeff_r, &coeff_g, &coeff_b
        };

        ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func,
                                           DIV_UP(out->width, BLOCKX), DIV_UP(out->height, BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, stream, args, NULL));
        
        if (tex_y) CHECK_CU(cu->cuTexObjectDestroy(tex_y));
        if (tex_uv) CHECK_CU(cu->cuTexObjectDestroy(tex_uv));
    } else if (s->in_fmt == AV_PIX_FMT_YUV420P10LE || s->in_fmt == AV_PIX_FMT_YUV420P16LE) {
        // YUV format handling
        CUDA_TEXTURE_DESC tex_desc_y = {
            .filterMode = CU_TR_FILTER_MODE_POINT,
            .flags = CU_TRSF_READ_AS_INTEGER,
        };
        
        CUDA_TEXTURE_DESC tex_desc_uv = {
            .filterMode = CU_TR_FILTER_MODE_POINT,
            .flags = CU_TRSF_READ_AS_INTEGER,
        };

        CUDA_RESOURCE_DESC res_desc_y = {
            .resType = CU_RESOURCE_TYPE_PITCH2D,
            .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT16,
            .res.pitch2D.numChannels = 1,
            .res.pitch2D.width = in->width,
            .res.pitch2D.height = in->height,
            .res.pitch2D.pitchInBytes = in->linesize[0],
            .res.pitch2D.devPtr = (CUdeviceptr)in->data[0],
        };

        CUDA_RESOURCE_DESC res_desc_uv = {
            .resType = CU_RESOURCE_TYPE_PITCH2D,
            .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT16,
            .res.pitch2D.numChannels = 2,
            .res.pitch2D.width = in->width / 2,
            .res.pitch2D.height = in->height / 2,
            .res.pitch2D.pitchInBytes = in->linesize[1],
            .res.pitch2D.devPtr = (CUdeviceptr)in->data[1],
        };

        CUtexObject tex_y = 0, tex_uv = 0;
        ret = CHECK_CU(cu->cuTexObjectCreate(&tex_y, &res_desc_y, &tex_desc_y, NULL));
        if (ret < 0) goto exit;
        
        ret = CHECK_CU(cu->cuTexObjectCreate(&tex_uv, &res_desc_uv, &tex_desc_uv, NULL));
        if (ret < 0) goto exit;

        float coeff_r = av_q2d(s->coeffs->cr);
        float coeff_g = av_q2d(s->coeffs->cg);
        float coeff_b = av_q2d(s->coeffs->cb);

        void *args[] = {
            &tex_y, &tex_uv,
            &out->data[0], &out->data[1], &out->data[2],
            &out->width, &out->height, &out->linesize[0], &out->linesize[1],
            &s->tonemap, &s->param, &s->desat, &s->peak,
            &coeff_r, &coeff_g, &coeff_b
        };

        ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func,
                                           DIV_UP(out->width, BLOCKX), DIV_UP(out->height, BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, stream, args, NULL));
        
        if (tex_y) CHECK_CU(cu->cuTexObjectDestroy(tex_y));
        if (tex_uv) CHECK_CU(cu->cuTexObjectDestroy(tex_uv));
    } else {
        // 16-bit formats with texture
        CUDA_TEXTURE_DESC tex_desc = {
            .filterMode = CU_TR_FILTER_MODE_POINT,
            .flags = CU_TRSF_READ_AS_INTEGER,
        };

        CUDA_RESOURCE_DESC res_desc = {
            .resType = CU_RESOURCE_TYPE_PITCH2D,
            .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT16,
            .res.pitch2D.numChannels = 4,
            .res.pitch2D.width = in->width,
            .res.pitch2D.height = in->height,
            .res.pitch2D.pitchInBytes = in->linesize[0],
            .res.pitch2D.devPtr = (CUdeviceptr)in->data[0],
        };

        ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
        if (ret < 0)
            goto exit;

        int bit_depth = (s->in_fmt == AV_PIX_FMT_RGB48) ? 16 : 16;
        float coeff_r = av_q2d(s->coeffs->cr);
        float coeff_g = av_q2d(s->coeffs->cg);
        float coeff_b = av_q2d(s->coeffs->cb);

        void *args[] = {
            &tex,
            &out->data[0],
            &out->width, &out->height, &out->linesize[0],
            &s->tonemap, &s->param, &s->desat, &s->peak,
            &coeff_r, &coeff_g, &coeff_b,
            &bit_depth
        };

        ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func,
                                           DIV_UP(out->width, BLOCKX), DIV_UP(out->height, BLOCKY), 1,
                                           BLOCKX, BLOCKY, 1, 0, stream, args, NULL));
    }

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    return ret;
}

static int call_tonemap_kernel(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    TonemapCudaContext *s = ctx->priv;
    return call_tonemap_kernel_async(ctx, out, in, s->cu_stream);
}

static int tonemap_cuda_filter_frame_async(AVFilterContext *ctx, AVFrame *in)
{
    TonemapCudaContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    CudaAsyncFrame *async_frame;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;

    // Try to get a completed frame first
    async_frame = cuda_async_queue_get_completed_frame(&s->async_queue);
    if (async_frame) {
        AVFrame *completed_out = async_frame->output_frame;
        async_frame->output_frame = NULL;
        
        ret = ff_filter_frame(outlink, completed_out);
        if (ret < 0)
            return ret;
    }

    // Get a free frame for processing
    async_frame = cuda_async_queue_get_free_frame(&s->async_queue);
    if (!async_frame) {
        // Queue is full, wait for any frame to complete
        cuda_async_queue_sync_all(&s->async_queue);
        
        // Try to get a completed frame and output it
        async_frame = cuda_async_queue_get_completed_frame(&s->async_queue);
        if (async_frame) {
            AVFrame *completed_out = async_frame->output_frame;
            async_frame->output_frame = NULL;
            
            ret = ff_filter_frame(outlink, completed_out);
            if (ret < 0)
                return ret;
        }
        
        // Now get a free frame for new processing
        async_frame = cuda_async_queue_get_free_frame(&s->async_queue);
        if (!async_frame)
            return AVERROR(EAGAIN);
    }

    // Setup input frame
    av_frame_move_ref(async_frame->frame, in);
    
    // Use pre-allocated output frame buffer
    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
        
    ret = av_hwframe_get_buffer(s->frames_ctx, async_frame->output_frame, 0);
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }
    
    // Submit frame to queue BEFORE processing
    ret = cuda_async_queue_submit_frame(&s->async_queue, async_frame);
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }
    
    // Launch kernel on the assigned stream
    CUstream stream = s->async_queue.streams[async_frame->stream_idx];
    ret = call_tonemap_kernel_async(ctx, async_frame->output_frame, async_frame->frame, stream);
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }
    
    // Record completion event
    ret = CHECK_CU(cu->cuEventRecord(async_frame->event_done, stream));
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }
    
    // Copy frame properties
    ret = av_frame_copy_props(async_frame->output_frame, async_frame->frame);
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        return ret;
        
    return 0;
}

static int tonemap_cuda_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext       *ctx = link->dst;
    TonemapCudaContext        *s = ctx->priv;
    AVFilterLink      *outlink = ctx->outputs[0];
    CudaFunctions          *cu = s->hwctx->internal->cuda_dl;

    AVFrame *out = NULL;
    CUcontext dummy;
    int ret = 0;

    if (s->passthrough)
        return ff_filter_frame(outlink, in);

    // Use async processing if enabled
    if (s->async_enabled)
        return tonemap_cuda_filter_frame_async(ctx, in);

    // Original synchronous processing
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

    ret = call_tonemap_kernel(ctx, out, in);
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

#define OFFSET(x) offsetof(TonemapCudaContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption tonemap_cuda_options[] = {
    { "tonemap",      "tonemap algorithm selection", OFFSET(tonemap), AV_OPT_TYPE_INT, {.i64=TONEMAP_NONE}, 0, TONEMAP_MAX-1, FLAGS, .unit = "tonemap" },
        { "none",     0, 0, AV_OPT_TYPE_CONST, {.i64=TONEMAP_NONE},     0, 0, FLAGS, .unit = "tonemap" },
        { "linear",   0, 0, AV_OPT_TYPE_CONST, {.i64=TONEMAP_LINEAR},   0, 0, FLAGS, .unit = "tonemap" },
        { "gamma",    0, 0, AV_OPT_TYPE_CONST, {.i64=TONEMAP_GAMMA},    0, 0, FLAGS, .unit = "tonemap" },
        { "clip",     0, 0, AV_OPT_TYPE_CONST, {.i64=TONEMAP_CLIP},     0, 0, FLAGS, .unit = "tonemap" },
        { "reinhard", 0, 0, AV_OPT_TYPE_CONST, {.i64=TONEMAP_REINHARD}, 0, 0, FLAGS, .unit = "tonemap" },
        { "hable",    0, 0, AV_OPT_TYPE_CONST, {.i64=TONEMAP_HABLE},    0, 0, FLAGS, .unit = "tonemap" },
        { "mobius",   0, 0, AV_OPT_TYPE_CONST, {.i64=TONEMAP_MOBIUS},   0, 0, FLAGS, .unit = "tonemap" },
    { "param",        "tonemap parameter", OFFSET(param), AV_OPT_TYPE_DOUBLE, {.dbl=NAN}, DBL_MIN, DBL_MAX, FLAGS },
    { "desat",        "desaturation strength", OFFSET(desat), AV_OPT_TYPE_DOUBLE, {.dbl=2}, 0, DBL_MAX, FLAGS },
    { "peak",         "signal peak override", OFFSET(peak), AV_OPT_TYPE_DOUBLE, {.dbl=0}, 0, DBL_MAX, FLAGS },
    { "passthrough",  "Do not process frames at all if parameters match", OFFSET(passthrough), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { "async_depth",  "Async frame queue depth for pipeline parallelism", OFFSET(async_depth), AV_OPT_TYPE_INT, {.i64=DEFAULT_FRAME_QUEUE_SIZE}, 1, MAX_FRAME_QUEUE_SIZE, FLAGS },
    { "async_streams", "Number of CUDA streams for concurrent processing", OFFSET(async_streams), AV_OPT_TYPE_INT, {.i64=DEFAULT_CUDA_STREAMS}, 1, MAX_CUDA_STREAMS, FLAGS },
    { NULL },
};

static const AVClass tonemap_cuda_class = {
    .class_name = "tonemap_cuda",
    .item_name  = av_default_item_name,
    .option     = tonemap_cuda_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad tonemap_cuda_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .filter_frame = tonemap_cuda_filter_frame,
    },
};

static const AVFilterPad tonemap_cuda_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = tonemap_cuda_config_props,
    },
};

const FFFilter ff_vf_tonemap_cuda = {
    .p.name        = "tonemap_cuda",
    .p.description = NULL_IF_CONFIG_SMALL("GPU accelerated HDR to SDR tonemap filter"),

    .p.priv_class  = &tonemap_cuda_class,

    .init          = tonemap_cuda_init,
    .uninit        = tonemap_cuda_uninit,

    .priv_size = sizeof(TonemapCudaContext),

    FILTER_INPUTS(tonemap_cuda_inputs),
    FILTER_OUTPUTS(tonemap_cuda_outputs),

    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};