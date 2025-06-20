/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <string.h>

#include "libavutil/common.h"
#include "libavutil/cuda_check.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "filters.h"
#include "colorspace.h"

#include "cuda/load_helper.h"
#include "cuda_async_queue.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV444P,
};

#define DIV_UP(a, b) (((a) + (b)-1) / (b))
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct CUDAColorspaceContext {
    const AVClass* class;

    AVCUDADeviceContext* hwctx;
    AVBufferRef* frames_ctx;
    AVFrame* own_frame;
    AVFrame* tmp_frame;

    CUcontext cu_ctx;
    CUstream cu_stream;
    CUmodule cu_module;
    CUfunction cu_convert[AVCOL_RANGE_NB];
    CUfunction cu_convert_colorspace;

    // Async processing queue
    CudaAsyncQueue async_queue;
    int async_depth;
    int async_streams;
    int async_enabled;

    enum AVPixelFormat pix_fmt;
    enum AVColorRange range;
    enum AVColorSpace in_csp, out_csp, user_csp;
    enum AVColorPrimaries in_prm, out_prm, user_prm;
    enum AVColorTransferCharacteristic in_trc, out_trc, user_trc;

    // Conversion matrices for YUV-to-YUV colorspace conversion
    float yuv2yuv_matrix[3][3];
    float yuv_offset[2]; // input and output offsets
    int colorspace_conversion_needed;

    int num_planes;
} CUDAColorspaceContext;

static int calculate_yuv2yuv_matrix(CUDAColorspaceContext* s)
{
    const AVLumaCoefficients *in_lumacoef, *out_lumacoef;
    double yuv2rgb[3][3], rgb2yuv[3][3], yuv2yuv[3][3];
    int i, j;

    // Get luma coefficients for input and output colorspaces
    in_lumacoef = av_csp_luma_coeffs_from_avcsp(s->in_csp);
    out_lumacoef = av_csp_luma_coeffs_from_avcsp(s->out_csp);

    if (!in_lumacoef || !out_lumacoef) {
        return AVERROR(EINVAL);
    }

    // Calculate YUV to RGB matrix for input colorspace
    ff_fill_rgb2yuv_table(in_lumacoef, rgb2yuv);
    ff_matrix_invert_3x3(rgb2yuv, yuv2rgb);

    // Calculate RGB to YUV matrix for output colorspace
    ff_fill_rgb2yuv_table(out_lumacoef, rgb2yuv);

    // Combine: YUV_in -> RGB -> YUV_out
    ff_matrix_mul_3x3(yuv2yuv, yuv2rgb, rgb2yuv);

    // Convert to float for CUDA kernel
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            s->yuv2yuv_matrix[i][j] = (float)yuv2yuv[i][j];
        }
    }

    return 0;
}

static av_cold int cudacolorspace_init(AVFilterContext* ctx)
{
    CUDAColorspaceContext* s = ctx->priv;

    s->own_frame = av_frame_alloc();
    if (!s->own_frame)
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

    return 0;
}

static av_cold void cudacolorspace_uninit(AVFilterContext* ctx)
{
    CUDAColorspaceContext* s = ctx->priv;

    // Uninitialize async queue first
    if (s->async_enabled)
        cuda_async_queue_uninit(&s->async_queue);

    if (s->hwctx && s->cu_module) {
        CudaFunctions* cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        CHECK_CU(cu->cuModuleUnload(s->cu_module));
        s->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_frame_free(&s->own_frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static av_cold int init_hwframe_ctx(CUDAColorspaceContext* s, AVBufferRef* device_ctx,
                                    int width, int height)
{
    AVBufferRef* out_ref = NULL;
    AVHWFramesContext* out_ctx;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);

    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->pix_fmt;
    out_ctx->width = FFALIGN(width, 32);
    out_ctx->height = FFALIGN(height, 32);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->own_frame);
    ret = av_hwframe_get_buffer(out_ref, s->own_frame, 0);
    if (ret < 0)
        goto fail;

    s->own_frame->width = width;
    s->own_frame->height = height;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (fmt == supported_formats[i])
            return 1;

    return 0;
}

static av_cold int init_processing_chain(AVFilterContext* ctx, int width,
                                         int height)
{
    FilterLink          *inl = ff_filter_link(ctx->inputs[0]);
    FilterLink         *outl = ff_filter_link(ctx->outputs[0]);
    CUDAColorspaceContext* s = ctx->priv;
    AVHWFramesContext* in_frames_ctx;

    int ret;

    if (!inl->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }

    in_frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    s->pix_fmt = in_frames_ctx->sw_format;

    if (!format_is_supported(s->pix_fmt)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
               av_get_pix_fmt_name(s->pix_fmt));
        return AVERROR(EINVAL);
    }

    if ((AVCOL_RANGE_MPEG != s->range) && (AVCOL_RANGE_JPEG != s->range) && 
        (AVCOL_RANGE_UNSPECIFIED != s->range)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported color range\n");
        return AVERROR(EINVAL);
    }

    s->num_planes = av_pix_fmt_count_planes(s->pix_fmt);

    // Determine if colorspace conversion is needed
    s->colorspace_conversion_needed = (s->in_csp != AVCOL_SPC_UNSPECIFIED && 
                                       s->out_csp != AVCOL_SPC_UNSPECIFIED && 
                                       s->in_csp != s->out_csp);

    if (s->colorspace_conversion_needed) {
        ret = calculate_yuv2yuv_matrix(s);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to calculate colorspace conversion matrix\n");
            return ret;
        }
        av_log(ctx, AV_LOG_INFO, "Colorspace conversion: %s -> %s\n",
               av_color_space_name(s->in_csp), av_color_space_name(s->out_csp));
    }

    ret = init_hwframe_ctx(s, in_frames_ctx->device_ref, width, height);
    if (ret < 0)
        return ret;

    outl->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!outl->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int cudacolorspace_load_functions(AVFilterContext* ctx)
{
    CUDAColorspaceContext* s = ctx->priv;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    CudaFunctions* cu = s->hwctx->internal->cuda_dl;
    int ret;

    extern const unsigned char ff_vf_colorspace_cuda_ptx_data[];
    extern const unsigned int ff_vf_colorspace_cuda_ptx_len;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    ret = ff_cuda_load_module(ctx, s->hwctx, &s->cu_module,
                              ff_vf_colorspace_cuda_ptx_data,
                              ff_vf_colorspace_cuda_ptx_len);
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_convert[AVCOL_RANGE_MPEG], s->cu_module, "to_mpeg_cuda"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_convert[AVCOL_RANGE_JPEG], s->cu_module, "to_jpeg_cuda"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_convert_colorspace, s->cu_module, "colorspace_convert_cuda"));
    if (ret < 0)
        goto fail;

fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static av_cold int cudacolorspace_config_props(AVFilterLink* outlink)
{
    AVFilterContext* ctx = outlink->src;
    AVFilterLink* inlink = outlink->src->inputs[0];
    FilterLink      *inl = ff_filter_link(inlink);
    CUDAColorspaceContext* s = ctx->priv;
    AVHWFramesContext* frames_ctx;
    AVCUDADeviceContext* device_hwctx;
    int ret;

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    // Set defaults - actual colorspace will be determined from frame properties
    s->in_csp = inlink->colorspace != AVCOL_SPC_UNSPECIFIED ? inlink->colorspace : AVCOL_SPC_BT709;
    s->in_prm = AVCOL_PRI_BT709;  // Default, will be updated from frame
    s->in_trc = AVCOL_TRC_BT709;  // Default, will be updated from frame

    // Set output colorspace properties
    s->out_csp = s->user_csp != AVCOL_SPC_UNSPECIFIED ? s->user_csp : s->in_csp;
    s->out_prm = s->user_prm != AVCOL_PRI_UNSPECIFIED ? s->user_prm : s->in_prm;
    s->out_trc = s->user_trc != AVCOL_TRC_UNSPECIFIED ? s->user_trc : s->in_trc;

    // Set output link properties
    outlink->colorspace = s->out_csp;

    ret = init_processing_chain(ctx, inlink->w, inlink->h);
    if (ret < 0)
        return ret;

    frames_ctx = (AVHWFramesContext*)inl->hw_frames_ctx->data;
    device_hwctx = frames_ctx->device_ctx->hwctx;

    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    if (inlink->sample_aspect_ratio.num) {
        outlink->sample_aspect_ratio = av_mul_q(
            (AVRational){outlink->h * inlink->w, outlink->w * inlink->h},
            inlink->sample_aspect_ratio);
    } else {
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    }

    ret = cudacolorspace_load_functions(ctx);
    if (ret < 0)
        return ret;

    return ret;
}

static int conv_cuda_convert(AVFilterContext* ctx, AVFrame* out, AVFrame* in)
{
    CUDAColorspaceContext* s = ctx->priv;
    CudaFunctions* cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = s->hwctx->cuda_ctx;
    int ret;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    // Update input colorspace properties from frame
    s->in_csp = in->colorspace != AVCOL_SPC_UNSPECIFIED ? in->colorspace : s->in_csp;
    s->in_prm = in->color_primaries != AVCOL_PRI_UNSPECIFIED ? in->color_primaries : s->in_prm;
    s->in_trc = in->color_trc != AVCOL_TRC_UNSPECIFIED ? in->color_trc : s->in_trc;

    // Update output colorspace if user didn't specify
    if (s->user_csp == AVCOL_SPC_UNSPECIFIED) s->out_csp = s->in_csp;
    if (s->user_prm == AVCOL_PRI_UNSPECIFIED) s->out_prm = s->in_prm;
    if (s->user_trc == AVCOL_TRC_UNSPECIFIED) s->out_trc = s->in_trc;

    // Check if we need to recalculate matrix
    int need_recalc = (s->colorspace_conversion_needed && 
                      (s->in_csp != s->out_csp));
    
    if (need_recalc) {
        ret = calculate_yuv2yuv_matrix(s);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to recalculate colorspace conversion matrix\n");
            goto fail;
        }
    }

    // Set output frame properties
    out->color_range = s->range != AVCOL_RANGE_UNSPECIFIED ? s->range : in->color_range;
    out->colorspace = s->out_csp;
    out->color_primaries = s->out_prm;
    out->color_trc = s->out_trc;

    // If only colorspace conversion is needed (no range conversion)
    if (s->colorspace_conversion_needed && (in->color_range == out->color_range || s->range == AVCOL_RANGE_UNSPECIFIED)) {
        for (int i = 0; i < s->num_planes; i++) {
            int width = in->width, height = in->height;

            switch (s->pix_fmt) {
            case AV_PIX_FMT_YUV444P:
                break;
            case AV_PIX_FMT_YUV420P:
                width = (i > 0) ? in->width / 2 : in->width;
                /* fall-through */
            case AV_PIX_FMT_NV12:
                height = (i > 0) ? in->height / 2 : in->height;
                break;
            default:
                av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
                       av_get_pix_fmt_name(s->pix_fmt));
                return AVERROR(EINVAL);
            }

            void* args[] = {&in->data[i], &out->data[i], &in->linesize[i], &out->linesize[i],
                            &width, &height, &i, 
                            &s->yuv2yuv_matrix[0][0], &s->yuv2yuv_matrix[0][1], &s->yuv2yuv_matrix[0][2],
                            &s->yuv2yuv_matrix[1][0], &s->yuv2yuv_matrix[1][1], &s->yuv2yuv_matrix[1][2],
                            &s->yuv2yuv_matrix[2][0], &s->yuv2yuv_matrix[2][1], &s->yuv2yuv_matrix[2][2]};
            ret = CHECK_CU(cu->cuLaunchKernel(
                s->cu_convert_colorspace, DIV_UP(width, BLOCKX),
                DIV_UP(height, BLOCKY), 1, BLOCKX, BLOCKY, 1, 0, s->cu_stream,
                args, NULL));
            if (ret < 0)
                goto fail;
        }
    } else {
        // Handle range conversion or passthrough
        for (int i = 0; i < s->num_planes; i++) {
            int width = in->width, height = in->height, comp_id = (i > 0);

            switch (s->pix_fmt) {
            case AV_PIX_FMT_YUV444P:
                break;
            case AV_PIX_FMT_YUV420P:
                width = comp_id ? in->width / 2 : in->width;
                /* fall-through */
            case AV_PIX_FMT_NV12:
                height = comp_id ? in->height / 2 : in->height;
                break;
            default:
                av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
                       av_get_pix_fmt_name(s->pix_fmt));
                return AVERROR(EINVAL);
            }

            if (s->range != AVCOL_RANGE_UNSPECIFIED && in->color_range != out->color_range) {
                if (!s->cu_convert[out->color_range]) {
                    av_log(ctx, AV_LOG_ERROR, "Unsupported color range\n");
                    return AVERROR(EINVAL);
                }

                void* args[] = {&in->data[i], &out->data[i], &in->linesize[i], &comp_id};
                ret = CHECK_CU(cu->cuLaunchKernel(
                    s->cu_convert[out->color_range], DIV_UP(width, BLOCKX),
                    DIV_UP(height, BLOCKY), 1, BLOCKX, BLOCKY, 1, 0, s->cu_stream,
                    args, NULL));
                if (ret < 0)
                    goto fail;
            } else {
                ret = av_hwframe_transfer_data(out, in, 0);
                if (ret < 0)
                    goto fail;
            }
        }
    }

fail:
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return ret;
}

static int cudacolorspace_conv(AVFilterContext* ctx, AVFrame* out, AVFrame* in)
{
    CUDAColorspaceContext* s = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];
    AVFrame* src = in;
    int ret;

    ret = conv_cuda_convert(ctx, s->own_frame, src);
    if (ret < 0)
        return ret;

    src = s->own_frame;
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->tmp_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, s->own_frame);
    av_frame_move_ref(s->own_frame, s->tmp_frame);

    s->own_frame->width = outlink->w;
    s->own_frame->height = outlink->h;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudacolorspace_filter_frame_async(AVFilterContext* ctx, AVFrame* in)
{
    CUDAColorspaceContext* s = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];
    AVFilterLink* inlink = ctx->inputs[0];
    CudaAsyncFrame* async_frame;
    CudaFunctions* cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;

    // Try to get a completed frame first
    async_frame = cuda_async_queue_get_completed_frame(&s->async_queue);
    if (async_frame) {
        AVFrame* completed_out = async_frame->output_frame;
        async_frame->output_frame = NULL;
        
        av_reduce(&completed_out->sample_aspect_ratio.num, &completed_out->sample_aspect_ratio.den,
                  (int64_t)async_frame->frame->sample_aspect_ratio.num * outlink->h * inlink->w,
                  (int64_t)async_frame->frame->sample_aspect_ratio.den * outlink->w * inlink->h,
                  INT_MAX);
        
        ret = ff_filter_frame(outlink, completed_out);
        if (ret < 0)
            return ret;
    }

    // Get a free frame for processing
    async_frame = cuda_async_queue_get_free_frame(&s->async_queue);
    if (!async_frame) {
        // Queue is full, wait for the oldest frame to complete
        async_frame = &s->async_queue.frames[s->async_queue.head];
        ret = cuda_async_queue_wait_for_completion(&s->async_queue, async_frame);
        if (ret < 0)
            return ret;
        
        // Output the completed frame
        AVFrame* completed_out = async_frame->output_frame;
        async_frame->output_frame = NULL;
        async_frame->in_use = 0;
        s->async_queue.count--;
        s->async_queue.head = (s->async_queue.head + 1) % s->async_queue.queue_size;
        
        av_reduce(&completed_out->sample_aspect_ratio.num, &completed_out->sample_aspect_ratio.den,
                  (int64_t)async_frame->frame->sample_aspect_ratio.num * outlink->h * inlink->w,
                  (int64_t)async_frame->frame->sample_aspect_ratio.den * outlink->w * inlink->h,
                  INT_MAX);
        
        ret = ff_filter_frame(outlink, completed_out);
        if (ret < 0)
            return ret;
    }

    // Setup input frame
    av_frame_move_ref(async_frame->frame, in);
    
    // Allocate output frame buffer
    async_frame->output_frame = av_frame_alloc();
    if (!async_frame->output_frame)
        return AVERROR(ENOMEM);
    
    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
        
    ret = cudacolorspace_conv(ctx, async_frame->output_frame, async_frame->frame);
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }
    
    // Record completion event on the assigned stream
    CUstream stream = s->async_queue.streams[async_frame->stream_idx];
    ret = CHECK_CU(cu->cuEventRecord(async_frame->event_done, stream));
    if (ret < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return ret;
    }
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    
    // Submit frame to queue
    ret = cuda_async_queue_submit_frame(&s->async_queue, async_frame);
    if (ret < 0)
        return ret;
        
    return 0;
}

static int cudacolorspace_filter_frame(AVFilterLink* link, AVFrame* in)
{
    AVFilterContext* ctx = link->dst;
    CUDAColorspaceContext* s = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];
    CudaFunctions* cu = s->hwctx->internal->cuda_dl;

    AVFrame* out = NULL;
    CUcontext dummy;
    int ret = 0;

    // Use async processing if enabled
    if (s->async_enabled)
        return cudacolorspace_filter_frame_async(ctx, in);

    // Original synchronous processing
    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = cudacolorspace_conv(ctx, out, in);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    av_reduce(&out->sample_aspect_ratio.num, &out->sample_aspect_ratio.den,
              (int64_t)in->sample_aspect_ratio.num * outlink->h * link->w,
              (int64_t)in->sample_aspect_ratio.den * outlink->w * link->h,
              INT_MAX);

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(CUDAColorspaceContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    {"range", "Output video range", OFFSET(range), AV_OPT_TYPE_INT, { .i64 = AVCOL_RANGE_UNSPECIFIED }, AVCOL_RANGE_UNSPECIFIED, AVCOL_RANGE_NB - 1, FLAGS, .unit = "range"},
        {"tv",   "Limited range", 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_MPEG }, 0, 0, FLAGS, .unit = "range"},
        {"mpeg", "Limited range", 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_MPEG }, 0, 0, FLAGS, .unit = "range"},
        {"pc",   "Full range",    0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_JPEG }, 0, 0, FLAGS, .unit = "range"},
        {"jpeg", "Full range",    0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_RANGE_JPEG }, 0, 0, FLAGS, .unit = "range"},

    {"space", "Output colorspace", OFFSET(user_csp), AV_OPT_TYPE_INT, { .i64 = AVCOL_SPC_UNSPECIFIED }, AVCOL_SPC_RGB, AVCOL_SPC_NB - 1, FLAGS, .unit = "csp"},
        {"bt709",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT709 },       0, 0, FLAGS, .unit = "csp"},
        {"bt470bg",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT470BG },     0, 0, FLAGS, .unit = "csp"},
        {"smpte170m",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_SMPTE170M },   0, 0, FLAGS, .unit = "csp"},
        {"bt2020nc",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT2020_NCL },  0, 0, FLAGS, .unit = "csp"},
        {"bt2020ncl",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_BT2020_NCL },  0, 0, FLAGS, .unit = "csp"},
        {"bt601",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_SPC_SMPTE170M },   0, 0, FLAGS, .unit = "csp"},

    {"primaries", "Output color primaries", OFFSET(user_prm), AV_OPT_TYPE_INT, { .i64 = AVCOL_PRI_UNSPECIFIED }, AVCOL_PRI_RESERVED0, AVCOL_PRI_NB - 1, FLAGS, .unit = "prm"},
        {"bt709",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_BT709 },      0, 0, FLAGS, .unit = "prm"},
        {"bt470bg",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_BT470BG },    0, 0, FLAGS, .unit = "prm"},
        {"smpte170m",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_SMPTE170M },  0, 0, FLAGS, .unit = "prm"},
        {"bt2020",      NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_PRI_BT2020 },     0, 0, FLAGS, .unit = "prm"},

    {"trc", "Output transfer characteristics", OFFSET(user_trc), AV_OPT_TYPE_INT, { .i64 = AVCOL_TRC_UNSPECIFIED }, AVCOL_TRC_RESERVED0, AVCOL_TRC_NB - 1, FLAGS, .unit = "trc"},
        {"bt709",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_BT709 },      0, 0, FLAGS, .unit = "trc"},
        {"bt2020-10",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_BT2020_10 },  0, 0, FLAGS, .unit = "trc"},
        {"smpte170m",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AVCOL_TRC_SMPTE170M },  0, 0, FLAGS, .unit = "trc"},
    
    { "async_depth",  "Async frame queue depth for pipeline parallelism", OFFSET(async_depth), AV_OPT_TYPE_INT, {.i64=DEFAULT_FRAME_QUEUE_SIZE}, 1, MAX_FRAME_QUEUE_SIZE, FLAGS },
    { "async_streams", "Number of CUDA streams for concurrent processing", OFFSET(async_streams), AV_OPT_TYPE_INT, {.i64=DEFAULT_CUDA_STREAMS}, 1, MAX_CUDA_STREAMS, FLAGS },

    {NULL},
};

static const AVClass cudacolorspace_class = {
    .class_name = "colorspace_cuda",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudacolorspace_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = cudacolorspace_filter_frame,
    },
};

static const AVFilterPad cudacolorspace_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = cudacolorspace_config_props,
    },
};

const FFFilter ff_vf_colorspace_cuda = {
    .p.name        = "colorspace_cuda",
    .p.description = NULL_IF_CONFIG_SMALL("CUDA accelerated video color converter"),

    .p.priv_class  = &cudacolorspace_class,

    .init = cudacolorspace_init,
    .uninit = cudacolorspace_uninit,

    .priv_size = sizeof(CUDAColorspaceContext),

    FILTER_INPUTS(cudacolorspace_inputs),
    FILTER_OUTPUTS(cudacolorspace_outputs),

    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_CUDA),

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
