/*
 * Copyright (c) 2025 Claude Code Assistant
 * Based on the original showinfo filter by Stefano Sabatini
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * CUDA-compatible filter for showing textual video frame information
 */

#include <ctype.h>
#include <inttypes.h>

#include "libavutil/buffer.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/bswap.h"
#include "libavutil/adler32.h"
#include "libavutil/display.h"
#include "libavutil/dovi_meta.h"
#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/film_grain_params.h"
#include "libavutil/hdr_dynamic_metadata.h"
#include "libavutil/hdr_dynamic_vivid_metadata.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/spherical.h"
#include "libavutil/stereo3d.h"
#include "libavutil/timestamp.h"
#include "libavutil/timecode.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/video_enc_params.h"
#include "libavutil/detection_bbox.h"
#include "libavutil/ambient_viewing_environment.h"
#include "libavutil/uuid.h"

#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct ShowInfoCudaContext {
    const AVClass *class;
    
    AVCUDADeviceContext *hwctx;
    AVBufferRef *hw_frames_ctx;
    
    int calculate_checksums;
    int udu_sei_as_ascii;
    
    AVFrame *tmp_frame;
} ShowInfoCudaContext;

#define OFFSET(x) offsetof(ShowInfoCudaContext, x)
#define VF AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption showinfo_cuda_options[] = {
    { "checksum", "calculate checksums", OFFSET(calculate_checksums), AV_OPT_TYPE_BOOL, {.i64=1}, 0, 1, VF },
    { "udu_sei_as_ascii", "try to print user data unregistered SEI as ascii character when possible",
        OFFSET(udu_sei_as_ascii), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VF },
    { NULL }
};

AVFILTER_DEFINE_CLASS(showinfo_cuda);

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_P010,
    AV_PIX_FMT_P016,
    AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_0RGB32,
    AV_PIX_FMT_0BGR32,
    AV_PIX_FMT_RGB32,
    AV_PIX_FMT_BGR32,
};

static int format_is_supported(enum AVPixelFormat fmt)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static int showinfo_cuda_query_formats(const AVFilterContext *avctx,
                                       AVFilterFormatsConfig **cfg_in,
                                       AVFilterFormatsConfig **cfg_out)
{
    int err;

    if ((err = ff_formats_ref(ff_formats_pixdesc_filter(AV_PIX_FMT_FLAG_HWACCEL, 0),
                              &cfg_in[0]->formats)) < 0)
        return err;

    if ((err = ff_formats_ref(ff_formats_pixdesc_filter(AV_PIX_FMT_FLAG_HWACCEL, 0),
                              &cfg_out[0]->formats)) < 0)
        return err;

    return 0;
}

static int showinfo_cuda_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    ShowInfoCudaContext *s = ctx->priv;
    FilterLink *il = ff_filter_link(inlink);
    FilterLink *ol = ff_filter_link(outlink);
    AVHWFramesContext *frames_ctx;

    av_buffer_unref(&s->hw_frames_ctx);

    if (!il->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }

    s->hw_frames_ctx = av_buffer_ref(il->hw_frames_ctx);
    if (!s->hw_frames_ctx)
        return AVERROR(ENOMEM);

    frames_ctx = (AVHWFramesContext*)s->hw_frames_ctx->data;
    s->hwctx = frames_ctx->device_ctx->hwctx;

    if (!format_is_supported(frames_ctx->sw_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n", 
               av_get_pix_fmt_name(frames_ctx->sw_format));
        return AVERROR(ENOSYS);
    }

    ol->hw_frames_ctx = av_buffer_ref(s->hw_frames_ctx);
    if (!ol->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

// Include all the dump functions from the original showinfo filter
// (These functions work on metadata which doesn't require CUDA processing)

static void dump_spherical(AVFilterContext *ctx, AVFrame *frame, const AVFrameSideData *sd)
{
    const AVSphericalMapping *spherical = (const AVSphericalMapping *)sd->data;
    double yaw, pitch, roll;

    av_log(ctx, AV_LOG_INFO, "%s ", av_spherical_projection_name(spherical->projection));

    if (spherical->yaw || spherical->pitch || spherical->roll) {
        yaw = ((double)spherical->yaw) / (1 << 16);
        pitch = ((double)spherical->pitch) / (1 << 16);
        roll = ((double)spherical->roll) / (1 << 16);
        av_log(ctx, AV_LOG_INFO, "(%f/%f/%f) ", yaw, pitch, roll);
    }

    if (spherical->projection == AV_SPHERICAL_EQUIRECTANGULAR_TILE) {
        size_t l, t, r, b;
        av_spherical_tile_bounds(spherical, frame->width, frame->height,
                                 &l, &t, &r, &b);
        av_log(ctx, AV_LOG_INFO,
               "[%"SIZE_SPECIFIER", %"SIZE_SPECIFIER", %"SIZE_SPECIFIER", %"SIZE_SPECIFIER"] ",
               l, t, r, b);
    } else if (spherical->projection == AV_SPHERICAL_CUBEMAP) {
        av_log(ctx, AV_LOG_INFO, "[pad %"PRIu32"] ", spherical->padding);
    }
}

static void dump_color_property(AVFilterContext *ctx, AVFrame *frame)
{
    const char *color_range_str     = av_color_range_name(frame->color_range);
    const char *colorspace_str      = av_color_space_name(frame->colorspace);
    const char *color_primaries_str = av_color_primaries_name(frame->color_primaries);
    const char *color_trc_str       = av_color_transfer_name(frame->color_trc);

    if (!color_range_str || frame->color_range == AVCOL_RANGE_UNSPECIFIED) {
        av_log(ctx, AV_LOG_INFO, "color_range:unknown");
    } else {
        av_log(ctx, AV_LOG_INFO, "color_range:%s", color_range_str);
    }

    if (!colorspace_str || frame->colorspace == AVCOL_SPC_UNSPECIFIED) {
        av_log(ctx, AV_LOG_INFO, " color_space:unknown");
    } else {
        av_log(ctx, AV_LOG_INFO, " color_space:%s", colorspace_str);
    }

    if (!color_primaries_str || frame->color_primaries == AVCOL_PRI_UNSPECIFIED) {
        av_log(ctx, AV_LOG_INFO, " color_primaries:unknown");
    } else {
        av_log(ctx, AV_LOG_INFO, " color_primaries:%s", color_primaries_str);
    }

    if (!color_trc_str || frame->color_trc == AVCOL_TRC_UNSPECIFIED) {
        av_log(ctx, AV_LOG_INFO, " color_trc:unknown");
    } else {
        av_log(ctx, AV_LOG_INFO, " color_trc:%s", color_trc_str);
    }
    av_log(ctx, AV_LOG_INFO, "\n");
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    FilterLink *inl = ff_filter_link(inlink);
    AVFilterContext *ctx = inlink->dst;
    ShowInfoCudaContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc;
    AVHWFramesContext *frames_ctx;
    uint32_t plane_checksum[4] = {0}, checksum = 0;
    int32_t pixelcount[4] = {0};
    int err, i;

    if (!frame->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "Input frame missing hw_frames_ctx\n");
        return AVERROR(EINVAL);
    }

    frames_ctx = (AVHWFramesContext*)frame->hw_frames_ctx->data;
    desc = av_pix_fmt_desc_get(frames_ctx->sw_format);

    // For CUDA frames, we show info without downloading unless checksums are requested
    av_log(ctx, AV_LOG_INFO,
           "n:%4"PRId64" pts:%7s pts_time:%-7s duration:%7"PRId64
           " duration_time:%-7s "
           "fmt:%s(cuda) cl:%s sar:%d/%d s:%dx%d i:%c iskey:%d type:%c ",
           inl->frame_count_out,
           av_ts2str(frame->pts), av_ts2timestr(frame->pts, &inlink->time_base),
           frame->duration, av_ts2timestr(frame->duration, &inlink->time_base),
           desc->name, av_chroma_location_name(frame->chroma_location),
           frame->sample_aspect_ratio.num, frame->sample_aspect_ratio.den,
           frame->width, frame->height,
           !(frame->flags & AV_FRAME_FLAG_INTERLACED)     ? 'P' :         /* Progressive  */
           (frame->flags & AV_FRAME_FLAG_TOP_FIELD_FIRST) ? 'T' : 'B',    /* Top / Bottom */
           !!(frame->flags & AV_FRAME_FLAG_KEY),
           av_get_picture_type_char(frame->pict_type));

    // If checksums are requested, we need to download the frame
    if (s->calculate_checksums) {
        if (!s->tmp_frame) {
            s->tmp_frame = av_frame_alloc();
            if (!s->tmp_frame)
                return AVERROR(ENOMEM);
        }

        err = av_hwframe_transfer_data(s->tmp_frame, frame, 0);
        if (err < 0) {
            av_log(ctx, AV_LOG_WARNING, "Failed to download frame for checksum calculation: %s\n", 
                   av_err2str(err));
            av_log(ctx, AV_LOG_INFO, "checksum:N/A");
        } else {
            // Calculate checksums on downloaded frame
            int bitdepth = desc->comp[0].depth;
            int plane, vsub = desc->log2_chroma_h;

            for (plane = 0; plane < 4 && s->tmp_frame->data[plane] && s->tmp_frame->linesize[plane]; plane++) {
                uint8_t *data = s->tmp_frame->data[plane];
                int h = plane == 1 || plane == 2 ? AV_CEIL_RSHIFT(inlink->h, vsub) : inlink->h;
                int linesize = av_image_get_linesize(s->tmp_frame->format, s->tmp_frame->width, plane);
                int width = linesize >> (bitdepth > 8);

                if (linesize < 0) {
                    av_frame_unref(s->tmp_frame);
                    return linesize;
                }

                for (i = 0; i < h; i++) {
                    plane_checksum[plane] = av_adler32_update(plane_checksum[plane], data, linesize);
                    checksum = av_adler32_update(checksum, data, linesize);
                    pixelcount[plane] += width;
                    data += s->tmp_frame->linesize[plane];
                }
            }

            av_log(ctx, AV_LOG_INFO,
                   "checksum:%08"PRIX32" plane_checksum:[%08"PRIX32,
                   checksum, plane_checksum[0]);

            for (plane = 1; plane < 4 && s->tmp_frame->data[plane] && s->tmp_frame->linesize[plane]; plane++)
                av_log(ctx, AV_LOG_INFO, " %08"PRIX32, plane_checksum[plane]);
            av_log(ctx, AV_LOG_INFO, "]");
        }

        av_frame_unref(s->tmp_frame);
    }

    av_log(ctx, AV_LOG_INFO, "\n");

    // Show side data (this doesn't require frame data download)
    for (i = 0; i < frame->nb_side_data; i++) {
        AVFrameSideData *sd = frame->side_data[i];
        const char *name = av_frame_side_data_name(sd->type);

        av_log(ctx, AV_LOG_INFO, "  side data - ");
        if (name)
            av_log(ctx, AV_LOG_INFO, "%s: ", name);
        
        switch (sd->type) {
        case AV_FRAME_DATA_SPHERICAL:
            dump_spherical(ctx, frame, sd);
            break;
        case AV_FRAME_DATA_DISPLAYMATRIX:
            av_log(ctx, AV_LOG_INFO, "rotation of %.2f degrees",
                   av_display_rotation_get((int32_t *)sd->data));
            break;
        case AV_FRAME_DATA_AFD:
            av_log(ctx, AV_LOG_INFO, "value of %"PRIu8, sd->data[0]);
            break;
        default:
            if (name)
                av_log(ctx, AV_LOG_INFO,
                       "(%"SIZE_SPECIFIER" bytes)", sd->size);
            else
                av_log(ctx, AV_LOG_WARNING, "unknown side data type %d "
                       "(%"SIZE_SPECIFIER" bytes)", sd->type, sd->size);
            break;
        }

        av_log(ctx, AV_LOG_INFO, "\n");
    }

    dump_color_property(ctx, frame);

    return ff_filter_frame(inlink->dst->outputs[0], frame);
}

static av_cold void showinfo_cuda_uninit(AVFilterContext *ctx)
{
    ShowInfoCudaContext *s = ctx->priv;

    av_frame_free(&s->tmp_frame);
    av_buffer_unref(&s->hw_frames_ctx);
}

static const AVFilterPad showinfo_cuda_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
};

static const AVFilterPad showinfo_cuda_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = showinfo_cuda_config_props,
    },
};

const AVFilter ff_vf_showinfo_cuda = {
    .name        = "showinfo_cuda",
    .description = NULL_IF_CONFIG_SMALL("Show textual information for each CUDA video frame."),
    .priv_class  = &showinfo_cuda_class,
    .flags       = AVFILTER_FLAG_METADATA_ONLY,
    .priv_size     = sizeof(ShowInfoCudaContext),
    .uninit        = showinfo_cuda_uninit,
    FILTER_INPUTS(showinfo_cuda_inputs),
    FILTER_OUTPUTS(showinfo_cuda_outputs),
    FILTER_QUERY_FUNC2(showinfo_cuda_query_formats),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};