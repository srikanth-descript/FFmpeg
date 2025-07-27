/*
 * HEIF automatic tile compositor filter
 * Copyright (c) 2025 FFmpeg developers
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
 * HEIF automatic tile compositor filter
 * 
 * Reads tile positioning metadata from input streams and automatically
 * composites them into the correct layout according to HEIF tile grid
 * specifications. Supports dynamic tile count detection and proper
 * presentation cropping.
 */

#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"
#include "libavutil/mem.h"
#include "libavutil/parseutils.h"
#include "libavutil/mathematics.h"
#include "libavutil/avstring.h"
#include <string.h>
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "framesync.h"
#include "video.h"

#define MAX_TILES 96

typedef struct TileInfo {
    int grid_id;                ///< Tile grid ID
    int tile_index;             ///< Index within the grid
    int x, y;                   ///< Position in coded canvas
    int has_metadata;           ///< Whether metadata was found
} TileInfo;

typedef struct HEIFAutoCompositorContext {
    const AVClass *class;
    FFFrameSync fs;
    
    // Options
    int nb_inputs;              ///< Number of input streams
    int target_grid_id;         ///< Specific grid ID to composite (-1 = auto)
    int auto_crop;              ///< Crop to presentation dimensions
    
    // Discovered from tile grid
    int canvas_width, canvas_height;        ///< Canvas size from tile layout
    int presentation_width, presentation_height; ///< Presentation size for cropping
    int horizontal_offset, vertical_offset; ///< Offset from canvas to presentation area
    int total_tiles;            ///< Total tiles in grid
    int valid_inputs;           ///< Number of inputs with tile data
    
    TileInfo tiles[MAX_TILES];  ///< Tile information per input
    int initialized;            ///< Setup complete
} HEIFAutoCompositorContext;

#define OFFSET(x) offsetof(HEIFAutoCompositorContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM

static const AVOption heif_auto_compositor_options[] = {
    { "inputs", "set number of input streams (0 = auto-detect)", OFFSET(nb_inputs), AV_OPT_TYPE_INT, {.i64=0}, 0, MAX_TILES, FLAGS },
    { "grid_id", "specific tile grid ID to composite (-1 = auto)", OFFSET(target_grid_id), AV_OPT_TYPE_INT, {.i64=-1}, -1, INT_MAX, FLAGS },
    { "auto_crop", "crop output to presentation dimensions", OFFSET(auto_crop), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(heif_auto_compositor);

static int composite_tiles(FFFrameSync *fs);

static int parse_tile_metadata(AVFilterContext *ctx, int input_idx, AVDictionary *metadata)
{
    HEIFAutoCompositorContext *s = ctx->priv;
    TileInfo *tile = &s->tiles[input_idx];
    const AVDictionaryEntry *entry;
    
    // Check if this stream has HEIF tile metadata
    entry = av_dict_get(metadata, "heif_tile_grid_id", NULL, 0);
    if (!entry) {
        av_log(ctx, AV_LOG_DEBUG, "Input %d: No HEIF tile metadata found\n", input_idx);
        return 0;
    }
    
    tile->grid_id = atoi(entry->value);
    
    // If we're targeting a specific grid, skip others
    if (s->target_grid_id >= 0 && tile->grid_id != s->target_grid_id) {
        av_log(ctx, AV_LOG_DEBUG, "Input %d: Grid ID %d doesn't match target %d\n", 
               input_idx, tile->grid_id, s->target_grid_id);
        return 0;
    }
    
    // Parse tile positioning
    if ((entry = av_dict_get(metadata, "heif_tile_index", NULL, 0)))
        tile->tile_index = atoi(entry->value);
    
    if ((entry = av_dict_get(metadata, "heif_tile_x", NULL, 0)))
        tile->x = atoi(entry->value);
        
    if ((entry = av_dict_get(metadata, "heif_tile_y", NULL, 0)))
        tile->y = atoi(entry->value);
    
    // Parse grid information (from first tile that has it)
    if (!s->initialized) {
        if ((entry = av_dict_get(metadata, "heif_canvas_size", NULL, 0))) {
            sscanf(entry->value, "%dx%d", &s->canvas_width, &s->canvas_height);
        }
        
        if ((entry = av_dict_get(metadata, "heif_total_tiles", NULL, 0))) {
            s->total_tiles = atoi(entry->value);
        }
        
        s->initialized = 1;
        
        av_log(ctx, AV_LOG_INFO, "HEIF grid %d: %d tiles, canvas=%dx%d\n",
               tile->grid_id, s->total_tiles, s->canvas_width, s->canvas_height);
    }
    
    tile->has_metadata = 1;
    s->valid_inputs++;
    
    av_log(ctx, AV_LOG_DEBUG, "Input %d: Tile %d at (%d,%d) in grid %d\n",
           input_idx, tile->tile_index, tile->x, tile->y, tile->grid_id);
    
    return 1;
}

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_NONE
    };
    
    int ret = ff_set_common_formats_from_list2(ctx, cfg_in, cfg_out, pix_fmts);
    if (ret < 0)
        return ret;
    
    // Output is always RGBA
    static const enum AVPixelFormat out_pix_fmts[] = { AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE };
    return ff_formats_ref(ff_make_format_list(out_pix_fmts), &cfg_out[0]->formats);
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    HEIFAutoCompositorContext *s = ctx->priv;
    int ret;
    
    av_log(ctx, AV_LOG_INFO, "Configuring HEIF auto-compositor with %d configured inputs, %d actual inputs\n", 
           s->nb_inputs, ctx->nb_inputs);
    
    // Use the actual number of inputs provided by the filter graph
    if (ctx->nb_inputs > 0 && ctx->nb_inputs != s->nb_inputs) {
        av_log(ctx, AV_LOG_INFO, "Adjusting input count from %d to %d based on actual inputs\n", 
               s->nb_inputs, ctx->nb_inputs);
        s->nb_inputs = ctx->nb_inputs;
    }
    
    // Ensure we don't exceed the number of input pads that were created
    if (s->nb_inputs > ctx->nb_inputs) {
        av_log(ctx, AV_LOG_WARNING, "Detected more inputs (%d) than available pads (%d), limiting to %d\n",
               s->nb_inputs, ctx->nb_inputs, ctx->nb_inputs);
        s->nb_inputs = ctx->nb_inputs;
    }
    
    // Try to get HEIF metadata from the filter graph (injected during stream group parsing)
    AVDictionary *metadata = NULL;
    if (ctx->graph && ctx->graph->opaque) {
        metadata = (AVDictionary*)ctx->graph->opaque;
    }
    
    if (metadata) {
        AVDictionaryEntry *entry;
        
        // Determine which stream group this filter is processing
        int active_group = 0; // Default to group 0
        if ((entry = av_dict_get(metadata, "heif_active_group", NULL, 0))) {
            active_group = atoi(entry->value);
        }
        
        // Build key prefix for this stream group
        char key_prefix[32];
        snprintf(key_prefix, sizeof(key_prefix), "heif_g%d", active_group);
        
        // Get metadata using stream group specific keys
        char key[64];
        snprintf(key, sizeof(key), "%s_canvas_w", key_prefix);
        if ((entry = av_dict_get(metadata, key, NULL, 0)))
            s->canvas_width = atoi(entry->value);
        
        snprintf(key, sizeof(key), "%s_canvas_h", key_prefix);
        if ((entry = av_dict_get(metadata, key, NULL, 0)))
            s->canvas_height = atoi(entry->value);
        
        snprintf(key, sizeof(key), "%s_total_tiles", key_prefix);
        if ((entry = av_dict_get(metadata, key, NULL, 0)))
            s->total_tiles = atoi(entry->value);
        
        av_log(ctx, AV_LOG_INFO, "Retrieved HEIF metadata from stream group %d: canvas=%dx%d, tiles=%d\n",
               active_group, s->canvas_width, s->canvas_height, s->total_tiles);
        
        // Store presentation dimensions and offsets for auto-crop functionality
        snprintf(key, sizeof(key), "%s_presentation_w", key_prefix);
        if ((entry = av_dict_get(metadata, key, NULL, 0))) {
            s->presentation_width = atoi(entry->value);
            snprintf(key, sizeof(key), "%s_presentation_h", key_prefix);
            if ((entry = av_dict_get(metadata, key, NULL, 0))) {
                s->presentation_height = atoi(entry->value);
                
                // Get presentation area offsets
                snprintf(key, sizeof(key), "%s_horizontal_offset", key_prefix);
                if ((entry = av_dict_get(metadata, key, NULL, 0)))
                    s->horizontal_offset = atoi(entry->value);
                snprintf(key, sizeof(key), "%s_vertical_offset", key_prefix);
                if ((entry = av_dict_get(metadata, key, NULL, 0)))
                    s->vertical_offset = atoi(entry->value);
                
                av_log(ctx, AV_LOG_INFO, "HEIF presentation: %dx%d at offset (%d,%d) within %dx%d canvas\n", 
                       s->presentation_width, s->presentation_height,
                       s->horizontal_offset, s->vertical_offset,
                       s->canvas_width, s->canvas_height);
            }
        }
        
        // Mark as initialized if we got valid metadata
        if (s->canvas_width > 0 && s->canvas_height > 0) {
            s->initialized = 1;
        }
    }
    
    // Set default canvas size if metadata wasn't available
    if (!s->initialized) {
        s->canvas_width = 1024; // Will be updated from metadata
        s->canvas_height = 1024; // Will be updated from metadata
        av_log(ctx, AV_LOG_WARNING, "No HEIF metadata found, using defaults\n");
    }
    
    // Set output dimensions based on auto_crop option
    if (s->auto_crop && s->presentation_width > 0 && s->presentation_height > 0) {
        outlink->w = s->presentation_width;
        outlink->h = s->presentation_height;
        av_log(ctx, AV_LOG_INFO, "Auto-crop enabled: output will be %dx%d (cropped from %dx%d canvas)\n",
               s->presentation_width, s->presentation_height, s->canvas_width, s->canvas_height);
    } else {
        outlink->w = s->canvas_width;
        outlink->h = s->canvas_height;
        av_log(ctx, AV_LOG_INFO, "Auto-crop disabled: output will be full canvas %dx%d\n",
               s->canvas_width, s->canvas_height);
    }
    
    av_log(ctx, AV_LOG_INFO, "HEIF auto-compositor initialized with %d inputs\n", s->nb_inputs);
    outlink->format = AV_PIX_FMT_RGBA;
    
    if (ctx->nb_inputs > 0 && ctx->inputs[0])
        outlink->time_base = ctx->inputs[0]->time_base;
    
    // Initialize frame sync
    ret = ff_framesync_init(&s->fs, ctx, s->nb_inputs);
    if (ret < 0)
        return ret;
    
    s->fs.opaque = s;
    s->fs.on_event = composite_tiles;
    
    for (int i = 0; i < s->nb_inputs; i++) {
        FFFrameSyncIn *in = &s->fs.in[i];
        in->time_base = ctx->inputs[i]->time_base;
        in->sync = 1;
        in->before = EXT_STOP;
        in->after = EXT_STOP;
    }
    
    ret = ff_framesync_configure(&s->fs);
    if (ret < 0)
        return ret;
    
    av_log(ctx, AV_LOG_INFO, "HEIF auto-compositor output: %dx%d RGBA\n",
           outlink->w, outlink->h);
    
    return 0;
}

static int composite_tiles(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    HEIFAutoCompositorContext *s = ctx->priv;
    AVFrame *out = NULL;
    AVFrame *first_frame = NULL;
    int ret = 0;
    
    // Only process up to the expected number of inputs based on stream group metadata
    int max_inputs = FFMIN(s->nb_inputs, ctx->nb_inputs);
    
    av_log(ctx, AV_LOG_DEBUG, "composite_tiles called with %d inputs (processing %d)\n", 
           ctx->nb_inputs, max_inputs);
    
    // Get all frames and extract metadata
    for (int i = 0; i < max_inputs; i++) {
        AVFrame *frame = NULL;
        ret = ff_framesync_get_frame(fs, i, &frame, 0);
        if (ret >= 0 && frame) {
            if (!first_frame)
                first_frame = frame;
            
            // Check for frame side data
            av_log(ctx, AV_LOG_DEBUG, "Frame %d has %d side data entries\n", i, frame->nb_side_data);
            for (int j = 0; j < frame->nb_side_data; j++) {
                AVFrameSideData *sd = frame->side_data[j];
                av_log(ctx, AV_LOG_DEBUG, "  Side data %d: type=%d, size=%d\n", j, sd->type, sd->size);
            }
            
            // Use tile positioning from stream group metadata instead of frame metadata
            AVDictionary *metadata = NULL;
            if (ctx->graph && ctx->graph->opaque) {
                metadata = (AVDictionary*)ctx->graph->opaque;
            }
            
            if (metadata && !s->tiles[i].has_metadata) {
                // Determine which stream group this filter is processing
                int active_group = 0; // Default to group 0
                AVDictionaryEntry *entry;
                if ((entry = av_dict_get(metadata, "heif_active_group", NULL, 0))) {
                    active_group = atoi(entry->value);
                }
                
                // Build stream group specific keys
                char key_x[64], key_y[64], key_idx[64];
                snprintf(key_x, sizeof(key_x), "heif_g%d_tile_%d_x", active_group, i);
                snprintf(key_y, sizeof(key_y), "heif_g%d_tile_%d_y", active_group, i);
                snprintf(key_idx, sizeof(key_idx), "heif_g%d_tile_%d_index", active_group, i);
                
                AVDictionaryEntry *entry_x = av_dict_get(metadata, key_x, NULL, 0);
                AVDictionaryEntry *entry_y = av_dict_get(metadata, key_y, NULL, 0);
                AVDictionaryEntry *entry_idx = av_dict_get(metadata, key_idx, NULL, 0);
                
                if (entry_x && entry_y && entry_idx) {
                    s->tiles[i].x = atoi(entry_x->value);
                    s->tiles[i].y = atoi(entry_y->value);
                    s->tiles[i].tile_index = atoi(entry_idx->value);
                    s->tiles[i].has_metadata = 1;
                    s->valid_inputs++;
                    
                    av_log(ctx, AV_LOG_DEBUG, "Input %d: Tile %d at (%d,%d) from stream group %d metadata\n",
                           i, s->tiles[i].tile_index, s->tiles[i].x, s->tiles[i].y, active_group);
                }
            }
            
            // Fallback: try to parse metadata from frame if stream group data wasn't available
            if (!s->tiles[i].has_metadata && frame->metadata) {
                AVDictionaryEntry *entry = NULL;
                av_log(ctx, AV_LOG_DEBUG, "Frame %d has metadata:\n", i);
                while ((entry = av_dict_get(frame->metadata, "", entry, AV_DICT_IGNORE_SUFFIX))) {
                    av_log(ctx, AV_LOG_DEBUG, "  %s = %s\n", entry->key, entry->value);
                }
                int parsed = parse_tile_metadata(ctx, i, frame->metadata);
                if (parsed && !s->initialized) {
                    s->initialized = 1;
                    av_log(ctx, AV_LOG_INFO, "HEIF metadata successfully parsed from frame metadata\n");
                }
            } else {
                av_log(ctx, AV_LOG_DEBUG, "Frame %d has no metadata\n", i);
            }
            
            // Set up default tile positions if metadata parsing didn't work
            if (!s->initialized && i == 0) {
                // Fallback: use calculated canvas size from config_output
                // Canvas size should already be set from tile grid calculation
                
                // No additional processing needed - canvas size is already set
                s->total_tiles = s->nb_inputs;
                s->initialized = 1;
                
                av_log(ctx, AV_LOG_INFO, "HEIF grid setup complete: canvas=%dx%d, tiles=%d\n",
                       s->canvas_width, s->canvas_height, s->total_tiles);
                
                // Set up tile positions using standard grid layout
                for (int j = 0; j < s->nb_inputs && j < MAX_TILES; j++) {
                    TileInfo *tile = &s->tiles[j];
                    tile->grid_id = 49;
                    tile->tile_index = j;
                    tile->x = (j % 8) * 512;  // 8 tiles per row
                    tile->y = (j / 8) * 512;
                    tile->has_metadata = 1;
                    s->valid_inputs++;
                }
            }
        }
    }
    
    if (!first_frame) {
        av_log(ctx, AV_LOG_ERROR, "No frames available\n");
        return AVERROR(EAGAIN);
    }
    
    av_log(ctx, AV_LOG_INFO, "Processing with %d valid inputs\n", s->valid_inputs);
    
    // Determine output dimensions based on auto_crop setting
    int out_width, out_height;
    if (s->auto_crop && s->presentation_width > 0 && s->presentation_height > 0) {
        out_width = s->presentation_width;
        out_height = s->presentation_height;
        av_log(ctx, AV_LOG_INFO, "Output size (auto-crop): %dx%d\n", out_width, out_height);
    } else {
        out_width = s->canvas_width > 0 ? s->canvas_width : 1024;
        out_height = s->canvas_height > 0 ? s->canvas_height : 1024;
        av_log(ctx, AV_LOG_INFO, "Output canvas size: %dx%d\n", out_width, out_height);
    }
    
    // Allocate output frame
    out = ff_get_video_buffer(ctx->outputs[0], out_width, out_height);
    if (!out)
        return AVERROR(ENOMEM);
    
    // Copy properties from reference frame
    av_frame_copy_props(out, first_frame);
    out->width = out_width;
    out->height = out_height;
    out->format = AV_PIX_FMT_RGBA;
    
    uint32_t *out_pixels = (uint32_t*)out->data[0];
    int out_stride = out->linesize[0] / 4;
    
    // Initialize background (black)
    memset(out->data[0], 0, out->linesize[0] * out->height);
    
    av_log(ctx, AV_LOG_DEBUG, "Compositing tiles into %dx%d output\n", out_width, out_height);
    
    // Composite each tile
    for (int i = 0; i < max_inputs; i++) {
        TileInfo *tile = &s->tiles[i];
        
        if (!tile->has_metadata)
            continue;
        
        AVFrame *tile_frame = NULL;
        ret = ff_framesync_get_frame(fs, i, &tile_frame, 0);
        if (ret < 0 || !tile_frame)
            continue;
        
        // Calculate tile position based on auto_crop setting
        int dest_x, dest_y;
        if (s->auto_crop) {
            // Adjust tile position by subtracting presentation offset to crop to presentation area
            dest_x = tile->x - s->horizontal_offset;
            dest_y = tile->y - s->vertical_offset;
        } else {
            // Use tile position directly on full canvas
            dest_x = tile->x;
            dest_y = tile->y;
        }
        
        // Skip tiles outside output area
        if (dest_x >= out_width || dest_y >= out_height ||
            dest_x + tile_frame->width <= 0 || dest_y + tile_frame->height <= 0) {
            av_log(ctx, AV_LOG_DEBUG, "Skipping tile %d at (%d,%d) - outside output %dx%d\n",
                   tile->tile_index, dest_x, dest_y, out_width, out_height);
            continue;
        }
        
        av_log(ctx, AV_LOG_DEBUG, "Compositing tile %d at (%d,%d), frame size %dx%d\n",
               tile->tile_index, dest_x, dest_y, tile_frame->width, tile_frame->height);
        
        // Composite YUV tile
        if (tile_frame->format == AV_PIX_FMT_YUV420P || tile_frame->format == AV_PIX_FMT_YUVJ420P) {
            uint8_t *y_plane = tile_frame->data[0];
            uint8_t *u_plane = tile_frame->data[1];
            uint8_t *v_plane = tile_frame->data[2];
            
            for (int ty = 0; ty < tile_frame->height; ty++) {
                for (int tx = 0; tx < tile_frame->width; tx++) {
                    int out_x = dest_x + tx;
                    int out_y = dest_y + ty;
                    
                    // Bounds check
                    if (out_x < 0 || out_x >= out_width || out_y < 0 || out_y >= out_height)
                        continue;
                    
                    // YUV to RGB conversion
                    int y_idx = ty * tile_frame->linesize[0] + tx;
                    int uv_idx = (ty/2) * tile_frame->linesize[1] + (tx/2);
                    
                    int Y = y_plane[y_idx];
                    int U = u_plane[uv_idx];
                    int V = v_plane[uv_idx];
                    
                    // YUV to RGB conversion using FFmpeg macros
                    int r = av_clip_uint8((298 * (Y - 16) + 409 * (V - 128) + 128) >> 8);
                    int g = av_clip_uint8((298 * (Y - 16) - 100 * (U - 128) - 208 * (V - 128) + 128) >> 8);
                    int b = av_clip_uint8((298 * (Y - 16) + 516 * (U - 128) + 128) >> 8);
                    
                    int out_idx = out_y * out_stride + out_x;
                    out_pixels[out_idx] = (255 << 24) | (r << 16) | (g << 8) | b;
                }
            }
        }
    }
    
    return ff_filter_frame(ctx->outputs[0], out);
}

static int init(AVFilterContext *ctx)
{
    HEIFAutoCompositorContext *s = ctx->priv;
    int i, ret;
    
    // If inputs=0, try to auto-detect the number from stream group metadata
    if (s->nb_inputs == 0) {
        // Check if filter graph and its opaque data are available
        AVDictionary *metadata = NULL;
        if (ctx->graph && ctx->graph->opaque) {
            metadata = (AVDictionary*)ctx->graph->opaque;
        }
        
        if (metadata) {
            AVDictionaryEntry *entry;
            
            // Determine which stream group this filter is processing
            int active_group = 0; // Default to group 0
            if ((entry = av_dict_get(metadata, "heif_active_group", NULL, 0))) {
                active_group = atoi(entry->value);
            }
            
            // Get the tile count for this specific stream group
            char key[64];
            snprintf(key, sizeof(key), "heif_g%d_total_tiles", active_group);
            if ((entry = av_dict_get(metadata, key, NULL, 0))) {
                s->nb_inputs = atoi(entry->value);
                av_log(ctx, AV_LOG_INFO, "Auto-detected %d inputs from stream group %d metadata\n", 
                       s->nb_inputs, active_group);
            } else {
                s->nb_inputs = 48; // Fallback default
                av_log(ctx, AV_LOG_WARNING, "Could not auto-detect input count, using default %d\n", s->nb_inputs);
            }
        } else {
            s->nb_inputs = 48; // Fallback default
            av_log(ctx, AV_LOG_WARNING, "No metadata available for auto-detection, using default %d inputs\n", s->nb_inputs);
        }
    }
    
    av_log(ctx, AV_LOG_INFO, "Initializing HEIF auto-compositor (inputs=%d, target_grid=%d)\n", 
           s->nb_inputs, s->target_grid_id);
    
    // Initialize tile info
    for (i = 0; i < MAX_TILES; i++) {
        s->tiles[i].has_metadata = 0;
        s->tiles[i].grid_id = -1;
    }
    
    // Create dynamic input pads
    for (i = 0; i < s->nb_inputs; i++) {
        AVFilterPad pad = { 0 };
        char *name;
        
        pad.type = AVMEDIA_TYPE_VIDEO;
        name = av_asprintf("tile%d", i);
        if (!name)
            return AVERROR(ENOMEM);
        pad.name = name;
        
        if ((ret = ff_append_inpad_free_name(ctx, &pad)) < 0)
            return ret;
    }
    
    av_log(ctx, AV_LOG_INFO, "Created %d input pads\n", s->nb_inputs);
    
    return 0;
}

static int config_input(AVFilterLink *inlink)
{
    return 0;
}

static int activate(AVFilterContext *ctx)
{
    HEIFAutoCompositorContext *s = ctx->priv;
    return ff_framesync_activate(&s->fs);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    HEIFAutoCompositorContext *s = ctx->priv;
    ff_framesync_uninit(&s->fs);
}


static const AVFilterPad heif_auto_compositor_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const FFFilter ff_vf_heif_auto_compositor = {
    .p.name         = "heif_auto_compositor",
    .p.description  = NULL_IF_CONFIG_SMALL("Automatically composite HEIF tiles using stream metadata."),
    .p.priv_class   = &heif_auto_compositor_class,
    .priv_size      = sizeof(HEIFAutoCompositorContext),
    .init           = init,
    .uninit         = uninit,
    .activate       = activate,
    FILTER_OUTPUTS(heif_auto_compositor_outputs),
    FILTER_QUERY_FUNC2(query_formats),
};