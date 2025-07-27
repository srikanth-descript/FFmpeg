/*
 * HEIF color space conversion utilities
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

#ifndef AVUTIL_HEIF_COLOR_H
#define AVUTIL_HEIF_COLOR_H

#include <stdint.h>
#include <stddef.h>

/**
 * Convert a single RGB pixel from Display P3 to sRGB color space.
 * 
 * @param r pointer to red component (0-255), will be modified in place
 * @param g pointer to green component (0-255), will be modified in place  
 * @param b pointer to blue component (0-255), will be modified in place
 */
void av_heif_convert_p3_to_srgb_pixel(uint8_t *r, uint8_t *g, uint8_t *b);

/**
 * Convert YUV pixel to RGB using HEIF/JPEG full-range conversion.
 * Matches libheif's conversion exactly for compatibility.
 * 
 * @param Y  luma component (0-255)
 * @param U  chroma U component (0-255) 
 * @param V  chroma V component (0-255)
 * @param r  pointer to output red component (0-255)
 * @param g  pointer to output green component (0-255)
 * @param b  pointer to output blue component (0-255)
 */
void av_heif_yuv_to_rgb_pixel(int Y, int U, int V, uint8_t *r, uint8_t *g, uint8_t *b);

#if HAVE_LCMS2
/**
 * Apply ICC color profile transformation to RGBA image data.
 * 
 * @param rgba_data  image data in RGBA format, modified in place
 * @param width      image width in pixels
 * @param height     image height in pixels
 * @param icc_data   ICC profile data
 * @param icc_size   size of ICC profile data in bytes
 * @return           0 on success, negative value on error
 */
int av_heif_apply_icc_profile(uint8_t *rgba_data, int width, int height,
                              const uint8_t *icc_data, size_t icc_size);
#endif

#endif /* AVUTIL_HEIF_COLOR_H */