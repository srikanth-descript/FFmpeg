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

#include "heif_color.h"
#include "common.h"
#include "log.h"
#include <math.h>

#if HAVE_LCMS2
#include <lcms2.h>
#endif

void av_heif_convert_p3_to_srgb_pixel(uint8_t *r, uint8_t *g, uint8_t *b)
{
    // Convert RGB to linear values (assuming gamma 2.2 approximation)
    float rf = powf(*r / 255.0f, 2.2f);
    float gf = powf(*g / 255.0f, 2.2f);
    float bf = powf(*b / 255.0f, 2.2f);
    
    // Create conversion matrix from P3 to XYZ
    // These are pre-calculated values for Display P3 to XYZ
    float p3_to_xyz[3][3] = {
        {0.4865709f, 0.2656677f, 0.1982173f},
        {0.2289746f, 0.6917385f, 0.0792869f},
        {0.0000000f, 0.0451134f, 1.0439444f}
    };
    
    // Create conversion matrix from XYZ to sRGB
    // These are pre-calculated values for XYZ to sRGB
    float xyz_to_srgb[3][3] = {
        { 3.2404542f, -1.5371385f, -0.4985314f},
        {-0.9692660f,  1.8760108f,  0.0415560f},
        { 0.0556434f, -0.2040259f,  1.0572252f}
    };
    
    // Convert P3 RGB to XYZ
    float x = p3_to_xyz[0][0] * rf + p3_to_xyz[0][1] * gf + p3_to_xyz[0][2] * bf;
    float y = p3_to_xyz[1][0] * rf + p3_to_xyz[1][1] * gf + p3_to_xyz[1][2] * bf;
    float z = p3_to_xyz[2][0] * rf + p3_to_xyz[2][1] * gf + p3_to_xyz[2][2] * bf;
    
    // Convert XYZ to sRGB
    float r_lin = xyz_to_srgb[0][0] * x + xyz_to_srgb[0][1] * y + xyz_to_srgb[0][2] * z;
    float g_lin = xyz_to_srgb[1][0] * x + xyz_to_srgb[1][1] * y + xyz_to_srgb[1][2] * z;
    float b_lin = xyz_to_srgb[2][0] * x + xyz_to_srgb[2][1] * y + xyz_to_srgb[2][2] * z;
    
    // Apply gamma correction (inverse of 2.2) and clamp
    r_lin = FFMAX(0.0f, FFMIN(1.0f, r_lin));
    g_lin = FFMAX(0.0f, FFMIN(1.0f, g_lin));
    b_lin = FFMAX(0.0f, FFMIN(1.0f, b_lin));
    
    *r = (uint8_t)(powf(r_lin, 1.0f/2.2f) * 255.0f + 0.5f);
    *g = (uint8_t)(powf(g_lin, 1.0f/2.2f) * 255.0f + 0.5f);
    *b = (uint8_t)(powf(b_lin, 1.0f/2.2f) * 255.0f + 0.5f);
}

void av_heif_yuv_to_rgb_pixel(int Y, int U, int V, uint8_t *r, uint8_t *g, uint8_t *b)
{
    // HEIC uses full-range YUV (yuvj420p)
    // Use full range conversion
    int Y_val = Y;
    int Cb = U - 128;
    int Cr = V - 128;
    
    // BT.601 coefficients for JPEG/JFIF full-range
    // Using exact values to match libheif
    // R = Y + 1.402 * Cr
    // G = Y - 0.344136 * Cb - 0.714136 * Cr  
    // B = Y + 1.772 * Cb
    
    // Use integer arithmetic for better precision matching
    int R_i = Y_val + (1402 * Cr + 500) / 1000;
    int G_i = Y_val - (344 * Cb + 714 * Cr + 500) / 1000;
    int B_i = Y_val + (1772 * Cb + 500) / 1000;
    
    // Clamp to valid range
    *r = av_clip_uint8(R_i);
    *g = av_clip_uint8(G_i);
    *b = av_clip_uint8(B_i);
}

#if HAVE_LCMS2
int av_heif_apply_icc_profile(uint8_t *rgba_data, int width, int height,
                              const uint8_t *icc_data, size_t icc_size)
{
    cmsHPROFILE input_profile = NULL;
    cmsHPROFILE output_profile = NULL;
    cmsHTRANSFORM transform = NULL;
    int ret = 0;
    
    // Open the input ICC profile from the HEIC file
    input_profile = cmsOpenProfileFromMem(icc_data, icc_size);
    if (!input_profile) {
        av_log(NULL, AV_LOG_WARNING, "Failed to open input ICC profile\n");
        return AVERROR_EXTERNAL;
    }
    
    // Create sRGB output profile
    output_profile = cmsCreate_sRGBProfile();
    if (!output_profile) {
        av_log(NULL, AV_LOG_WARNING, "Failed to create sRGB profile\n");
        cmsCloseProfile(input_profile);
        return AVERROR_EXTERNAL;
    }
    
    // Get color space information
    cmsColorSpaceSignature input_space = cmsGetColorSpace(input_profile);
    av_log(NULL, AV_LOG_DEBUG, "Input profile color space: 0x%08X\n", input_space);
    
    // Create color transform based on input color space
    if (input_space == cmsSigGrayData) {
        // For grayscale profiles, skip transformation as we've already converted to RGB
        av_log(NULL, AV_LOG_DEBUG, "Skipping ICC transform for grayscale profile\n");
        ret = 0;
        goto cleanup;
    }
    
    transform = cmsCreateTransform(input_profile, TYPE_RGBA_8,
                                  output_profile, TYPE_RGBA_8,
                                  INTENT_PERCEPTUAL, 0);
    if (!transform) {
        av_log(NULL, AV_LOG_WARNING, "Failed to create color transform (color space: 0x%08X)\n", input_space);
        ret = AVERROR_EXTERNAL;
        goto cleanup;
    }
    
    // Apply the color transform in-place
    cmsDoTransform(transform, rgba_data, rgba_data, width * height);
    
    av_log(NULL, AV_LOG_DEBUG, "Applied ICC profile transformation to %dx%d image\n", width, height);
    
cleanup:
    if (transform)
        cmsDeleteTransform(transform);
    if (output_profile)
        cmsCloseProfile(output_profile);
    if (input_profile)
        cmsCloseProfile(input_profile);
    
    return ret;
}
#endif