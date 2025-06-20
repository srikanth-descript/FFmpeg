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

#include "cuda/vector_helpers.cuh"

#define BLOCKX 32
#define BLOCKY 16

extern "C" {

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

__device__ static inline float hable_cuda(float in)
{
    float a = 0.15f, b = 0.50f, c = 0.10f, d = 0.20f, e = 0.02f, f = 0.30f;
    return (in * (in * a + b * c) + d * e) / (in * (in * a + b) + d * f) - e / f;
}

__device__ static inline float mobius_cuda(float in, float j, float peak)
{
    float a, b;

    if (in <= j)
        return in;

    a = -j * j * (peak - 1.0f) / (j * j - 2.0f * j + peak);
    b = (j * j - 2.0f * j * peak + peak) / fmaxf(peak - 1.0f, 1e-6f);

    return (b * b + 2.0f * b * j + j * j) / (b - a) * (in + a) / (in + b);
}

__device__ static inline float apply_tonemap_cuda(int algorithm, float sig, float param, float peak)
{
    switch (algorithm) {
    case TONEMAP_NONE:
        return sig;
    case TONEMAP_LINEAR:
        return sig * param / peak;
    case TONEMAP_GAMMA:
        return sig > 0.05f ? powf(sig / peak, 1.0f / param)
                           : sig * powf(0.05f / peak, 1.0f / param) / 0.05f;
    case TONEMAP_CLIP:
        return fminf(fmaxf(sig * param, 0.0f), 1.0f);
    case TONEMAP_HABLE:
        return hable_cuda(sig) / hable_cuda(peak);
    case TONEMAP_REINHARD:
        return sig / (sig + param) * (peak + param) / peak;
    case TONEMAP_MOBIUS:
        return mobius_cuda(sig, param, peak);
    default:
        return sig;
    }
}

__global__ void tonemap_cuda_float(
    cudaTextureObject_t src_tex,
    float *dst_r, float *dst_g, float *dst_b,
    int dst_width, int dst_height, int dst_pitch,
    int algorithm, float param, float desat, float peak,
    float coeff_r, float coeff_g, float coeff_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    float4 src = tex2D<float4>(src_tex, x, y);
    float r = src.x;
    float g = src.y;
    float b = src.z;
    
    // Desaturate to prevent unnatural colors
    if (desat > 0) {
        float luma = coeff_r * r + coeff_g * g + coeff_b * b;
        float overbright = fmaxf(luma - desat, 1e-6f) / fmaxf(luma, 1e-6f);
        r = r * (1.0f - overbright) + luma * overbright;
        g = g * (1.0f - overbright) + luma * overbright;
        b = b * (1.0f - overbright) + luma * overbright;
    }
    
    // Pick the brightest component
    float sig = fmaxf(fmaxf(fmaxf(r, g), b), 1e-6f);
    float sig_orig = sig;
    
    // Apply tonemap algorithm
    sig = apply_tonemap_cuda(algorithm, sig, param, peak);
    
    // Apply the computed scale factor to the color
    float scale = sig / sig_orig;
    r *= scale;
    g *= scale;
    b *= scale;
    
    int idx = y * (dst_pitch / sizeof(float)) + x;
    dst_r[idx] = r;
    dst_g[idx] = g;
    dst_b[idx] = b;
}

__global__ void tonemap_cuda_planar_float(
    float *src_r, float *src_g, float *src_b,
    float *dst_r, float *dst_g, float *dst_b,
    int width, int height, int src_pitch, int dst_pitch,
    int algorithm, float param, float desat, float peak,
    float coeff_r, float coeff_g, float coeff_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    int src_idx = y * (src_pitch / sizeof(float)) + x;
    int dst_idx = y * (dst_pitch / sizeof(float)) + x;
    
    float r = src_r[src_idx];
    float g = src_g[src_idx];
    float b = src_b[src_idx];
    
    // Desaturate to prevent unnatural colors
    if (desat > 0) {
        float luma = coeff_r * r + coeff_g * g + coeff_b * b;
        float overbright = fmaxf(luma - desat, 1e-6f) / fmaxf(luma, 1e-6f);
        r = r * (1.0f - overbright) + luma * overbright;
        g = g * (1.0f - overbright) + luma * overbright;
        b = b * (1.0f - overbright) + luma * overbright;
    }
    
    // Pick the brightest component
    float sig = fmaxf(fmaxf(fmaxf(r, g), b), 1e-6f);
    float sig_orig = sig;
    
    // Apply tonemap algorithm
    sig = apply_tonemap_cuda(algorithm, sig, param, peak);
    
    // Apply the computed scale factor to the color
    float scale = sig / sig_orig;
    r *= scale;
    g *= scale;
    b *= scale;
    
    dst_r[dst_idx] = r;
    dst_g[dst_idx] = g;
    dst_b[dst_idx] = b;
}

__global__ void tonemap_cuda_16bit(
    cudaTextureObject_t src_tex,
    ushort4 *dst, int dst_width, int dst_height, int dst_pitch,
    int algorithm, float param, float desat, float peak,
    float coeff_r, float coeff_g, float coeff_b, int bit_depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    ushort4 src = tex2D<ushort4>(src_tex, x, y);
    float max_val = (1 << bit_depth) - 1;
    
    float r = src.x / max_val;
    float g = src.y / max_val;
    float b = src.z / max_val;
    
    // Desaturate to prevent unnatural colors
    if (desat > 0) {
        float luma = coeff_r * r + coeff_g * g + coeff_b * b;
        float overbright = fmaxf(luma - desat, 1e-6f) / fmaxf(luma, 1e-6f);
        r = r * (1.0f - overbright) + luma * overbright;
        g = g * (1.0f - overbright) + luma * overbright;
        b = b * (1.0f - overbright) + luma * overbright;
    }
    
    // Pick the brightest component
    float sig = fmaxf(fmaxf(fmaxf(r, g), b), 1e-6f);
    float sig_orig = sig;
    
    // Apply tonemap algorithm
    sig = apply_tonemap_cuda(algorithm, sig, param, peak);
    
    // Apply the computed scale factor to the color
    float scale = sig / sig_orig;
    r *= scale;
    g *= scale;
    b *= scale;
    
    dst[y * (dst_pitch / sizeof(ushort4)) + x] = make_ushort4(
        (unsigned short)(r * max_val + 0.5f),
        (unsigned short)(g * max_val + 0.5f),
        (unsigned short)(b * max_val + 0.5f),
        src.w
    );
}

__global__ void tonemap_cuda_p016_to_nv12(
    cudaTextureObject_t src_y_tex, cudaTextureObject_t src_uv_tex,
    unsigned char *dst_y, unsigned char *dst_uv,
    int dst_width, int dst_height, int dst_pitch_y, int dst_pitch_uv,
    int algorithm, float param, float desat, float peak,
    float coeff_r, float coeff_g, float coeff_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    // Sample Y component (16-bit)
    unsigned short y_val = tex2D<unsigned short>(src_y_tex, x, y);
    
    // Sample UV components (subsampled, 16-bit) - ensure consistent sampling for 2x2 blocks
    int uv_x = (x / 2);
    int uv_y = (y / 2);
    ushort2 uv_val = tex2D<ushort2>(src_uv_tex, uv_x, uv_y);
    
    // Convert YUV to RGB (BT.2020 for HDR, P016LE format: 10-bit left-justified in 16-bit)
    // P016LE: 10-bit values are stored in upper 10 bits, so divide by 64 to get 10-bit range
    // Add safety clamping to prevent extreme values
    unsigned short y_10bit = y_val >> 6;
    unsigned short u_10bit = uv_val.x >> 6;
    unsigned short v_10bit = uv_val.y >> 6;
    
    // Clamp to valid 10-bit TV range before normalization
    y_10bit = max(64, min(940, (int)y_10bit));
    u_10bit = max(64, min(960, (int)u_10bit));
    v_10bit = max(64, min(960, (int)v_10bit));
    
    float Y = (y_10bit - 64.0f) / 876.0f; // 10-bit Y: range 64-940 -> 0.0-1.0
    float U = (u_10bit - 512.0f) / 448.0f; // 10-bit U: range 64-960, centered at 512, scale by half-range
    float V = (v_10bit - 512.0f) / 448.0f; // 10-bit V: range 64-960, centered at 512, scale by half-range
    
    // BT.2020 YUV to RGB conversion (corrected coefficients)
    float r = Y + 1.7166f * V;
    float g = Y - 0.1873f * U - 0.6526f * V;
    float b = Y + 2.0426f * U;
    
    r = fmaxf(r, 0.0f);
    g = fmaxf(g, 0.0f);
    b = fmaxf(b, 0.0f);
    
    // Desaturate to prevent unnatural colors
    if (desat > 0) {
        float luma = coeff_r * r + coeff_g * g + coeff_b * b;
        float overbright = fmaxf(luma - desat, 1e-6f) / fmaxf(luma, 1e-6f);
        r = r * (1.0f - overbright) + luma * overbright;
        g = g * (1.0f - overbright) + luma * overbright;
        b = b * (1.0f - overbright) + luma * overbright;
    }
    
    // Pick the brightest component
    float sig = fmaxf(fmaxf(fmaxf(r, g), b), 1e-6f);
    float sig_orig = sig;
    
    // Apply tonemap algorithm
    sig = apply_tonemap_cuda(algorithm, sig, param, peak);
    
    // Apply the computed scale factor to the color
    float scale = sig / sig_orig;
    
    // Debug: clamp extreme scale values for hable to prevent artifacts
    if (algorithm == TONEMAP_HABLE) {
        scale = fmaxf(0.01f, fminf(10.0f, scale));
    }
    
    r *= scale;
    g *= scale;
    b *= scale;
    
    // Clamp RGB to valid range [0.0, 1.0] before YUV conversion
    r = fmaxf(0.0f, fminf(1.0f, r));
    g = fmaxf(0.0f, fminf(1.0f, g));
    b = fmaxf(0.0f, fminf(1.0f, b));
    
    // Convert back to YUV (BT.2020 to preserve input colorspace)
    float out_Y = 0.2627f * r + 0.6780f * g + 0.0593f * b;
    float out_U = -0.1396f * r - 0.3604f * g + 0.5000f * b;
    float out_V = 0.5000f * r - 0.4598f * g - 0.0402f * b;
    
    // Scale to 8-bit TV range and clamp
    int y_out = (int)(out_Y * 219.0f + 16.0f + 0.5f);
    int u_out = (int)(out_U * 224.0f + 128.0f + 0.5f);
    int v_out = (int)(out_V * 224.0f + 128.0f + 0.5f);
    
    // Clamp to valid TV range
    y_out = max(16, min(235, y_out));
    u_out = max(16, min(240, u_out));  // Proper TV range for chroma
    v_out = max(16, min(240, v_out));
    
    // Write Y component
    dst_y[y * dst_pitch_y + x] = y_out;
    
    // Write UV components in NV12 format (only for one thread per 2x2 block to avoid race conditions)
    if (x % 2 == 0 && y % 2 == 0) {
        int uv_idx = (y / 2) * dst_pitch_uv + (x / 2) * 2;  // Correct NV12 UV indexing
        dst_uv[uv_idx] = u_out;
        dst_uv[uv_idx + 1] = v_out;
    }
}

}