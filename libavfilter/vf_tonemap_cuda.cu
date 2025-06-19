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
    
    // Sample UV components (subsampled, 16-bit)
    int uv_x = x / 2;
    int uv_y = y / 2;
    ushort2 uv_val = tex2D<ushort2>(src_uv_tex, uv_x, uv_y);
    
    // Convert YUV to RGB (BT.2020 for HDR, 10-bit in 16-bit container)
    float Y = (y_val - 64.0f * 64.0f) / (940.0f * 64.0f); // 10-bit scaling (shifted up by 6 bits in 16-bit)
    float U = (uv_val.x - 512.0f * 64.0f) / (896.0f * 64.0f);
    float V = (uv_val.y - 512.0f * 64.0f) / (896.0f * 64.0f);
    
    // BT.2020 YUV to RGB conversion
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
    r *= scale;
    g *= scale;
    b *= scale;
    
    // Convert back to YUV (BT.709 for SDR)
    float out_Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float out_U = -0.1146f * r - 0.3854f * g + 0.5000f * b;
    float out_V = 0.5000f * r - 0.4542f * g - 0.0458f * b;
    
    // Scale to 8-bit range
    unsigned char y_out = (unsigned char)(out_Y * 219.0f + 16.0f + 0.5f);
    unsigned char u_out = (unsigned char)(out_U * 224.0f + 128.0f + 0.5f);
    unsigned char v_out = (unsigned char)(out_V * 224.0f + 128.0f + 0.5f);
    
    // Clamp to valid range
    y_out = min(max(y_out, (unsigned char)16), (unsigned char)235);
    u_out = min(max(u_out, (unsigned char)16), (unsigned char)240);
    v_out = min(max(v_out, (unsigned char)16), (unsigned char)240);
    
    // Write Y component
    dst_y[y * dst_pitch_y + x] = y_out;
    
    // Write UV components in NV12 format (only for even pixels to maintain 4:2:0 subsampling)
    if (x % 2 == 0 && y % 2 == 0) {
        int uv_idx = (y / 2) * dst_pitch_uv + (x / 2) * 2;
        dst_uv[uv_idx] = u_out;
        dst_uv[uv_idx + 1] = v_out;
    }
}

}