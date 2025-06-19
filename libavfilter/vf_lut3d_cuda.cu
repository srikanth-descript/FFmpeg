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

__device__ static inline float3 lerpf3(float3 a, float3 b, float t)
{
    return make_float3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    );
}

__device__ static inline float3 interp_nearest_cuda(float3 *lut, int lutsize, float3 s)
{
    int r = (int)(s.x + 0.5f);
    int g = (int)(s.y + 0.5f);
    int b = (int)(s.z + 0.5f);
    
    r = min(max(r, 0), lutsize - 1);
    g = min(max(g, 0), lutsize - 1);
    b = min(max(b, 0), lutsize - 1);
    
    return lut[r * lutsize * lutsize + g * lutsize + b];
}

__device__ static inline float3 interp_trilinear_cuda(float3 *lut, int lutsize, float3 s)
{
    int lutsize2 = lutsize * lutsize;
    
    int prev_r = (int)s.x;
    int prev_g = (int)s.y;
    int prev_b = (int)s.z;
    
    int next_r = min(prev_r + 1, lutsize - 1);
    int next_g = min(prev_g + 1, lutsize - 1);
    int next_b = min(prev_b + 1, lutsize - 1);
    
    float3 d = make_float3(s.x - prev_r, s.y - prev_g, s.z - prev_b);
    
    float3 c000 = lut[prev_r * lutsize2 + prev_g * lutsize + prev_b];
    float3 c001 = lut[prev_r * lutsize2 + prev_g * lutsize + next_b];
    float3 c010 = lut[prev_r * lutsize2 + next_g * lutsize + prev_b];
    float3 c011 = lut[prev_r * lutsize2 + next_g * lutsize + next_b];
    float3 c100 = lut[next_r * lutsize2 + prev_g * lutsize + prev_b];
    float3 c101 = lut[next_r * lutsize2 + prev_g * lutsize + next_b];
    float3 c110 = lut[next_r * lutsize2 + next_g * lutsize + prev_b];
    float3 c111 = lut[next_r * lutsize2 + next_g * lutsize + next_b];
    
    float3 c00 = lerpf3(c000, c100, d.x);
    float3 c10 = lerpf3(c010, c110, d.x);
    float3 c01 = lerpf3(c001, c101, d.x);
    float3 c11 = lerpf3(c011, c111, d.x);
    float3 c0 = lerpf3(c00, c10, d.y);
    float3 c1 = lerpf3(c01, c11, d.y);
    
    return lerpf3(c0, c1, d.z);
}

__device__ static inline float3 interp_tetrahedral_cuda(float3 *lut, int lutsize, float3 s)
{
    int lutsize2 = lutsize * lutsize;
    
    int prev_r = (int)s.x;
    int prev_g = (int)s.y;
    int prev_b = (int)s.z;
    
    int next_r = min(prev_r + 1, lutsize - 1);
    int next_g = min(prev_g + 1, lutsize - 1);
    int next_b = min(prev_b + 1, lutsize - 1);
    
    float3 d = make_float3(s.x - prev_r, s.y - prev_g, s.z - prev_b);
    
    float3 c000 = lut[prev_r * lutsize2 + prev_g * lutsize + prev_b];
    float3 c111 = lut[next_r * lutsize2 + next_g * lutsize + next_b];
    
    float3 c;
    if (d.x > d.y && d.x > d.z) {
        if (d.y > d.z) {
            float3 c100 = lut[next_r * lutsize2 + prev_g * lutsize + prev_b];
            float3 c110 = lut[next_r * lutsize2 + next_g * lutsize + prev_b];
            c = make_float3(
                c000.x + d.x * (c100.x - c000.x) + d.y * (c110.x - c100.x) + d.z * (c111.x - c110.x),
                c000.y + d.x * (c100.y - c000.y) + d.y * (c110.y - c100.y) + d.z * (c111.y - c110.y),
                c000.z + d.x * (c100.z - c000.z) + d.y * (c110.z - c100.z) + d.z * (c111.z - c110.z)
            );
        } else {
            float3 c100 = lut[next_r * lutsize2 + prev_g * lutsize + prev_b];
            float3 c101 = lut[next_r * lutsize2 + prev_g * lutsize + next_b];
            c = make_float3(
                c000.x + d.x * (c100.x - c000.x) + d.z * (c101.x - c100.x) + d.y * (c111.x - c101.x),
                c000.y + d.x * (c100.y - c000.y) + d.z * (c101.y - c100.y) + d.y * (c111.y - c101.y),
                c000.z + d.x * (c100.z - c000.z) + d.z * (c101.z - c100.z) + d.y * (c111.z - c101.z)
            );
        }
    } else if (d.y > d.z) {
        if (d.x > d.z) {
            float3 c010 = lut[prev_r * lutsize2 + next_g * lutsize + prev_b];
            float3 c110 = lut[next_r * lutsize2 + next_g * lutsize + prev_b];
            c = make_float3(
                c000.x + d.y * (c010.x - c000.x) + d.x * (c110.x - c010.x) + d.z * (c111.x - c110.x),
                c000.y + d.y * (c010.y - c000.y) + d.x * (c110.y - c010.y) + d.z * (c111.y - c110.y),
                c000.z + d.y * (c010.z - c000.z) + d.x * (c110.z - c010.z) + d.z * (c111.z - c110.z)
            );
        } else {
            float3 c010 = lut[prev_r * lutsize2 + next_g * lutsize + prev_b];
            float3 c011 = lut[prev_r * lutsize2 + next_g * lutsize + next_b];
            c = make_float3(
                c000.x + d.y * (c010.x - c000.x) + d.z * (c011.x - c010.x) + d.x * (c111.x - c011.x),
                c000.y + d.y * (c010.y - c000.y) + d.z * (c011.y - c010.y) + d.x * (c111.y - c011.y),
                c000.z + d.y * (c010.z - c000.z) + d.z * (c011.z - c010.z) + d.x * (c111.z - c011.z)
            );
        }
    } else {
        if (d.x > d.y) {
            float3 c001 = lut[prev_r * lutsize2 + prev_g * lutsize + next_b];
            float3 c101 = lut[next_r * lutsize2 + prev_g * lutsize + next_b];
            c = make_float3(
                c000.x + d.z * (c001.x - c000.x) + d.x * (c101.x - c001.x) + d.y * (c111.x - c101.x),
                c000.y + d.z * (c001.y - c000.y) + d.x * (c101.y - c001.y) + d.y * (c111.y - c101.y),
                c000.z + d.z * (c001.z - c000.z) + d.x * (c101.z - c001.z) + d.y * (c111.z - c101.z)
            );
        } else {
            float3 c001 = lut[prev_r * lutsize2 + prev_g * lutsize + next_b];
            float3 c011 = lut[prev_r * lutsize2 + next_g * lutsize + next_b];
            c = make_float3(
                c000.x + d.z * (c001.x - c000.x) + d.y * (c011.x - c001.x) + d.x * (c111.x - c011.x),
                c000.y + d.z * (c001.y - c000.y) + d.y * (c011.y - c001.y) + d.x * (c111.y - c011.y),
                c000.z + d.z * (c001.z - c000.z) + d.y * (c011.z - c001.z) + d.x * (c111.z - c011.z)
            );
        }
    }
    
    return c;
}

__global__ void lut3d_interp_8_nearest(
    cudaTextureObject_t src_tex,
    uchar4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    uchar4 src = tex2D<uchar4>(src_tex, x, y);
    
    float3 s = make_float3(
        src.x / 255.0f * scale_r,
        src.y / 255.0f * scale_g,
        src.z / 255.0f * scale_b
    );
    
    float3 result = interp_nearest_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(uchar4)) + x] = make_uchar4(
        (unsigned char)(result.x * 255.0f + 0.5f),
        (unsigned char)(result.y * 255.0f + 0.5f),
        (unsigned char)(result.z * 255.0f + 0.5f),
        src.w
    );
}

__global__ void lut3d_interp_10_nearest(
    cudaTextureObject_t src_tex,
    ushort4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    ushort4 src = tex2D<ushort4>(src_tex, x, y);
    
    float3 s = make_float3(
        (src.x >> 6) * scale_r,  // Convert 10-bit to normalized range
        (src.y >> 6) * scale_g,
        (src.z >> 6) * scale_b
    );
    
    float3 result = interp_nearest_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(ushort4)) + x] = make_ushort4(
        (unsigned short)(result.x * 1023.0f + 0.5f) << 6,  // Convert back to 10-bit in 16-bit container
        (unsigned short)(result.y * 1023.0f + 0.5f) << 6,
        (unsigned short)(result.z * 1023.0f + 0.5f) << 6,
        src.w
    );
}

__global__ void lut3d_interp_12_nearest(
    cudaTextureObject_t src_tex,
    ushort4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    ushort4 src = tex2D<ushort4>(src_tex, x, y);
    
    float3 s = make_float3(
        (src.x >> 4) * scale_r,  // Convert 12-bit to normalized range
        (src.y >> 4) * scale_g,
        (src.z >> 4) * scale_b
    );
    
    float3 result = interp_nearest_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(ushort4)) + x] = make_ushort4(
        (unsigned short)(result.x * 4095.0f + 0.5f) << 4,  // Convert back to 12-bit in 16-bit container
        (unsigned short)(result.y * 4095.0f + 0.5f) << 4,
        (unsigned short)(result.z * 4095.0f + 0.5f) << 4,
        src.w
    );
}

__global__ void lut3d_interp_8_trilinear(
    cudaTextureObject_t src_tex,
    uchar4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    uchar4 src = tex2D<uchar4>(src_tex, x, y);
    
    float3 s = make_float3(
        src.x / 255.0f * scale_r,
        src.y / 255.0f * scale_g,
        src.z / 255.0f * scale_b
    );
    
    float3 result = interp_trilinear_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(uchar4)) + x] = make_uchar4(
        (unsigned char)(result.x * 255.0f + 0.5f),
        (unsigned char)(result.y * 255.0f + 0.5f),
        (unsigned char)(result.z * 255.0f + 0.5f),
        src.w
    );
}

__global__ void lut3d_interp_10_trilinear(
    cudaTextureObject_t src_tex,
    ushort4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    ushort4 src = tex2D<ushort4>(src_tex, x, y);
    
    float3 s = make_float3(
        (src.x >> 6) / 1023.0f * scale_r,
        (src.y >> 6) / 1023.0f * scale_g,
        (src.z >> 6) / 1023.0f * scale_b
    );
    
    float3 result = interp_trilinear_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(ushort4)) + x] = make_ushort4(
        (unsigned short)(result.x * 1023.0f + 0.5f) << 6,
        (unsigned short)(result.y * 1023.0f + 0.5f) << 6,
        (unsigned short)(result.z * 1023.0f + 0.5f) << 6,
        src.w
    );
}

__global__ void lut3d_interp_12_trilinear(
    cudaTextureObject_t src_tex,
    ushort4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    ushort4 src = tex2D<ushort4>(src_tex, x, y);
    
    float3 s = make_float3(
        (src.x >> 4) / 4095.0f * scale_r,
        (src.y >> 4) / 4095.0f * scale_g,
        (src.z >> 4) / 4095.0f * scale_b
    );
    
    float3 result = interp_trilinear_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(ushort4)) + x] = make_ushort4(
        (unsigned short)(result.x * 4095.0f + 0.5f) << 4,
        (unsigned short)(result.y * 4095.0f + 0.5f) << 4,
        (unsigned short)(result.z * 4095.0f + 0.5f) << 4,
        src.w
    );
}

__global__ void lut3d_interp_8_tetrahedral(
    cudaTextureObject_t src_tex,
    uchar4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    uchar4 src = tex2D<uchar4>(src_tex, x, y);
    
    float3 s = make_float3(
        src.x / 255.0f * scale_r,
        src.y / 255.0f * scale_g,
        src.z / 255.0f * scale_b
    );
    
    float3 result = interp_tetrahedral_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(uchar4)) + x] = make_uchar4(
        (unsigned char)(result.x * 255.0f + 0.5f),
        (unsigned char)(result.y * 255.0f + 0.5f),
        (unsigned char)(result.z * 255.0f + 0.5f),
        src.w
    );
}

__global__ void lut3d_interp_10_tetrahedral(
    cudaTextureObject_t src_tex,
    ushort4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    ushort4 src = tex2D<ushort4>(src_tex, x, y);
    
    float3 s = make_float3(
        (src.x >> 6) / 1023.0f * scale_r,
        (src.y >> 6) / 1023.0f * scale_g,
        (src.z >> 6) / 1023.0f * scale_b
    );
    
    float3 result = interp_tetrahedral_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(ushort4)) + x] = make_ushort4(
        (unsigned short)(result.x * 1023.0f + 0.5f) << 6,
        (unsigned short)(result.y * 1023.0f + 0.5f) << 6,
        (unsigned short)(result.z * 1023.0f + 0.5f) << 6,
        src.w
    );
}

__global__ void lut3d_interp_12_tetrahedral(
    cudaTextureObject_t src_tex,
    ushort4 *dst, int dst_width, int dst_height, int dst_pitch,
    float3 *lut, int lutsize, float scale_r, float scale_g, float scale_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    ushort4 src = tex2D<ushort4>(src_tex, x, y);
    
    float3 s = make_float3(
        (src.x >> 4) / 4095.0f * scale_r,
        (src.y >> 4) / 4095.0f * scale_g,
        (src.z >> 4) / 4095.0f * scale_b
    );
    
    float3 result = interp_tetrahedral_cuda(lut, lutsize, s);
    
    dst[y * (dst_pitch / sizeof(ushort4)) + x] = make_ushort4(
        (unsigned short)(result.x * 4095.0f + 0.5f) << 4,
        (unsigned short)(result.y * 4095.0f + 0.5f) << 4,
        (unsigned short)(result.z * 4095.0f + 0.5f) << 4,
        src.w
    );
}

}