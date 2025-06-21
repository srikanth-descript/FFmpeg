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

extern "C" {
#define MPEG_LUMA_MIN   (16)
#define MPEG_CHROMA_MIN (16)
#define MPEG_LUMA_MAX   (235)
#define MPEG_CHROMA_MAX (240)

#define JPEG_LUMA_MIN   (0)
#define JPEG_CHROMA_MIN (1)
#define JPEG_LUMA_MAX   (255)
#define JPEG_CHROMA_MAX (255)

__device__ int mpeg_min[] = {MPEG_LUMA_MIN, MPEG_CHROMA_MIN};
__device__ int mpeg_max[] = {MPEG_LUMA_MAX, MPEG_CHROMA_MAX};

__device__ int jpeg_min[] = {JPEG_LUMA_MIN, JPEG_CHROMA_MIN};
__device__ int jpeg_max[] = {JPEG_LUMA_MAX, JPEG_CHROMA_MAX};

__device__ int clamp(int val, int min, int max)
{
    if (val < min)
        return min;
    else if (val > max)
        return max;
    else
        return val;
}

__global__ void to_jpeg_cuda(const unsigned char* src, unsigned char* dst,
                             int pitch, int comp_id)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int src_, dst_;

    // 8 bit -> 15 bit for better precision
    src_ = static_cast<int>(src[x + y * pitch]) << 7;

    // Conversion
    dst_ = comp_id ? (min(src_, 30775) * 4663 - 9289992) >> 12    // chroma
                   : (min(src_, 30189) * 19077 - 39057361) >> 14; // luma

    // Dither replacement
    dst_ = dst_ + 64;

    // Back to 8 bit
    dst_ = clamp(dst_ >> 7, jpeg_min[comp_id], jpeg_max[comp_id]);
    dst[x + y * pitch] = static_cast<unsigned char>(dst_);
}

__global__ void to_mpeg_cuda(const unsigned char* src, unsigned char* dst,
                             int pitch, int comp_id)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int src_, dst_;

    // 8 bit -> 15 bit for better precision
    src_ = static_cast<int>(src[x + y * pitch]) << 7;

    // Conversion
    dst_ = comp_id ? (src_ * 1799 + 4081085) >> 11    // chroma
                   : (src_ * 14071 + 33561947) >> 14; // luma

    // Dither replacement
    dst_ = dst_ + 64;

    // Back to 8 bit
    dst_ = clamp(dst_ >> 7, mpeg_min[comp_id], mpeg_max[comp_id]);
    dst[x + y * pitch] = static_cast<unsigned char>(dst_);
}

__global__ void colorspace_convert_cuda(const unsigned char* src, unsigned char* dst,
                                       int src_pitch, int dst_pitch, int width, int height, int plane_id,
                                       float m00, float m01, float m02,
                                       float m10, float m11, float m12,
                                       float m20, float m21, float m22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (plane_id == 0) {
        // Luma plane (Y) - apply only first row of matrix
        float src_y = (float)src[x + y * src_pitch];
        float dst_y = m00 * src_y + m01 * 128.0f + m02 * 128.0f;
        dst[x + y * dst_pitch] = (unsigned char)clamp((int)(dst_y + 0.5f), 0, 255);
    } else if (plane_id == 1) {
        // U plane - apply second row of matrix, but we only have U component
        float src_u = (float)src[x + y * src_pitch] - 128.0f;
        // For U plane conversion, we apply the matrix considering U input and zero V
        float dst_u = m10 * 128.0f + m11 * (src_u + 128.0f) + m12 * 128.0f;
        dst[x + y * dst_pitch] = (unsigned char)clamp((int)(dst_u + 0.5f), 0, 255);
    } else {
        // V plane - apply third row of matrix, but we only have V component  
        float src_v = (float)src[x + y * src_pitch] - 128.0f;
        // For V plane conversion, we apply the matrix considering zero U input and V
        float dst_v = m20 * 128.0f + m21 * 128.0f + m22 * (src_v + 128.0f);
        dst[x + y * dst_pitch] = (unsigned char)clamp((int)(dst_v + 0.5f), 0, 255);
    }
}

}
