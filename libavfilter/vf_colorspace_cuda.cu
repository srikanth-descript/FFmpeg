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

// 16-bit constants (10-bit values shifted to 16-bit)
#define MPEG_LUMA_MIN_16   (16 << 6)
#define MPEG_CHROMA_MIN_16 (16 << 6)
#define MPEG_LUMA_MAX_16   (235 << 6)
#define MPEG_CHROMA_MAX_16 (240 << 6)

#define JPEG_LUMA_MIN_16   (0)
#define JPEG_CHROMA_MIN_16 (1 << 6)
#define JPEG_LUMA_MAX_16   (65535)
#define JPEG_CHROMA_MAX_16 (65535)

__device__ int mpeg_min[] = {MPEG_LUMA_MIN, MPEG_CHROMA_MIN};
__device__ int mpeg_max[] = {MPEG_LUMA_MAX, MPEG_CHROMA_MAX};

__device__ int jpeg_min[] = {JPEG_LUMA_MIN, JPEG_CHROMA_MIN};
__device__ int jpeg_max[] = {JPEG_LUMA_MAX, JPEG_CHROMA_MAX};

__device__ int mpeg_min_16[] = {MPEG_LUMA_MIN_16, MPEG_CHROMA_MIN_16};
__device__ int mpeg_max_16[] = {MPEG_LUMA_MAX_16, MPEG_CHROMA_MAX_16};

__device__ int jpeg_min_16[] = {JPEG_LUMA_MIN_16, JPEG_CHROMA_MIN_16};
__device__ int jpeg_max_16[] = {JPEG_LUMA_MAX_16, JPEG_CHROMA_MAX_16};

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

__global__ void to_jpeg_cuda_16(const unsigned short* src, unsigned short* dst,
                                int pitch, int comp_id)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pitch_pixels = pitch / 2;  // Convert byte pitch to pixel pitch
    int src_, dst_;

    // 10-bit input in 16-bit container -> scale for processing
    src_ = static_cast<int>(src[x + y * pitch_pixels]);

    // Scale to full 16-bit range for conversion
    dst_ = comp_id ? (src_ * 65535 - 64 * 65535) / (1023 - 64)     // chroma
                   : (src_ * 65535 - 64 * 65535) / (940 - 64);    // luma

    dst[x + y * pitch_pixels] = static_cast<unsigned short>(clamp(dst_, jpeg_min_16[comp_id], jpeg_max_16[comp_id]));
}

__global__ void to_mpeg_cuda_16(const unsigned short* src, unsigned short* dst,
                                int pitch, int comp_id)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pitch_pixels = pitch / 2;  // Convert byte pitch to pixel pitch
    int src_, dst_;

    // 10-bit input in 16-bit container -> scale to limited range
    src_ = static_cast<int>(src[x + y * pitch_pixels]);

    // Scale from full range to limited range
    dst_ = comp_id ? (src_ * (240 - 16) / 65535) + 16     // chroma
                   : (src_ * (235 - 16) / 65535) + 16;    // luma

    // Scale to 16-bit container (10-bit values)
    dst_ = dst_ << 6;

    dst[x + y * pitch_pixels] = static_cast<unsigned short>(clamp(dst_, mpeg_min_16[comp_id], mpeg_max_16[comp_id]));
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

__global__ void colorspace_convert_cuda_16(const unsigned short* src, unsigned short* dst,
                                          int src_pitch, int dst_pitch, int width, int height, int plane_id,
                                          float m00, float m01, float m02,
                                          float m10, float m11, float m12,
                                          float m20, float m21, float m22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int src_pitch_pixels = src_pitch / 2;  // Convert byte pitch to pixel pitch
    int dst_pitch_pixels = dst_pitch / 2;  // Convert byte pitch to pixel pitch

    if (x >= width || y >= height)
        return;

    // For 10-bit in 16-bit container, chroma center is at 512 << 6 = 32768
    const float chroma_center = 32768.0f;

    if (plane_id == 0) {
        // Luma plane (Y) - apply only first row of matrix
        float src_y = (float)src[x + y * src_pitch_pixels];
        float dst_y = m00 * src_y + m01 * chroma_center + m02 * chroma_center;
        dst[x + y * dst_pitch_pixels] = (unsigned short)clamp((int)(dst_y + 0.5f), 0, 65535);
    } else if (plane_id == 1) {
        // U plane - apply second row of matrix, but we only have U component
        float src_u = (float)src[x + y * src_pitch_pixels] - chroma_center;
        // For U plane conversion, we apply the matrix considering U input and zero V
        float dst_u = m10 * chroma_center + m11 * (src_u + chroma_center) + m12 * chroma_center;
        dst[x + y * dst_pitch_pixels] = (unsigned short)clamp((int)(dst_u + 0.5f), 0, 65535);
    } else {
        // V plane - apply third row of matrix, but we only have V component  
        float src_v = (float)src[x + y * src_pitch_pixels] - chroma_center;
        // For V plane conversion, we apply the matrix considering zero U input and V
        float dst_v = m20 * chroma_center + m21 * chroma_center + m22 * (src_v + chroma_center);
        dst[x + y * dst_pitch_pixels] = (unsigned short)clamp((int)(dst_v + 0.5f), 0, 65535);
    }
}

// New kernel for proper YUV colorspace conversion that processes all components together
__global__ void colorspace_convert_yuv_cuda(const unsigned char* src_y, const unsigned char* src_u, const unsigned char* src_v,
                                            unsigned char* dst_y, unsigned char* dst_u, unsigned char* dst_v,
                                            int src_pitch_y, int src_pitch_uv, int dst_pitch_y, int dst_pitch_uv,
                                            int width, int height,
                                            float m00, float m01, float m02,
                                            float m10, float m11, float m12,
                                            float m20, float m21, float m22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Read YUV components
    float src_Y = (float)src_y[x + y * src_pitch_y];
    
    // For chroma subsampling, calculate chroma coordinates
    int chroma_x = x / 2;
    int chroma_y = y / 2;
    
    float src_U = (float)src_u[chroma_x + chroma_y * src_pitch_uv] - 128.0f;
    float src_V = (float)src_v[chroma_x + chroma_y * src_pitch_uv] - 128.0f;

    // Apply 3x3 matrix transformation
    float dst_Y = m00 * src_Y + m01 * (src_U + 128.0f) + m02 * (src_V + 128.0f);
    float dst_U = m10 * src_Y + m11 * (src_U + 128.0f) + m12 * (src_V + 128.0f);
    float dst_V = m20 * src_Y + m21 * (src_U + 128.0f) + m22 * (src_V + 128.0f);

    // Clamp and write Y component
    dst_y[x + y * dst_pitch_y] = (unsigned char)clamp((int)(dst_Y + 0.5f), 0, 255);

    // Write UV components (only for appropriate positions to handle subsampling)
    if (x % 2 == 0 && y % 2 == 0) {
        dst_u[chroma_x + chroma_y * dst_pitch_uv] = (unsigned char)clamp((int)(dst_U + 0.5f), 0, 255);
        dst_v[chroma_x + chroma_y * dst_pitch_uv] = (unsigned char)clamp((int)(dst_V + 0.5f), 0, 255);
    }
}

// New kernel for 16-bit YUV colorspace conversion 
__global__ void colorspace_convert_yuv_cuda_16(const unsigned short* src_y, const unsigned short* src_u, const unsigned short* src_v,
                                               unsigned short* dst_y, unsigned short* dst_u, unsigned short* dst_v,
                                               int src_pitch_y, int src_pitch_uv, int dst_pitch_y, int dst_pitch_uv,
                                               int width, int height,
                                               float m00, float m01, float m02,
                                               float m10, float m11, float m12,
                                               float m20, float m21, float m22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Convert byte pitch to pixel pitch for 16-bit
    int src_pitch_y_pixels = src_pitch_y / 2;
    int src_pitch_uv_pixels = src_pitch_uv / 2;
    int dst_pitch_y_pixels = dst_pitch_y / 2;
    int dst_pitch_uv_pixels = dst_pitch_uv / 2;

    // For 10-bit in 16-bit container, chroma center is at 512 << 6 = 32768
    const float chroma_center = 32768.0f;

    // Read YUV components
    float src_Y = (float)src_y[x + y * src_pitch_y_pixels];
    
    // For chroma subsampling, calculate chroma coordinates
    int chroma_x = x / 2;
    int chroma_y = y / 2;
    
    float src_U = (float)src_u[chroma_x + chroma_y * src_pitch_uv_pixels] - chroma_center;
    float src_V = (float)src_v[chroma_x + chroma_y * src_pitch_uv_pixels] - chroma_center;

    // Apply 3x3 matrix transformation
    float dst_Y = m00 * src_Y + m01 * (src_U + chroma_center) + m02 * (src_V + chroma_center);
    float dst_U = m10 * src_Y + m11 * (src_U + chroma_center) + m12 * (src_V + chroma_center);
    float dst_V = m20 * src_Y + m21 * (src_U + chroma_center) + m22 * (src_V + chroma_center);

    // Clamp and write Y component
    dst_y[x + y * dst_pitch_y_pixels] = (unsigned short)clamp((int)(dst_Y + 0.5f), 0, 65535);

    // Write UV components (only for appropriate positions to handle subsampling)
    if (x % 2 == 0 && y % 2 == 0) {
        dst_u[chroma_x + chroma_y * dst_pitch_uv_pixels] = (unsigned short)clamp((int)(dst_U + 0.5f), 0, 65535);
        dst_v[chroma_x + chroma_y * dst_pitch_uv_pixels] = (unsigned short)clamp((int)(dst_V + 0.5f), 0, 65535);
    }
}

// Specialized kernel for NV12 format (Y plane + interleaved UV plane)
__global__ void colorspace_convert_nv12_cuda(const unsigned char* src_y, const unsigned char* src_uv,
                                             unsigned char* dst_y, unsigned char* dst_uv,
                                             int src_pitch_y, int src_pitch_uv, int dst_pitch_y, int dst_pitch_uv,
                                             int width, int height,
                                             float m00, float m01, float m02,
                                             float m10, float m11, float m12,
                                             float m20, float m21, float m22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Read Y component
    float src_Y = (float)src_y[x + y * src_pitch_y];
    
    // For chroma subsampling, calculate chroma coordinates
    int chroma_x = x / 2;
    int chroma_y = y / 2;
    
    // Check bounds for UV plane access
    if (chroma_x >= (width / 2) || chroma_y >= (height / 2))
        return;
    
    int uv_idx = chroma_x * 2 + chroma_y * src_pitch_uv; // NV12: UVUVUV...
    
    // Read interleaved UV components safely
    float src_U = (float)src_uv[uv_idx] - 128.0f;
    float src_V = (float)src_uv[uv_idx + 1] - 128.0f;

    // Apply 3x3 matrix transformation
    float dst_Y = m00 * src_Y + m01 * (src_U + 128.0f) + m02 * (src_V + 128.0f);
    float dst_U = m10 * src_Y + m11 * (src_U + 128.0f) + m12 * (src_V + 128.0f);
    float dst_V = m20 * src_Y + m21 * (src_U + 128.0f) + m22 * (src_V + 128.0f);

    // Clamp and write Y component
    dst_y[x + y * dst_pitch_y] = (unsigned char)clamp((int)(dst_Y + 0.5f), 0, 255);

    // Write UV components (only for appropriate positions to handle subsampling)
    if (x % 2 == 0 && y % 2 == 0) {
        int dst_uv_idx = chroma_x * 2 + chroma_y * dst_pitch_uv;
        dst_uv[dst_uv_idx] = (unsigned char)clamp((int)(dst_U + 0.5f), 0, 255);
        dst_uv[dst_uv_idx + 1] = (unsigned char)clamp((int)(dst_V + 0.5f), 0, 255);
    }
}

}
