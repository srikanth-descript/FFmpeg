/*
 * CUDA Async Frame Queue for FFmpeg filters
 * Copyright (c) 2024
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

#ifndef AVFILTER_CUDA_ASYNC_QUEUE_H
#define AVFILTER_CUDA_ASYNC_QUEUE_H

#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/frame.h"

#define MAX_CUDA_STREAMS 4
#define MAX_FRAME_QUEUE_SIZE 8
#define DEFAULT_FRAME_QUEUE_SIZE 4
#define DEFAULT_CUDA_STREAMS 2

typedef struct CudaAsyncFrame {
    AVFrame *frame;
    AVFrame *output_frame;
    CUevent event_start;
    CUevent event_done;
    int in_use;
    int stream_idx;
} CudaAsyncFrame;

typedef struct CudaAsyncQueue {
    CudaAsyncFrame frames[MAX_FRAME_QUEUE_SIZE];
    CUstream streams[MAX_CUDA_STREAMS];
    
    int queue_size;
    int num_streams;
    int head;
    int tail;
    int count;
    
    AVBufferRef *frames_ctx;
    AVCUDADeviceContext *hwctx;
    CudaFunctions *cu;
    
    int initialized;
} CudaAsyncQueue;

int cuda_async_queue_init(CudaAsyncQueue *queue, AVCUDADeviceContext *hwctx, 
                         AVBufferRef *frames_ctx, int queue_size, int num_streams);

void cuda_async_queue_uninit(CudaAsyncQueue *queue);

CudaAsyncFrame* cuda_async_queue_get_free_frame(CudaAsyncQueue *queue);

int cuda_async_queue_submit_frame(CudaAsyncQueue *queue, CudaAsyncFrame *async_frame);

CudaAsyncFrame* cuda_async_queue_get_completed_frame(CudaAsyncQueue *queue);

int cuda_async_queue_wait_for_completion(CudaAsyncQueue *queue, CudaAsyncFrame *async_frame);

void cuda_async_queue_sync_all(CudaAsyncQueue *queue);

#endif