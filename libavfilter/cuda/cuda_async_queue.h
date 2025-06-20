/*
 * CUDA Async Queue - Header file for asynchronous CUDA frame processing
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

#ifndef AVFILTER_CUDA_ASYNC_QUEUE_H
#define AVFILTER_CUDA_ASYNC_QUEUE_H

#include "libavutil/frame.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "../avfilter.h"

#define MAX_FRAME_QUEUE_SIZE 8
#define MAX_CUDA_STREAMS 4

typedef struct CudaAsyncFrame {
    AVFrame *input_frame;
    AVFrame *output_frame;
    CUevent event_start;
    CUevent event_done;
    int in_use;
    int stream_idx;
    int64_t submission_order;
    int64_t submit_time_us;
    int64_t complete_time_us;
} CudaAsyncFrame;

typedef struct CudaAsyncQueue {
    CudaAsyncFrame frames[MAX_FRAME_QUEUE_SIZE];
    CUstream streams[MAX_CUDA_STREAMS];

    int queue_size;
    int num_streams;

    int write_idx;
    int read_idx;
    int frames_in_queue;
    int64_t submission_counter;

    CUcontext cuda_ctx;
    AVCUDADeviceContext *hwctx;

    void *filter_ctx;
    int (*process_frame)(void *filter_ctx, AVFrame * out, AVFrame * in,
                         CUstream stream);
} CudaAsyncQueue;

int ff_cuda_async_queue_init(CudaAsyncQueue * queue,
                             AVCUDADeviceContext * hwctx, int queue_size,
                             int num_streams, void *filter_ctx,
                             int (*process_frame)(void *, AVFrame *,
                                                  AVFrame *, CUstream));

void ff_cuda_async_queue_uninit(CudaAsyncQueue * queue);

int ff_cuda_async_queue_submit(CudaAsyncQueue * queue, AVFrame * in_frame);

int ff_cuda_async_queue_receive(CudaAsyncQueue * queue,
                                AVFrame ** out_frame);

int ff_cuda_async_queue_flush(CudaAsyncQueue * queue);

int ff_cuda_async_queue_is_full(CudaAsyncQueue * queue);

int ff_cuda_async_queue_is_empty(CudaAsyncQueue * queue);

#endif                          /* AVFILTER_CUDA_ASYNC_QUEUE_H */
