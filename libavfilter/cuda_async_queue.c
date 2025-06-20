/*
 * CUDA Async Frame Queue implementation for FFmpeg filters
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

#include "cuda_async_queue.h"
#include "libavutil/log.h"
#include "libavutil/hwcontext.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(NULL, queue->cu, x)

int cuda_async_queue_init(CudaAsyncQueue *queue, AVCUDADeviceContext *hwctx, 
                         AVBufferRef *frames_ctx, int queue_size, int num_streams)
{
    CUcontext dummy;
    int ret, i;
    
    if (!queue || !hwctx || !frames_ctx)
        return AVERROR(EINVAL);
    
    if (queue_size <= 0 || queue_size > MAX_FRAME_QUEUE_SIZE)
        queue_size = DEFAULT_FRAME_QUEUE_SIZE;
    
    if (num_streams <= 0 || num_streams > MAX_CUDA_STREAMS)
        num_streams = DEFAULT_CUDA_STREAMS;
    
    memset(queue, 0, sizeof(*queue));
    
    queue->hwctx = hwctx;
    queue->cu = hwctx->internal->cuda_dl;
    queue->frames_ctx = av_buffer_ref(frames_ctx);
    queue->queue_size = queue_size;
    queue->num_streams = num_streams;
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    
    if (!queue->frames_ctx)
        return AVERROR(ENOMEM);
    
    ret = CHECK_CU(queue->cu->cuCtxPushCurrent(hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;
    
    // Create CUDA streams
    for (i = 0; i < num_streams; i++) {
        ret = CHECK_CU(queue->cu->cuStreamCreate(&queue->streams[i], CU_STREAM_NON_BLOCKING));
        if (ret < 0)
            goto fail_streams;
    }
    
    // Initialize frame queue
    for (i = 0; i < queue_size; i++) {
        CudaAsyncFrame *async_frame = &queue->frames[i];
        
        // Allocate input frame
        async_frame->frame = av_frame_alloc();
        if (!async_frame->frame) {
            ret = AVERROR(ENOMEM);
            goto fail_frames;
        }
        
        // Allocate output frame
        async_frame->output_frame = av_frame_alloc();
        if (!async_frame->output_frame) {
            ret = AVERROR(ENOMEM);
            goto fail_frames;
        }
        
        // Create CUDA events for synchronization
        ret = CHECK_CU(queue->cu->cuEventCreate(&async_frame->event_start, CU_EVENT_DEFAULT));
        if (ret < 0)
            goto fail_frames;
        
        ret = CHECK_CU(queue->cu->cuEventCreate(&async_frame->event_done, CU_EVENT_DEFAULT));
        if (ret < 0)
            goto fail_frames;
        
        async_frame->in_use = 0;
        async_frame->stream_idx = i % num_streams;
    }
    
    CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
    queue->initialized = 1;
    return 0;
    
fail_frames:
    for (i = 0; i < queue_size; i++) {
        CudaAsyncFrame *async_frame = &queue->frames[i];
        if (async_frame->event_start)
            CHECK_CU(queue->cu->cuEventDestroy(async_frame->event_start));
        if (async_frame->event_done)
            CHECK_CU(queue->cu->cuEventDestroy(async_frame->event_done));
        av_frame_free(&async_frame->frame);
        av_frame_free(&async_frame->output_frame);
    }
    
fail_streams:
    for (i = 0; i < num_streams; i++) {
        if (queue->streams[i])
            CHECK_CU(queue->cu->cuStreamDestroy(queue->streams[i]));
    }
    CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
    
fail:
    av_buffer_unref(&queue->frames_ctx);
    return ret;
}

void cuda_async_queue_uninit(CudaAsyncQueue *queue)
{
    CUcontext dummy;
    int i;
    
    if (!queue || !queue->initialized)
        return;
    
    if (queue->hwctx && queue->cu) {
        CHECK_CU(queue->cu->cuCtxPushCurrent(queue->hwctx->cuda_ctx));
        
        // Wait for all operations to complete
        cuda_async_queue_sync_all(queue);
        
        // Destroy events and free frames
        for (i = 0; i < queue->queue_size; i++) {
            CudaAsyncFrame *async_frame = &queue->frames[i];
            if (async_frame->event_start)
                CHECK_CU(queue->cu->cuEventDestroy(async_frame->event_start));
            if (async_frame->event_done)
                CHECK_CU(queue->cu->cuEventDestroy(async_frame->event_done));
            av_frame_free(&async_frame->frame);
            av_frame_free(&async_frame->output_frame);
        }
        
        // Destroy streams
        for (i = 0; i < queue->num_streams; i++) {
            if (queue->streams[i])
                CHECK_CU(queue->cu->cuStreamDestroy(queue->streams[i]));
        }
        
        CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
    }
    
    av_buffer_unref(&queue->frames_ctx);
    queue->initialized = 0;
}

CudaAsyncFrame* cuda_async_queue_get_free_frame(CudaAsyncQueue *queue)
{
    CudaAsyncFrame *async_frame;
    int i;
    
    if (!queue || !queue->initialized)
        return NULL;
    
    // Check if queue is full
    if (queue->count >= queue->queue_size)
        return NULL;
    
    // Find a free frame
    for (i = 0; i < queue->queue_size; i++) {
        async_frame = &queue->frames[(queue->head + i) % queue->queue_size];
        if (!async_frame->in_use)
            return async_frame;
    }
    
    return NULL;
}

int cuda_async_queue_submit_frame(CudaAsyncQueue *queue, CudaAsyncFrame *async_frame)
{
    CUcontext dummy;
    int ret;
    
    if (!queue || !async_frame || !queue->initialized)
        return AVERROR(EINVAL);
    
    ret = CHECK_CU(queue->cu->cuCtxPushCurrent(queue->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    
    // Record start event
    ret = CHECK_CU(queue->cu->cuEventRecord(async_frame->event_start, 
                                           queue->streams[async_frame->stream_idx]));
    if (ret < 0)
        goto fail;
    
    // Mark frame as in use
    async_frame->in_use = 1;
    queue->count++;
    queue->tail = (queue->tail + 1) % queue->queue_size;
    
fail:
    CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
    return ret;
}

CudaAsyncFrame* cuda_async_queue_get_completed_frame(CudaAsyncQueue *queue)
{
    CudaAsyncFrame *async_frame;
    CUcontext dummy;
    CUresult result;
    int ret;
    
    if (!queue || !queue->initialized || queue->count == 0)
        return NULL;
    
    ret = CHECK_CU(queue->cu->cuCtxPushCurrent(queue->hwctx->cuda_ctx));
    if (ret < 0)
        return NULL;
    
    async_frame = &queue->frames[queue->head];
    
    // Check if frame processing is complete
    result = queue->cu->cuEventQuery(async_frame->event_done);
    if (result == CUDA_SUCCESS) {
        // Frame is complete
        async_frame->in_use = 0;
        queue->count--;
        queue->head = (queue->head + 1) % queue->queue_size;
        
        CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
        return async_frame;
    } else if (result == CUDA_ERROR_NOT_READY) {
        // Frame not ready yet
        CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
        return NULL;
    } else {
        // Error occurred
        CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
        return NULL;
    }
}

int cuda_async_queue_wait_for_completion(CudaAsyncQueue *queue, CudaAsyncFrame *async_frame)
{
    CUcontext dummy;
    int ret;
    
    if (!queue || !async_frame || !queue->initialized)
        return AVERROR(EINVAL);
    
    ret = CHECK_CU(queue->cu->cuCtxPushCurrent(queue->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    
    // Wait for frame completion
    ret = CHECK_CU(queue->cu->cuEventSynchronize(async_frame->event_done));
    
    CHECK_CU(queue->cu->cuCtxPopCurrent(&dummy));
    return ret;
}

void cuda_async_queue_sync_all(CudaAsyncQueue *queue)
{
    int i;
    
    if (!queue || !queue->initialized)
        return;
    
    // Synchronize all streams
    for (i = 0; i < queue->num_streams; i++) {
        if (queue->streams[i])
            CHECK_CU(queue->cu->cuStreamSynchronize(queue->streams[i]));
    }
}