/*
 * CUDA Async Queue - Implementation for asynchronous CUDA frame processing
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

#include <string.h>
#include <sys/time.h>

#include "cuda_async_queue.h"
#include "libavutil/avassert.h"
#include "libavutil/log.h"
#include "libavutil/error.h"
#include "libavutil/hwcontext_cuda_internal.h"

#define CHECK_CU(x) do { \
    CUresult res = x; \
    if (res != CUDA_SUCCESS) { \
        av_log(NULL, AV_LOG_ERROR, "CUDA error: %d at %s:%d\n", res, __FILE__, __LINE__); \
        return AVERROR_EXTERNAL; \
    } \
} while(0)

static int64_t get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

int ff_cuda_async_queue_init(CudaAsyncQueue * queue,
                             AVCUDADeviceContext * hwctx, int queue_size,
                             int num_streams, void *filter_ctx,
                             int (*process_frame)(void *, AVFrame *,
                                                  AVFrame *, CUstream))
{

    if(!queue || !hwctx || !filter_ctx || !process_frame)
        return AVERROR(EINVAL);

    if (queue_size < 1 || queue_size > MAX_FRAME_QUEUE_SIZE)
        return AVERROR(EINVAL);

    if (num_streams < 1 || num_streams > MAX_CUDA_STREAMS)
        return AVERROR(EINVAL);

    memset(queue, 0, sizeof(*queue));

    queue->hwctx = hwctx;
    queue->cuda_ctx = hwctx->cuda_ctx;
    queue->queue_size = queue_size;
    queue->num_streams = num_streams;
    queue->filter_ctx = filter_ctx;
    queue->process_frame = process_frame;

    CudaFunctions *cu = queue->hwctx->internal->cuda_dl;

    CHECK_CU(cu->cuCtxPushCurrent(queue->cuda_ctx));

    for (int i = 0; i < num_streams; i++) {
        CHECK_CU(cu->
                 cuStreamCreate(&queue->streams[i],
                                CU_STREAM_NON_BLOCKING));
    }

    for (int i = 0; i < queue_size; i++) {
        CHECK_CU(cu->
                 cuEventCreate(&queue->frames[i].event_start,
                               CU_EVENT_DISABLE_TIMING));
        CHECK_CU(cu->
                 cuEventCreate(&queue->frames[i].event_done,
                               CU_EVENT_DISABLE_TIMING));
        queue->frames[i].in_use = 0;
        queue->frames[i].stream_idx = -1;
    }

    CHECK_CU(cu->cuCtxPopCurrent(NULL));

    av_log(NULL, AV_LOG_DEBUG,
           "Async queue initialized: depth=%d streams=%d\n", queue_size,
           num_streams);

    return 0;
}

void ff_cuda_async_queue_uninit(CudaAsyncQueue * queue)
{
    if (!queue || !queue->cuda_ctx)
        return;

    CudaFunctions *cu = queue->hwctx->internal->cuda_dl;

    cu->cuCtxPushCurrent(queue->cuda_ctx);

    // First, synchronize all streams to ensure all operations complete
    for (int i = 0; i < queue->num_streams; i++) {
        if (queue->streams[i]) {
            cu->cuStreamSynchronize(queue->streams[i]);
        }
    }

    ff_cuda_async_queue_flush(queue);

    for (int i = 0; i < queue->num_streams; i++) {
        if (queue->streams[i]) {
            cu->cuStreamDestroy(queue->streams[i]);
        }
    }

    for (int i = 0; i < queue->queue_size; i++) {
        // Synchronize events before destroying to prevent memory leaks
        if (queue->frames[i].event_start) {
            cu->cuEventSynchronize(queue->frames[i].event_start);
            cu->cuEventDestroy(queue->frames[i].event_start);
        }
        if (queue->frames[i].event_done) {
            cu->cuEventSynchronize(queue->frames[i].event_done);
            cu->cuEventDestroy(queue->frames[i].event_done);
        }

        // Ensure frames are properly unreferenced before freeing
        if (queue->frames[i].input_frame) {
            // For input frames, unreference all frame data including hardware buffers
            av_frame_unref(queue->frames[i].input_frame);
            av_frame_free(&queue->frames[i].input_frame);
        }
        if (queue->frames[i].output_frame) {
            // For output frames, ensure hardware frame buffers are unreferenced
            // This is critical for CUDA frames allocated via av_hwframe_get_buffer()
            av_frame_unref(queue->frames[i].output_frame);
            av_frame_free(&queue->frames[i].output_frame);
        }
    }

    cu->cuCtxPopCurrent(NULL);

    memset(queue, 0, sizeof(*queue));
}

int ff_cuda_async_queue_submit(CudaAsyncQueue * queue, AVFrame * in_frame)
{
    CudaAsyncFrame *async_frame;
    AVFrame *out_frame;
    int ret;
    int stream_idx;

    if (!queue || !in_frame)
        return AVERROR(EINVAL);

    if (ff_cuda_async_queue_is_full(queue))
        return AVERROR(EAGAIN);

    CudaFunctions *cu = queue->hwctx->internal->cuda_dl;

    CHECK_CU(cu->cuCtxPushCurrent(queue->cuda_ctx));

    async_frame = &queue->frames[queue->write_idx];
    av_assert0(!async_frame->in_use);

    async_frame->input_frame = av_frame_clone(in_frame);
    if (!async_frame->input_frame) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    out_frame = av_frame_alloc();
    if (!out_frame) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    // Frame is already clean from av_frame_alloc(), no need to unref
    
    stream_idx = queue->submission_counter % queue->num_streams;
    async_frame->stream_idx = stream_idx;
    async_frame->submission_order = queue->submission_counter++;
    async_frame->submit_time_us = get_time_us();

    CHECK_CU(cu->
             cuEventRecord(async_frame->event_start,
                           queue->streams[stream_idx]));

    av_log(NULL, AV_LOG_DEBUG,
           "[CUDA_ASYNC] Frame %" PRId64
           " submitted to stream %d (queue slot %d) at time %" PRId64 "\n",
           async_frame->submission_order, stream_idx, queue->write_idx,
           async_frame->submit_time_us);

    ret =
        queue->process_frame(queue->filter_ctx, out_frame,
                             async_frame->input_frame,
                             queue->streams[stream_idx]);
    if (ret < 0) {
        av_frame_free(&out_frame);
        goto fail;
    }
    // Don't synchronize here - let the kernel run asynchronously
    // The synchronization will happen in ff_cuda_async_queue_receive

    async_frame->output_frame = out_frame;

    CHECK_CU(cu->
             cuEventRecord(async_frame->event_done,
                           queue->streams[stream_idx]));

    av_log(NULL, AV_LOG_DEBUG,
           "[CUDA_ASYNC] Frame %" PRId64
           " processing initiated on stream %d\n",
           async_frame->submission_order, stream_idx);

    async_frame->in_use = 1;
    queue->write_idx = (queue->write_idx + 1) % queue->queue_size;
    queue->frames_in_queue++;

    // Log current queue status to show parallel activity
    av_log(NULL, AV_LOG_DEBUG,
           "[CUDA_ASYNC] Queue status: %d/%d frames active\n",
           queue->frames_in_queue, queue->queue_size);

    CHECK_CU(cu->cuCtxPopCurrent(NULL));

    return 0;

  fail:
    av_frame_free(&async_frame->input_frame);
    cu->cuCtxPopCurrent(NULL);
    return ret;
}

int ff_cuda_async_queue_receive(CudaAsyncQueue * queue,
                                AVFrame ** out_frame)
{
    CudaAsyncFrame *async_frame;
    CUresult cu_res;
    int ret = 0;

    if (!queue || !out_frame)
        return AVERROR(EINVAL);

    *out_frame = NULL;

    if (ff_cuda_async_queue_is_empty(queue))
        return AVERROR(EAGAIN);

    CudaFunctions *cu = queue->hwctx->internal->cuda_dl;

    CHECK_CU(cu->cuCtxPushCurrent(queue->cuda_ctx));

    async_frame = &queue->frames[queue->read_idx];
    av_assert0(async_frame->in_use);

    cu_res = cu->cuEventQuery(async_frame->event_done);
    if (cu_res == CUDA_ERROR_NOT_READY) {
        ret = AVERROR(EAGAIN);
        goto done;
    } else if (cu_res != CUDA_SUCCESS) {
        av_log(NULL, AV_LOG_ERROR, "CUDA error querying event: %d\n",
               cu_res);
        ret = AVERROR_EXTERNAL;
        goto done;
    }

    CHECK_CU(cu->cuEventSynchronize(async_frame->event_done));

    async_frame->complete_time_us = get_time_us();
    int64_t processing_time =
        async_frame->complete_time_us - async_frame->submit_time_us;

    av_log(NULL, AV_LOG_DEBUG,
           "[CUDA_ASYNC] Frame %" PRId64
           " completed on stream %d (queue slot %d) at time %" PRId64
           " - processing took %" PRId64 "us\n",
           async_frame->submission_order, async_frame->stream_idx,
           queue->read_idx, async_frame->complete_time_us,
           processing_time);

    *out_frame = async_frame->output_frame;
    async_frame->output_frame = NULL;

    av_frame_free(&async_frame->input_frame);
    async_frame->in_use = 0;
    async_frame->stream_idx = -1;

    queue->read_idx = (queue->read_idx + 1) % queue->queue_size;
    queue->frames_in_queue--;

  done:
    CHECK_CU(cu->cuCtxPopCurrent(NULL));
    return ret;
}

int ff_cuda_async_queue_flush(CudaAsyncQueue * queue)
{
    AVFrame *frame;
    int ret;

    if (!queue)
        return AVERROR(EINVAL);

    CudaFunctions *cu = queue->hwctx->internal->cuda_dl;

    CHECK_CU(cu->cuCtxPushCurrent(queue->cuda_ctx));

    for (int i = 0; i < queue->num_streams; i++) {
        if (queue->streams[i]) {
            CHECK_CU(cu->cuStreamSynchronize(queue->streams[i]));
        }
    }

    while (!ff_cuda_async_queue_is_empty(queue)) {
        ret = ff_cuda_async_queue_receive(queue, &frame);
        if (ret == 0 && frame) {
            av_frame_free(&frame);
        } else if (ret < 0 && ret != AVERROR(EAGAIN)) {
            // Force cleanup of remaining frames
            for (int i = 0; i < queue->queue_size; i++) {
                if (queue->frames[i].in_use) {
                    // Properly unreference frames before freeing
                    if (queue->frames[i].input_frame) {
                        av_frame_unref(queue->frames[i].input_frame);
                        av_frame_free(&queue->frames[i].input_frame);
                    }
                    if (queue->frames[i].output_frame) {
                        av_frame_unref(queue->frames[i].output_frame);
                        av_frame_free(&queue->frames[i].output_frame);
                    }
                    queue->frames[i].in_use = 0;
                }
            }
            queue->frames_in_queue = 0;
            queue->read_idx = queue->write_idx = 0;
            break;
        }
    }

    CHECK_CU(cu->cuCtxPopCurrent(NULL));

    return 0;
}

int ff_cuda_async_queue_is_full(CudaAsyncQueue * queue)
{
    return queue->frames_in_queue >= queue->queue_size;
}

int ff_cuda_async_queue_is_empty(CudaAsyncQueue * queue)
{
    return queue->frames_in_queue == 0;
}
