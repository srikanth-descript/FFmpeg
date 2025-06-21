# CUDA Async Processing Improvements for FFmpeg

## Overview

This implementation adds multiple frame queuing and multiple CUDA streams to the `tonemap_cuda` and `colorspace_cuda` filters to maximize GPU utilization. The previous implementation was achieving only ~6% GPU usage due to synchronous single-frame processing.

## Key Improvements

### 1. Multi-Stream Architecture
- **Multiple CUDA Streams**: 1-4 concurrent streams per filter (configurable)
- **Overlapping Computation**: Kernels execute concurrently on different streams
- **Pipeline Parallelism**: Memory transfers and computation overlap

### 2. Frame Queue Management
- **Circular Buffer**: 1-8 frame queue depth (configurable)
- **Async Processing**: Non-blocking frame submission and retrieval
- **Event-Based Synchronization**: CUDA events track completion

### 3. Configuration Options
- `async_depth`: Frame queue depth (1-8, default: 4)
- `async_streams`: Number of CUDA streams (1-4, default: 2)
- Backward compatible: `async_depth=1` and `async_streams=1` = original behavior

## Architecture

### Files Added
- `libavfilter/cuda_async_queue.h` - Async queue interface
- `libavfilter/cuda_async_queue.c` - Queue implementation

### Files Modified
- `libavfilter/vf_tonemap_cuda.c` - Added async processing
- `libavfilter/vf_colorspace_cuda.c` - Added async processing

### Core Components

#### CudaAsyncQueue Structure
```c
typedef struct CudaAsyncQueue {
    CudaAsyncFrame frames[MAX_FRAME_QUEUE_SIZE];
    CUstream streams[MAX_CUDA_STREAMS];
    int queue_size;
    int num_streams;
    // ... circular buffer management
} CudaAsyncQueue;
```

#### Async Frame Processing
```c
typedef struct CudaAsyncFrame {
    AVFrame *frame;
    AVFrame *output_frame;
    CUevent event_start;
    CUevent event_done;
    int in_use;
    int stream_idx;
} CudaAsyncFrame;
```

## Usage Examples

### Basic Async Processing
```bash
ffmpeg -hwaccel cuda -i input.mp4 \
  -vf "tonemap_cuda=tonemap=hable:async_depth=4:async_streams=2" \
  -c:v h264_nvenc output.mp4
```

### Maximum Performance
```bash
ffmpeg -hwaccel cuda -i input.mp4 \
  -vf "tonemap_cuda=tonemap=hable:async_depth=8:async_streams=4" \
  -c:v h264_nvenc output.mp4
```

### Colorspace Conversion
```bash
ffmpeg -hwaccel cuda -i input.mp4 \
  -vf "colorspace_cuda=space=bt2020nc:async_depth=4:async_streams=2" \
  -c:v h264_nvenc output.mp4
```

### Disable Async (Original Behavior)
```bash
ffmpeg -hwaccel cuda -i input.mp4 \
  -vf "tonemap_cuda=tonemap=hable:async_depth=1:async_streams=1" \
  -c:v h264_nvenc output.mp4
```

## Performance Benefits

### Expected Improvements
1. **GPU Utilization**: 6% → 60-90% (10-15x improvement)
2. **Throughput**: 2-4x faster processing for GPU-bound workloads
3. **Pipeline Efficiency**: Overlapped memory transfers and computation
4. **Scalability**: Performance scales with number of streams

### Monitoring Performance
```bash
# Monitor GPU utilization during processing
nvidia-smi -l 1

# Expected to see much higher GPU utilization with async processing
```

## Technical Details

### Processing Pipeline
1. **Frame Submission**: Frames queued asynchronously
2. **Stream Assignment**: Round-robin stream assignment
3. **Kernel Launch**: Non-blocking CUDA kernel execution
4. **Event Recording**: Completion events tracked per frame
5. **Frame Retrieval**: Completed frames returned when ready

### Memory Management
- Pre-allocated frame buffers for zero-copy operations
- CUDA events for precise synchronization
- Circular buffer prevents memory leaks

### Error Handling
- Graceful fallback to synchronous processing on errors
- Proper cleanup of CUDA resources
- Event synchronization ensures data consistency

## Testing

Run the performance test script:
```bash
./test_async_performance.sh
```

This will compare:
- Synchronous vs async processing times
- Different queue depths and stream counts
- Both tonemap_cuda and colorspace_cuda filters

## Backward Compatibility

- Default behavior unchanged when async parameters not specified
- `async_depth=1` and `async_streams=1` = original synchronous mode
- All existing command lines continue to work
- No performance regression for single-stream usage

## Future Optimizations

1. **Dynamic Stream Count**: Adjust streams based on workload
2. **Memory Pool**: Pre-allocated frame buffer pools
3. **Multi-GPU**: Distribute across multiple GPUs
4. **Profile-Guided**: Automatic parameter tuning
5. **Cross-Filter**: Share async queues between filters

## Debugging

Enable debug logging:
```bash
ffmpeg -loglevel debug -hwaccel cuda -i input.mp4 \
  -vf "tonemap_cuda=async_depth=4:async_streams=2" output.mp4
```

Look for log messages:
- "Async processing enabled: depth=X streams=Y"
- Stream creation/destruction events
- Frame queue status

## Notes

- Requires CUDA-capable GPU with compute capability 3.0+
- Performance gains are most significant for GPU-bound workloads
- CPU-bound scenarios may see minimal improvement
- Memory usage increases with queue depth (typically ~50-200MB)