# CUDA Hello World Program

This directory contains a comprehensive CUDA Hello World program that demonstrates basic CUDA programming concepts.

## Files

- `cuda_hello_world.cpp` - Main CUDA program
- `build.bat` - Windows build script
- `build.sh` - Linux/Mac build script
- `README.md` - This file

## Program Features

The CUDA Hello World program demonstrates:

1. **Basic kernel execution** - Simple kernel that prints from GPU threads
2. **Memory management** - Allocating and copying data between CPU and GPU
3. **Parameter passing** - Passing data to GPU kernels
4. **Error handling** - Comprehensive CUDA error checking
5. **Device information** - Displaying GPU properties and capabilities

## Prerequisites

- CUDA Toolkit installed (version 10.0 or later recommended)
- NVIDIA GPU with CUDA support
- C++ compiler (gcc, clang, or MSVC)

## Compilation and Execution

### Windows
```bash
cd cuda
build.bat
```

### Linux/Mac
```bash
cd cuda
chmod +x build.sh
./build.sh
```

### Manual Compilation
```bash
nvcc -o cuda_hello_world cuda_hello_world.cpp
./cuda_hello_world  # Linux/Mac
cuda_hello_world.exe  # Windows
```

## Expected Output

The program will output:
- CPU hello message
- GPU device information
- Messages from GPU threads in different blocks
- Messages from GPU threads with parameters
- Success confirmation

## Key CUDA Concepts Demonstrated

1. **`__global__` functions** - GPU kernels
2. **Thread and block indexing** - `threadIdx.x`, `blockIdx.x`
3. **Memory allocation** - `cudaMalloc`, `cudaFree`
4. **Data transfer** - `cudaMemcpy`
5. **Synchronization** - `cudaDeviceSynchronize`
6. **Error checking** - CUDA error handling macros

## Troubleshooting

- **"No CUDA devices found"** - Ensure you have an NVIDIA GPU and CUDA drivers installed
- **"nvcc not found"** - Add CUDA Toolkit bin directory to your PATH
- **Compilation errors** - Check that your CUDA Toolkit version is compatible with your GPU

## Next Steps

After running this program successfully, you can explore:
- More complex kernel implementations
- Shared memory usage
- Multi-GPU programming
- CUDA streams and asynchronous execution 