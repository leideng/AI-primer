#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel function - runs on GPU
__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d, block %d!\n", 
           threadIdx.x, blockIdx.x);
}

// CUDA kernel function with parameters
__global__ void helloWithParams(char* message, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        printf("GPU Thread %d: %s\n", idx, message);
    }
}

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    printf("Hello World from CPU!\n");
    
    // Check CUDA device availability
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return -1;
    }
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    // Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per grid: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    // Launch simple kernel
    printf("\n=== Simple Kernel Execution ===\n");
    helloFromGPU<<<2, 4>>>();
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Launch kernel with parameters
    printf("\n=== Kernel with Parameters ===\n");
    char message[] = "Hello from CUDA!";
    int messageLength = strlen(message);
    
    // Allocate memory on GPU
    char* d_message;
    CHECK_CUDA_ERROR(cudaMalloc(&d_message, messageLength + 1));
    
    // Copy data from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_message, message, messageLength + 1, 
                                cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 4;
    int gridSize = (messageLength + blockSize - 1) / blockSize;
    helloWithParams<<<gridSize, blockSize>>>(d_message, messageLength);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_message));
    
    printf("\n=== Program completed successfully! ===\n");
    
    return 0;
}
