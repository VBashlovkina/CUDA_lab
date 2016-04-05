#include <stdint.h>
#include <stdio.h>

#define N 34
#define THREADS_PER_BLOCK 32

__global__ void reduce(float* x, float* result) {

  __shared__ int counter = 1;

  while(counter < N && counter < 32) {
    // Compute the index this thread should use to access elements
    size_t index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    if(index < N && index%(counter*2) == 0) {
      x[index] = x[index] + x[index+counter]; //gopalasw
    }
    if(threadIdx.x == 0) {
      __syncthreads();
      counter *= 2;
    }
  }
    
  // Add the sum for this block to the result
  atomicAdd(result, sum);
}
   
int main() {
  // Allocate arrays for X and Y on the CPU
  float* cpu_x = (float*)malloc(sizeof(float) * N);
  float* cpu_result = (float*) malloc(sizeof(float));

  *cpu_result = 0.0;
  
  // Initialize X and Y
  int i;
  for(i=0; i<N; i++) {
    cpu_x[i] = (float)(i);
  }
  
  // Allocate space for X and Y on the GPU
  float* gpu_x;
  float* gpu_result;
  
  if(cudaMalloc(&gpu_x, sizeof(float) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }

  if(cudaMalloc(&gpu_result, sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate result value on GPU\n");
    exit(2);
  }
  
  // Copy the host X and Y arrays to the device X and Y arrays
  if(cudaMemcpy(gpu_x, cpu_x, sizeof(float) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X to the GPU\n");
  }

  if(cudaMemcpy(gpu_result, cpu_result, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy result to the GPU\n");
  }

  // How many blocks should be run, rounding up to include all threads?
  size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  
  // Run the saxpy kernel
  dotproduct<<<blocks, THREADS_PER_BLOCK>>>(gpu_x, gpu_result);
  
  // Wait for the kernel to finish
  cudaDeviceSynchronize();

  
  // Copy values from the GPU back to the CPU
  if(cudaMemcpy(cpu_result, gpu_result, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy result from the GPU\n");
  }

  for(int i = 0; i < N; i++) {
    printf(" %.1f ", cpu_x[i]);
  }
  printf("\nResult: %.1f\n", *cpu_result);
  
  cudaFree(gpu_x);
  cudaFree(gpu_result);
  free(cpu_x);
  free(cpu_result);
  
  return 0;
}
