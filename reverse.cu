#include <stdint.h>
#include <stdio.h>

#define N 34
#define THREADS_PER_BLOCK 32

__global__ void reverse(int* x) {
  size_t index = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
  if (index < N/2) {
    int temp = x[index];
    x[index] = x[N-1-index];
    x[N-1-index] = temp;
    }
}

int main() {
  // Allocate arrays for X and Y on the CPU
  int* cpu_x = (int*)malloc(sizeof(int) * N);
  
  // Initialize X and Y
  int i;
  for(i=0; i<N; i++) {
    cpu_x[i] = i;
  }
  
  // Allocate space for X and Y on the GPU
  int* gpu_x;
  
  if(cudaMalloc(&gpu_x, sizeof(int) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }
  
  
  // Copy the host X and Y arrays to the device X and Y arrays
  if(cudaMemcpy(gpu_x, cpu_x, sizeof(int) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X to the GPU\n");
  }
  

  // How many blocks should be run, rounding up to include all threads?
  size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  
  // Run the saxpy kernel
  reverse<<<blocks, THREADS_PER_BLOCK>>>(gpu_x);
  
  // Wait for the kernel to finish
  cudaDeviceSynchronize();
  
  // Copy values from the GPU back to the CPU
  if(cudaMemcpy(cpu_x, gpu_x,sizeof(int) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X from the GPU\n");
  }
  
  for(i=0; i<N; i++) {
    printf("%d: %d\n", i, cpu_x[i]);
  }
  
  cudaFree(gpu_x);
  free(cpu_x);
  
  return 0;
}
