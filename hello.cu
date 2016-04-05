#include <stdio.h>

__global__ void kernel() {
  printf("Hello from block %d thread %d\n", blockIdx.x, threadIdx.x);       
}

int main() {
  kernel<<<4,6>>>();
  cudaDeviceSynchronize();
  return 0;
}
