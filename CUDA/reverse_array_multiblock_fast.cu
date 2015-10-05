# include <iostream>
# include <time.h>
# include <stdio.h>
# include <assert.h>
// Simple utility function to check for CUDA runtime errors 
void checkCUDAError(const char* msg);
// Implement the kernel using shared memeory 
__global__ void reverseArrayBlock_shared_memo(int * d_out, int *d_in){
  
  extern __shared__ int s_data[];
  // (1) Note that the size is not indicated in the kernel -- rather it is obtained from the host through the execution configuration.
  // (2) This is shared pointer is available for all the threads inside the same block


  //************** Note that the use of shared memory requires manually generating the offset. 
  int inOffset = blockDim.x * blockIdx.x;
  int in = inOffset + threadIdx.x;
  
  // Load one element per thread from device memory and store it in reversed order into temporary shared memory 

  s_data[blockDim.x - 1 - threadIdx.x] = d_in[in];

  // block until all threads in the block have written their data to shared mem
  __syncthreads();

  //  write the data from shared memory in forward order but to the reversed block offset as before

  int outOffset = blockDim.x *(gridDim.x - 1 - blockIdx.x);
  int out = outOffset + threadIdx.x;
  d_out[out] = s_data[threadIdx.x];

}
int main(int argc, char ** argv){

  int *h_a;
  int dimA = 256*1024; // In my machine 1 int = 4 bytes therefore this is 256K elements (1MB size)

  // Pointer for device memory 
  int *d_a, *d_b;

  // define grid and block size
  int numThreadsPerBlock = 256;

  // Compute the number of blocks needed  based on array size and desired block size
  int numBlocks = dimA/numThreadsPerBlock;

  // allocate host and device memory
  size_t sharedMemSize = numThreadsPerBlock*sizeof(int);
  size_t memSize = numBlocks*sharedMemSize;
  
  h_a = (int *)malloc(memSize);
  // A key design feature of this program is that both arrays d_a and d_b reside in global memory on the device.
  cudaMalloc((void **)&d_a, memSize);
  cudaMalloc((void **)&d_b, memSize);

  // Initialize input array on host
  for(int i=0; i< dimA; i++){
    h_a[i] = i;
  }

  // Cpy host array to device arryr
  cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);

  // launch kernel 
  
  dim3 dimGrid(numBlocks);
  dim3 dimBlock(numThreadsPerBlock);
  
  clock_t device_start_shared_memo = clock();
  reverseArrayBlock_shared_memo<<<  dimGrid, dimBlock, sharedMemSize>>>(d_b, d_a);
  /*
    By default, the execution configuration assumes no shared memory is used. For example, in the host code of arrayReversal_multiblock_fast.cu, the following code snippet allocates shared memory for an array of integers containing a number of elements equal to the number of threads in a block:    
    here: sharedMemSize = numThreadsPerBlock*sizeof(int)
   */

  cudaThreadSynchronize();

  clock_t device_time_shared_memo = clock() - device_start_shared_memo;
  printf("Time elapsed on device using shared_memo: %f microseconds\n", (double)device_time_shared_memo/CLOCKS_PER_SEC/1000000);

  // check if kernel execution generated an error
  // check for any cuda errors
  checkCUDAError("kernel invocation");
  
  // This code returns me a cuda error:  unspecified launch failure.
  /*
    An unspecified launch failure is almost always a segfault. You've got an indexing mistake somewhere in your kernel, probably while accessing global memory.
  */
  // device to host copy 
  cudaMemcpy(h_a, d_b, memSize, cudaMemcpyDeviceToHost);
  
  // check for any CUDA erros
  checkCUDAError("memcpy");


  // verify the data returned to the host is correct
  for(int i=0; i<dimA; i++){
    assert( h_a[i] == dimA - 1 - i);
  }
  

  // free device memory
  cudaFree(d_a);
  cudaFree(d_b);

  // free host memory
  free(h_a);
  
  printf("Correct! \n");

  return 0;
  
}

void checkCUDAError( const char * msg){

  cudaError_t err = cudaGetLastError();
  // Properties of function "cudaGetLastError()" is discussed in the tutorial
  // Due to the asynchronous nature, the error get from here may not be the first error we met      
  if(cudaSuccess != err){
    fprintf(stderr, "Cuda error: %s: %s. \n ", msg, cudaGetErrorString(err));
    
    exit(EXIT_FAILURE);
  }

}
