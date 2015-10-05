# include <time.h>
# include <stdio.h>
# include <assert.h>

// Simple utility function to check for CUDA runtime errors 
void checkCUDAError(const char* msg);

// Implement the kernel 

__global__ void reverseArrayBlock(int *d_out, int *d_in){
  int inOffset = blockDim.x * blockIdx.x;
  int outOffset = blockDim.x *(gridDim.x - 1 - blockIdx.x);

  int in = inOffset + threadIdx.x;
  int out = outOffset + (blockDim.x  - 1 - threadIdx.x);

  d_out[out] = d_in[in];
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
  size_t memSize = numBlocks*numThreadsPerBlock*sizeof(int);
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
  
  clock_t device_start = clock();
  
  reverseArrayBlock<<<  dimGrid, dimBlock>>>(d_b, d_a);

  //  block until the device has completed
  cudaThreadSynchronize();
  /*
    Blocks until the device has completed all preceding requested tasks. cudaThreadSynchronize() returns an error if one of the preceding tasks has failed. If the cudaDeviceScheduleBlockingSync flag was set for this device, the host thread will block until the device has finished its work.
  */
  clock_t device_time = (clock() - device_start);
  printf("Time elapsed on device: %f microseconds\n", (double)device_time/CLOCKS_PER_SEC/1000000);

  // check if kernel execution generated an error
  // check for any cuda errors
  checkCUDAError("kernel invocation");
  
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
