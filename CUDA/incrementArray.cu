# include <stdio.h>
# include <assert.h>
# include <cuda.h>
# include <time.h>

void incrementArrayOnHost(float *a, int N){
  int i;
  for(i=0; i<N;i++){
    a[i] = a[i] + 1.f;
  }
}

__global__ void incrementArrayOnDevice(float *a, int N){// N is used to check idx 
  // This function is simultaneously executed by an array of threads on the CUDA device.   
  // This "block" is an array, so here we simply calculates the array ID.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  /*
    Here we have some built-in variables: 
    (1) "blockIdx" contains the block index within the grid. 
    (2) "blockDim" contains the number of threads in a block
    (3) "threadIdx" contains the thread index within the block
    
    These variables are structures that contain integer components of the variables. Blocks, for example, have x-, y-, and z- integer components because they are three-dimensional

    "idx" is then used to uniquely reference each element in the array 
   */
  
  if(idx<N){
    /*
      each thread is provided with a unique ID that can be used to compute different array indicies or make control decisions (such as not doing anything if the thread index exceeds the array size).
      Since the number of threads can be larger than the size of the array, idx is first checked against N, an argument passed to the kernel that specifies the number of elements in the array, to see if any work needs to be done.
     */
    a[idx] = a[idx] + 1.f;
  }
 
}
/*
  The function type qualifier __global__ declares a function as being an executable kernel on the CUDA device, which can only be called from the host.
  
  All kernels must be declared with a return type void 
 */

int main(void){


  float *a_h, * b_h;
  float *a_d;

  int i, N = 10000000;
  size_t size = N*sizeof(float);

  a_h = (float *)malloc(size);
  b_h = (float *)malloc(size);
  cudaMalloc((void**) &a_d, size);
  // initialize the host data 
  for(i=0; i<N; i++){
    a_h[i] = (float) i;
  }
  
  // copy data from host to device 
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  
  clock_t start_host = clock();
  // do calculation on host 
  incrementArrayOnHost(a_h, N);
  
  printf("Time elapsed on host: %f milliseconds\n", (double)(clock() - start_host)/(CLOCKS_PER_SEC / 1000));
  // do calculation on device. 

  // (1) Compute execution configuration 
  int blockSize = 400; 
  // Threads inside a block are able to cooperate.
  int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
  // In cases where N is not evenly divisible by blockSize, the last term in the nBlocks calculation adds an extra block, which for some cases implies some threads in the block will not perform any useful work.


  
  // (2) call incrementArrayOnDevice kernel

  /*A kernel is a function callable from the host and executed on the CUDA device -- simultaneously by many threads in parallel.
   */
  
  clock_t start_device = clock();

  
  incrementArrayOnDevice <<< nBlocks, blockSize>>>(a_d,N);
  /*
    How to execute a kernel? 
    1. specifying the name of the kernel plus an execution configuration
    2. number of parallel threads in a group/block (blockSize) and the number of groups/blocks in the grid(nBlocks)
    3. Synchronization: 
    Meanwhile, the host continues to the next line of code after the kernel launch. At this point, both the CUDA device and host are simultaneously running their separate programs. In the case of incrementArrays.cu, the host immediately calls cudaMemcpy, which waits until all threads have finished on the device (e.g., returned from incrementArrayOnDevice) after which it pulls the modified array back to the host
   */
  
  cudaThreadSynchronize();

  // retrieve resutl from device and store in b_h
  printf("Time elapsed on device: %f milliseconds\n", (double)(clock() - start_device)/(CLOCKS_PER_SEC / 1000));
  
  cudaMemcpy(b_h, a_d, size, cudaMemcpyDeviceToHost);
  // check result 

  for(i=0; i<N;i++){
    assert(a_h[i] == b_h[i]);
  }
  // clean up
  free(a_h);
  free(b_h);
  cudaFree(a_d);
  
}


// Overall comments 
/*

  kernel calls are asynchronous -- after a kernel launch, control immediately returns to the host CPU. The kernel will run on the CUDA device once all previous CUDA calls have finished. 

The asynchronous kernel call is a wonderful way to overlap computation on the host and device. In this example, the call to incrementArrayOnHost could be placed after the call to incrementArrayOnDevice to overlap computation on the host and device to get better performance. Depending on the amount of time the kernel takes to complete, it is possible for both host and device to compute simultaneously.

*/
