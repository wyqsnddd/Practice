# include <stdio.h>
# include <assert.h>
# include <cuda.h>

int main(void){

  float * a_h, * b_h; // pointers to host memory
  float * a_d, * b_d; // pointers to dvice memory
  int N = 14;
  int i;

  // allocate arrays on host 
  a_h = (float *)malloc(sizeof(float)*N);
  b_h = (float *)malloc(sizeof(float)*N);
  
  // allocate arrays on device 
  cudaMalloc((void **) &a_d, sizeof(float)*N);
  cudaMalloc((void **) &b_d, sizeof(float)*N);
  
  // initialize host data

  for(i = 0; i<N; i++){
    a_h[i] = 10.f + i;
    b_h[i] = 0.f;
    printf("a_h[%d] is: %f\n", i, a_h[i] );
    printf("b_h[%d] is: %f\n", i, b_h[i] );
    
  }
  // send data from host to device a_h to a_d

  cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, a_d, sizeof(float)*N, cudaMemcpyDeviceToDevice);
  cudaMemcpy(b_h, b_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

  // check result

  for(i = 0; i<N; i++){
    assert(a_h[i] == b_h[i]);
  }

  // clean up
  free(a_h); 
  free(b_h);
  
  cudaFree(a_d); 
  cudaFree(b_d);
}

