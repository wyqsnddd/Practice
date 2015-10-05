#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
// No semicolum!!!
# define NUM_THREADS 4
# define VECLEN 100

/*
  The following structure contains the necessary information  
  to allow the function "dotprod" to access its input data and 
  place its output into the structure.
*/

typedef struct{
  double *a;
  double *b;
  double sum;
  int veclen;
}dot_data;


// Globally accessible variables and the mutex to be used. 
dot_data dotstr;

pthread_t call_thread[NUM_THREADS];
// This is the mutex variable 
pthread_mutex_t mutex_sum; 

void *dot_prod(void *arg){

  /* Define and use local variables for convenience */

  int i, start, end, len;
  long offset;
  double mysum, *x, *y;
  offset = (long)arg;
     
  len = dotstr.veclen;

  start = offset*len;
  end   = start + len;
  x = dotstr.a;
  y = dotstr.b;
   
  mysum = 0;
  for(i=start; i<end ; i++){
    mysum += (x[i] * y[i]);
  }

  pthread_mutex_lock(&mutex_sum);
  // If the mutex is already locked by another thread, this call will block the calling thread until the mutex is unlocked.
  dotstr.sum += mysum;
  printf("Thread %ld did %d to %d:  mysum=%f global sum=%f\n",offset,start,end,mysum,dotstr.sum);
  pthread_mutex_unlock(&mutex_sum);
   
  pthread_exit((void *) 0);

}

/*
  The main program creates threads which do all the work and then 
  print out result upon completion. Before creating the threads,
  the input data is created. Since all threads update a shared structure, 
  we need a mutex for mutual exclusion. The main thread needs to wait for
  all threads to complete, it waits for each one of the threads. We specify
  a thread attribute value that allow the main thread to join with the
  threads it creates. Note also that we free up handles when they are
  no longer needed.
*/

int main(int argc, char*argv[]){
  long i;

  double *a, *b;

  void *status;
  pthread_attr_t attr;

  /* Assign storage and initialize values */
  // The data needs to be in the heap such that it is accessible to each thread.
  a = (double*) malloc (NUM_THREADS*VECLEN*sizeof(double));
  b = (double*) malloc (NUM_THREADS*VECLEN*sizeof(double));
  for(i = 0; i<NUM_THREADS*VECLEN; i++){
    a[i] = 1.0;
    b[i] = a[i];
  }


  dotstr.a = a;
  dotstr.b = b;
  dotstr.sum = 0;
  dotstr.veclen = VECLEN;

  pthread_mutex_init(&mutex_sum,NULL);
  

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  

  for(i = 0; i< NUM_THREADS; i++){
    pthread_create(&call_thread[i],&attr,dot_prod,(void *)i);
  }
  
  pthread_attr_destroy(&attr);

  // wait for the threads
  for(i = 0; i< NUM_THREADS; i++){
    pthread_join(call_thread[i], &status);
  }

  // after joining, print out the result and cleanup
  
  printf("Sum =  %f \n", dotstr.sum);
  free(a);
  free(b);

  pthread_mutex_destroy(&mutex_sum);
  pthread_exit(NULL);
}
