#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_THREADS 5

#define N 1000
#define MEGEXTRA 1000000

pthread_attr_t attr;

void *do_work(void * thread_id){

  double A[N][N];
  int i,j;
  long tid;
  size_t my_stack_size;

  tid = (long)thread_id;
  pthread_attr_getstacksize(&attr, &my_stack_size);
  printf("Thread %ld: stack size = %li bytes \n", tid, my_stack_size);
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      A[i][j] = ((i*j)/3.452) + (N-i);
  printf("Thread %ld: work is done \n", tid );
  
  pthread_exit(NULL);
   
}

int main(int argc, char *argv[])
{
  pthread_t threads[NUM_THREADS];
  size_t stacksize;
  int rc;
  long t;
 
  pthread_attr_init(&attr);
  pthread_attr_getstacksize (&attr, &stacksize);// This 'attr' could have multiple purposes. 
  printf("Default stack size = %li\n", stacksize);
  stacksize = sizeof(double)*N*N+MEGEXTRA;
  printf("Amount of stack needed per thread = %li\n",stacksize);
  pthread_attr_setstacksize (&attr, stacksize);// Notice that not using reference  
  printf("Creating threads with stack size = %li bytes\n",stacksize);
   
  for(t=0; t<NUM_THREADS; t++){
    rc = pthread_create(&threads[t], &attr, do_work, (void *)t);
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
   
  printf("Created %ld threads.\n", t);
  pthread_exit(NULL);
}
