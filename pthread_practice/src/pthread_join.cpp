#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_THREADS 5



void *busy_work(void *t)
{
   int i;
   long tid;
   double result=0.0;
   tid = (long)t; // Notice that int can not catch the (void* t) 
   printf("Thread %ld starting...\n",tid);
   for (i=0; i<1000000; i++)
   {
      result = result + sin(i) * tan(i);
   }
   printf("Thread %ld done. Result = %e, t is: %ld\n",tid, result, (long)t);
   pthread_exit((void*) t);
}


int main(int argc, char *argv[]){

  pthread_t thread[NUM_THREADS];

  pthread_attr_t attr;
  // (1) pthread_attr_t is used to explicitly create a thread as joinable or detached. The routine is: (1)pthread_attr_t attr. (2)pthread_attr_init(&attr); (3)pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); or pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED); (4)when done: pthread_attr_destroy(&attr) 
  // (2) blocks the calling thread until the specified threadid thread terminates.
  
  int status;
  // Initialize and sed thread detached attribute 
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for(long t=0; t<NUM_THREADS; t++){
    printf("In main: creaeting thread %ld\n", t);
    status = pthread_create(&thread[t], &attr, busy_work, (void *)t);
    // arg#4 is essentially a piece of data in the form of a void pointer. 
    if(status){
      printf("ERROR; return code from pthread_create() is %d\n", status);
      exit(-1);
    }
  }
  
  // Free attribute and wait for the other threads   
  pthread_attr_destroy(&attr);
  void *status_ptr;  

  for(long t= 0; t<NUM_THREADS; t++ ){
    status  = pthread_join(thread[t], &status_ptr);
    // blocks the calling thread until the specified threadid thread terminates.
    // blocks the 'main' thread until  the 't'th thread terminates.
    if(status){
      printf("ERROR; return code from pthread_join() is %d\n", status);
      exit(-1);
    }
    
    printf("In main: completed join with thread %ld and having a status of %ld\n", t, (long)status_ptr);// Error: cast from void to int loses precision. 
  }
  
  printf("In main: all done, exiting \n");
  pthread_exit(NULL);// Waiting for all the threads to be completed. 
}
