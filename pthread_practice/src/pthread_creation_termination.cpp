#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 5


void *print_hello(void *thread_id){
  long tid;
  
  tid = (long)thread_id;
  
  printf("Hello world, this is thread: %ld\n", tid);
  
  pthread_exit(NULL);

}

int main(int agc, char *argv[]){


  pthread_t threads[NUM_THREADS];

  long *task_ids[NUM_THREADS];// Array of long pointers 
  
  int status;
  for(long t = 0; t < NUM_THREADS; t++){

    task_ids[t] = (long *)malloc(sizeof(long));
    *task_ids[t] = t;
    printf("In main: Creating thread %ld\n", *task_ids[t]);
  
    status = pthread_create(&threads[t], NULL, print_hello, (void *)*task_ids[t] );// This would passs the value (in the form of a void pointer)  
    //    arg#1: pthread_t *thread
    //    arg#2: const pthread_attr_t *attr
    //    status = pthread_create(&threads[t], NULL, print_hello, (void *)task_ids[t] );// This would pass the address, because task_ids is an array of pointers   
	
    // arg#4 It must be passed by reference as a pointer cast of type void. NULL may be used if no argument is to be passed.
    
    if(status){
      printf("error! return code from pthread_create() is %d\n", status);
      exit(-1);
    }
  }// end of for 

  // Exit: last thing that main() should do, because:   
  // By having main() explicitly call pthread_exit() as the last thing it does, main() will block and be kept alive to SUPPORT the threads it created until they are DONE. 
  pthread_exit(NULL);
}
