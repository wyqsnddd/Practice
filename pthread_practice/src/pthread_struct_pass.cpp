#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Make sure that all passed data is thread safe - that it can not be changed by other threads. 
// In this example, Each thread receives a unique instance of the structure. So this example is thread-safe.

#define NUM_THREADS 5

struct thread_data{
  int thread_id;
  int sum;

  char * message;
};

struct thread_data thread_data_array[NUM_THREADS]; // When use a user-defined struct-type we need the keyword "struct" before the user-defined struct-type.

void *print_hello(void *thread_args){
  
  struct thread_data *my_data; 
  
  my_data = (struct thread_data *)thread_args; // cast the void/generic pointer before use it.  
  pthread_t calling_id =  pthread_self();
  
  printf("Hello world, this is thread: %d, the sum is: %d, the message is: %s. Calling_id: %x \n", my_data->thread_id, my_data->sum, my_data->message, calling_id);
  pthread_exit(NULL);

}

int main(int agc, char *argv[]){

  pthread_t threads[NUM_THREADS];
  
  int status;
  for(long t = 0; t < NUM_THREADS; t++){
    thread_data_array[t].thread_id = t;
    thread_data_array[t].sum = t+1;
    thread_data_array[t].message = "Anything is ok.";
    printf("In main: Creating thread %ld\n", t);
    
    status = pthread_create(&threads[t], NULL, print_hello, (void *)&thread_data_array[t] ); 
    // (void *)&thread_data_array[t] is two parts: cast "&thread_data_array[t]" into a void pointer
    // arg#4 It MUST be passed by reference as a pointer cast of type void. NULL may be used if no argument is to be passed.
    // arg#4 cast "&thread_data_array[t]" into "void *"
    if(status){
      printf("error! return code from pthread_create() is %d\n", status);
      exit(-1);
    }
  }// end of for 

  // Exit: last thing that main() should do, because:   
  // By having main() explicitly call pthread_exit() as the last thing it does, main() will block and be kept alive to SUPPORT the threads it created until they are DONE. 
  pthread_exit(NULL);
}
