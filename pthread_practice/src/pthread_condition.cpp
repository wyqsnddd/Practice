#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS 3
#define T_COUNT 10
#define COUNT_LIMIT 12

int count = 0;

pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

void *watch_count(void * id){

  long my_id = (long)id;
  
  printf("Starting watch_count(): thread %ld\n", my_id);
  

  /*
  Lock mutex and wait for signal.  Note that the pthread_cond_wait routine
  will automatically and atomically unlock mutex while it waits. 
  Also, note that if COUNT_LIMIT is reached before this routine is run by
  the waiting thread, the loop will be skipped to prevent pthread_cond_wait
  from never returning.
  */

  pthread_mutex_lock(&count_mutex);
  if(count<COUNT_LIMIT){
    printf("watch_count(): thread %ld Count= %d. Going into wait...\n", my_id,count);
    pthread_cond_wait(&count_threshold_cv, &count_mutex);
    printf("watch_count(): thread %ld Condition signal received. Count= %d\n", my_id,count);
    printf("watch_count(): thread %ld Updating the value of count...\n", my_id);
    count += 125;
    //    count = count>>1;
    printf("watch_count(): thread %ld count now = %d.\n", my_id, count);
  }
  
  printf("watch_count(): thread %ld Unlocking mutex.\n", my_id);
  pthread_mutex_unlock(&count_mutex);
  pthread_exit(NULL);

}

void *inc_count(void * id){
  int i;
  long my_id = (long)id;
  for(i = 0; i<T_COUNT; i++){
    pthread_mutex_lock(&count_mutex);
    count++;
    // check the value of the count and signal waiting thread when condition is reached. Notice that this occurs while mutex is locked. 

    if(count == COUNT_LIMIT){
      printf("inc_count(): thread %ld, count = %d  Threshold reached. ",
             my_id, count);
      pthread_cond_signal(&count_threshold_cv);
      //      pthread_cond_broadcast(&count_threshold_cv); // we could either use this to signal more than one waiting thread.
      printf("Just sent signal.\n");
    }
    printf("inc_count(): thread %ld, count = %d, unlocking mutex\n", 
	   my_id, count);
    pthread_mutex_unlock(&count_mutex);
    /* Do some work so threads can alternate on mutex lock */
    sleep(1);
  }
  
  pthread_exit((void *)0);
}




int main(int argc, char *argv[] ){

  int i;
  long t1 = 1, t2 = 2, t3 = 3;
  pthread_t threads[3];
  pthread_attr_t attr;
  // initialize mutex and cond variables 
  pthread_mutex_init(&count_mutex, NULL);
  pthread_cond_init(&count_threshold_cv, NULL);

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&threads[0], &attr, watch_count, (void *)t1);
  pthread_create(&threads[1], &attr, inc_count, (void *)t2);
  pthread_create(&threads[2], &attr, inc_count, (void *)t3);

  // wait for all threads to complete. 
  for(i = 0; i < NUM_THREADS; i++){
    pthread_join(threads[i], NULL);
  }
  printf ("Main(): Waited on %d  threads. Done.\n", NUM_THREADS);


  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&count_mutex);
  pthread_cond_destroy(&count_threshold_cv);
  pthread_exit(NULL);
  
}

