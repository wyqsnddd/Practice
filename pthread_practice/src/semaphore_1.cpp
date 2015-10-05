# include <pthread.h>
# include <semaphore.h>
# include <stdio.h>
# include <stdlib.h>

#define NITER 1000000

int count = 0;
sem_t sem_1;

void * threadAdd(void * a){
  
  int i, tmp;
  for(i=0; i<NITER; i++){
    sem_wait(&sem_1);
    tmp = count;
    tmp = tmp+1;
    count = tmp; // store the local to the global 
    sem_post(&sem_1);
  }

}

int main(int argc, char* argv[]){
  pthread_t t_1, t_2, t_3;
  
  if(sem_init(&sem_1, 0, 1)){
    printf("\n ERROR creating semaphore sem_1" );
  }
  
  if(pthread_create(&t_1, NULL, threadAdd, NULL)){
    printf("\n ERROR creating thread t_1" );
  }
  if(pthread_create(&t_2, NULL, threadAdd, NULL)){
    printf("\n ERROR creating thread t_2" );
  }
  if(pthread_create(&t_3, NULL, threadAdd, NULL)){
    printf("\n ERROR creating thread t_3" );
  }



  if(pthread_join(t_1, NULL)){
    printf("\n ERROR joining thread t_1" );
  }
  if(pthread_join(t_2, NULL)){
    printf("\n ERROR joining thread t_2" );
  }
  if(pthread_join(t_3, NULL)){
    printf("\n ERROR joining thread t_3" );
  }

  if (count < 3 * NITER) 
    printf("\n BOOM! count is [%d], should be %d\n", count, 2*NITER);
  else
    printf("\n OK! count is [%d]\n", count);


  pthread_exit(NULL);
  sem_destroy(&sem_1);
  
  return 0;
}
