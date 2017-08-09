# include <iostream>
# include "threadPool.h"
# include "test_sample.h"

#define ITERATIONS 10

int main(int argc, char ** argv){
  threadPool tp;

  //ThreadPool(N);
  //Create a Threadpool with N number of threads
  threadPool* myPool = new threadPool(10);

  //We will count time elapsed after initializeThreads()
  time_t t1=time(NULL);
  test_sample test;
  //Lets start bullying ThreadPool with tonnes of work !!!
  //  for(unsigned int i=0;i<ITERATIONS;i++){
    SampleWorkerThread_1* myThread_1 = new SampleWorkerThread_1(&test);
    myPool->assignWork(myThread_1);

    SampleWorkerThread_2* myThread_2 = new SampleWorkerThread_2(&test);
    myPool->assignWork(myThread_2);
    SampleWorkerThread_3* myThread_3 = new SampleWorkerThread_3(&test);
    myPool->assignWork(myThread_3);

    //  }
    myPool->initializeThreads();
  // destroyPool(int maxPollSecs)
  // Before actually destroying the ThreadPool, this function checks if all the pending work is completed.
  // If the work is still not done, then it will check again after maxPollSecs
  // The default value for maxPollSecs is 5 seconds.
  // And ofcourse the user is supposed to adjust it for his needs.
  myPool->destroyPool(2);
         
  time_t t2=time(NULL);
  std::cout<<t2-t1<<" seconds elapsed"<<std::endl;
  delete myPool;
         
  return 0;
  
}
