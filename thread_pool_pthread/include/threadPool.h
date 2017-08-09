# include <pthread.h>
# include <semaphore.h>
# include <iostream>

// This class needs to be subclassed by the user
class workerThread{
 public:
  unsigned virtual executeThis(){
    return 0;
  }
  
};

// ThreadPool class manages all the threadPool related activities. This includes keeping track of idel threads and synchronizations between all threads. 

class threadPool{
 private:
  int m_maxThreads;
  pthread_mutex_t m_mutexSync; 

  pthread_mutex_t m_mutexWorkCompletion;
  int m_incompleteWork;
  
  pthread_cond_t m_condCrit;
  sem_t m_availableWork;
  sem_t m_availableThreads;
  
  workerThread **m_workerQueue;
  
  int m_topIndex;
  int m_bottomIndex;
  
  int m_queueSize;
  
 public: 
  threadPool();
  threadPool(int maxThreadsTemp);
  
  virtual ~threadPool();
  
  void destroyPool(int maxPollSecs);
  
  bool assignWork(workerThread *worker);
  bool fetchWork(workerThread **worker);
    
  void initializeThreads();
  
  static void *threadExecute(void *param);
};
