# include <stdlib.h>
# include "threadPool.h"

threadPool::threadPool(){
  threadPool(2);
}


threadPool::threadPool(int maxThreads){
  if(maxThreads<1) maxThreads = 1;

  pthread_mutex_init(&m_mutexSync, NULL);
  pthread_mutex_init(&m_mutexWorkCompletion, NULL);

  pthread_mutex_lock(&m_mutexSync);
  this->m_maxThreads = maxThreads;
  this->m_queueSize = maxThreads;
  m_workerQueue = new workerThread*[maxThreads];

  m_topIndex = 0;
  m_bottomIndex = 0;
  m_incompleteWork = 0;
  
  sem_init(&m_availableWork, 0, 0);
  sem_init(&m_availableThreads, 0, m_queueSize);

  pthread_mutex_unlock(&m_mutexSync);

}

threadPool::~threadPool(){
}

void threadPool::destroyPool(int maxPollSecs = 3){
  while(m_incompleteWork > 0){
    std::cout<<"Work is still incomplete= "<<m_incompleteWork<<std::endl;
    sleep(maxPollSecs);
  }

  std::cout<<"All done. Ready to rest"<<std::endl;
  sem_destroy(&m_availableWork);
  sem_destroy(&m_availableThreads);
}

bool threadPool::assignWork(workerThread *workerThread){
  
  pthread_mutex_lock(&m_mutexWorkCompletion);
  m_incompleteWork++;
  pthread_mutex_unlock(&m_mutexWorkCompletion);  
  
  sem_wait(&m_availableThreads);
  pthread_mutex_lock(&m_mutexSync);
  m_workerQueue[m_topIndex] = workerThread;
  if(m_queueSize!=1){
    m_topIndex = (++m_topIndex)%(m_queueSize - 1);
  }
  pthread_mutex_unlock(&m_mutexSync);
  sem_post(&m_availableWork);
  int test;
  sem_getvalue(&m_availableThreads, &test);
  std::cout<< "m_availableThreads is:  "<<test <<std::endl;
 
  std::cout<< "One piece of work is assigned to m_workerQueue["<<m_topIndex - 1 <<"]"<<std::endl;
  return true;
}

bool threadPool::fetchWork(workerThread ** workerArg){
  //  std::cout<< " One thread is fetching work "<<std::endl;
    
  // Watch this duality m_availableWork and m_availableThreads
  sem_wait(&m_availableWork); 
  //  std::cout<< " Something is available  "<<std::endl;
  
  pthread_mutex_lock(&m_mutexSync);
  //  std::cout<< " m_bottomIndex is: "<<m_bottomIndex<<std::endl;

  workerThread * workerThread_temp = m_workerQueue[m_bottomIndex];

    
  *workerArg = workerThread_temp;
  if(m_queueSize!=1){// m_queueSize >= 1 and fixed after initialization.
    // ++ ??? 
    m_bottomIndex = (++m_bottomIndex)%(m_queueSize - 1);
  }
  //  std::cout<< " m_bottomIndex is: "<<m_bottomIndex<<std::endl;

  pthread_mutex_unlock(&m_mutexSync);
  sem_post(&m_availableThreads); 
  
  return true;
}



void* threadPool::threadExecute(void * param){
  //  std::cout<<"One thread is being initialied. "<<std::endl;
  
  workerThread *worker = NULL;
    while(
	  ((threadPool *) param)->fetchWork(&worker) // as long as there is a worker
	  ){
      
      if(worker){
	worker->executeThis();
	//	std::cout<<" One piece of work has been finished"<<std::endl;
	pthread_mutex_lock(
			   &(
			     ((threadPool *) param)->m_mutexWorkCompletion
			     )
			   );
	
	((threadPool *) param)->m_incompleteWork -- ;
	
	
	pthread_mutex_unlock(
			     &(
			       ((threadPool *) param)->m_mutexWorkCompletion
			       )
			     );
      } // If worker pointer exists
    }// end of while 
    return 0;
}

void threadPool::initializeThreads(){
  for (int i =0; i<m_maxThreads; i++){
    pthread_t tempThread;
    pthread_create(&tempThread, NULL, &threadPool::threadExecute, (void *)this);
  }
  std::cout<< " " << m_maxThreads<<" threads have been initialized"<<std::endl;
}
