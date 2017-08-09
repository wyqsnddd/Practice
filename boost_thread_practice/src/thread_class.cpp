# include <cstdio>
# include <iostream>
# include <boost/thread.hpp>

class animationClass{
private: 
  // The thread runs this object 
  boost::thread * m_thread; 
  int m_frame; // the current frame number 


  // variable that indicates to stop and the mutex to synchronize "must stop  "  on 
  bool m_mustStop;
  boost::mutex m_mustStopMutex;


public:

  // Default constructor 

  animationClass(){
    m_thread=NULL;
    m_mustStop = false;
    m_frame = 0;
  }



  // Destructor
  ~animationClass(){
    if(m_thread!=NULL)
      delete m_thread;
  }
  


  // start the thread 
  void start(){
    m_thread = new boost::thread(boost::ref(*this));
    // (1) Create the thread and start it with itself as argument. 
    // (2) need to use boost::ref(*this), since we don't want to let the object to copy the object itself
  }

  void stop(){
    // signal the thread to stop
    m_mustStopMutex.lock();
    m_mustStop = true;
    m_mustStopMutex.unlock();

    // wait for the thread to finish
    if(m_thread!=NULL)
      m_thread->join();
  }
  
  // dispaly next frame of the animation
  void displayNextFrame(){
    // simulate next frame
    std::cout<<"Press <RETURN> to stop. Frame: "<<m_frame++<<std::endl;
  }
  
  // Thread function 
  void operator()(){
    bool mustStop;
    
    do{
      displayNextFrame();
      // sleep for 40ms == 25 frames/sec
      boost::this_thread::sleep(boost::posix_time::millisec(40));
      m_mustStopMutex.lock();
      mustStop=m_mustStop;
      m_mustStopMutex.unlock();

    }while(!mustStop);
      }
};



int main(){
  animationClass ac_test;

  ac_test.start();
  // wait for the user to press return 
  getchar();
  
  // stop the animation class 
  std::cout << "Animation stopping..." << std::endl;
  ac_test.stop();
  std::cout << "Animation stopped." << std::endl;

  return 0;
}

