# include <iostream>
# include <boost/thread.hpp>

void threadFunction(){
  // Never ending loop, which mimic the typical thread behavior
  while(true){
    try{
      // If remove this line, the thread will never be interrupted. 
      boost::this_thread::interruption_point();
    }
    catch(const boost::thread_interrupted&){
      // Thread interruption request received, break the loop
      std::cout<<"Thread interruption request received, break the loop "<< std::endl;
      break;
    }
  }
}

int main(){
  
  boost::thread t(&threadFunction);

  std::cout<<"wait for 2 seconds for the thread to stop"<<std::endl;

  while(t.timed_join(boost::posix_time::seconds(2)) == false){
    // interupt the thread 
    std::cout<<"Thread not stopped, interrupt it now "<<std::endl;
    t.interrupt();
    std::cout<<"Thread interrupt request sent"<<std::endl;
    std::cout<<"wait to finish for 2 seconds again"<<std::endl;
  }

  // The thread had been stopped. 
  std::cout<<"Thread stopped"<<std::endl;


}
