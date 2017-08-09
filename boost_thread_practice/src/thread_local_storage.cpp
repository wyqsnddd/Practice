# include <iostream>
# include <boost/thread.hpp>

// global data, each thread has its own value
boost::thread_specific_ptr<int> threadLocalData;


// callable function 
void callableFunction(int id){
  // initilize thread local  data (for the current thread)
  threadLocalData.reset(new int);
  *threadLocalData = 0; 

  // Do this a number of times 

  for (int i = 0; i<5; i++){
    // print value of global data and increase value 
    std::cout<<"Thread: "<<id<<" - Value: "<<(*threadLocalData)++<<std::endl;
    
    // wait for one second
    boost::this_thread::sleep(boost::posix_time::seconds(1));
  }
  
}


int main(){
  // initialize thread local data (for the main thread)
  threadLocalData.reset(new int);
  *threadLocalData=0;

  // create threads adn add them to the thread group

  boost::thread_group threads;
  for(int i = 0; i<3; i++){
    boost::thread* t = new boost::thread(&callableFunction, i);
    threads.add_thread(t);
  }

  threads.join_all();

  // Display thread local storage value, should still be zero
  std::cout<<"Main - Value: "<<(*threadLocalData)<<std::endl;
  return 0;
}
