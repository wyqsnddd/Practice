# include <cstdio>
# include <iostream>
# include <boost/thread.hpp>
# include <queue>

template <class T>
class synchronizedQueue{
private:
  // Use stl queue to store data
  std::queue<T> m_queue; 
  // The mutex to synchronize on
  boost::mutex m_mutex;
  //The condition to wait for 
  boost::condition_variable m_cond;
  

public:
  // add data to the queue and notify others 
  void m_enqueue(const T & data){
    // acuqire lock on the queue 
    boost::unique_lock<boost::mutex> lock(m_mutex);
    
    // add the data to the queue 
    m_queue.push(data);
    
    // notify others that data is ready
    m_cond.notify_one();
  }// lock is automatically released

  // void m_test(){
  //   boost::unique_lock<boost::mutex> lock(m_mutex);
  //   while(m_queue.size==0)


  // }

  T m_dequeue(){

    // aquire lock on the queue 
    boost::unique_lock<boost::mutex> lock(m_mutex);
    
    //
    while(m_queue.size()==0)
      m_cond.wait(lock);

    // retrieve the data from the queue 
    T result = m_queue.front();
    m_queue.pop();

    return result;

  }

};

class producer{
private: 
  int m_id;// the id of the producer
  synchronizedQueue<std::string>* m_queue; // the queue to use 

public:

  // constructor with id and the queue to sue 

  producer(int id, synchronizedQueue<std::string>* queue){
    m_id = id;
    m_queue = queue;
  }
  

  // the thread function fills the queue with data 
  void operator () (){
    int data = 0;
    while(true){
      // produce a string and store in the queue
      std::stringstream s_temp_1, s_temp_2;
      s_temp_1<<m_id;
      s_temp_2<<++data;
      std::string str = "Producer: " + s_temp_1.str() +
	" data: " + s_temp_2.str();
      
      m_queue->m_enqueue(str);
      std::cout<<str<<std::endl;

      // sleep one second and sleep() is one of the predefined interruption points	    
      boost::this_thread::sleep(boost::posix_time::seconds(1));
    }
  }
  
};


class consumer{
private: 
  // id of the consumer
  int m_id; 
  synchronizedQueue<std::string>* m_queue; // the queue to use 

public:

  consumer(int id, synchronizedQueue<std::string>* queue){
    m_id = id;
    m_queue = queue;
  }

  // The thread function
  void operator () (){
    while(true){
      std::string str;

      std::cout<<"consumer "<<m_id
      	       <<" consumed: "<<m_queue->m_dequeue().c_str()
      	       <<std::endl;
      
      // boost::this_thread::interruption_point(); 
      // If we have sleep then it includes the interruption_point
      boost::this_thread::sleep(boost::posix_time::seconds(0.5));
    }
  }


};


int main(){
  // display the number of processors/cores
  std::cout<<boost::thread::hardware_concurrency()
	   <<" processors/cores detected."<<std::endl<<std::endl;
  std::cout<<"when threads are running, press enter to stop"<<std::endl;

  // The number of producers/consumers
  int n_producers, n_consumers;
  
  // the shared queue 
  synchronizedQueue<std::string> queue;


  // Ask the number of producers
  std::cout<<"How many producers do you want? : ";
  std::cin>>n_producers;
 
  // Ask the number of consumers
  std::cout<<"How many consumers do you want? : ";
  std::cin>>n_consumers;

  boost::thread_group producer_threads;
  for(int i = 0; i< n_producers; i++){
    producer p(i, &queue);
    producer_threads.create_thread(p);
    //    producer_threads.add_thread(p); // Only if p is a pointer to a thread !!!
  }
  
  boost::thread_group consumer_threads;
  for(int i = 0; i < n_consumers; i++){
    consumer c(i, &queue);
    consumer_threads.create_thread(c);
  }

  // Wait for enter (two times because the return from the
  // previous cin is still in the buffer)
  getchar(); getchar();
  
  
  
  // Interrupt the threads and stop them
  producer_threads.interrupt_all();
  producer_threads.join_all();
  
  consumer_threads.interrupt_all();
  consumer_threads.join_all();
  
  return (0);
}
