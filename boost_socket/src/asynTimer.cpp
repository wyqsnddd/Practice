# include <iostream>
# include <boost/asio.hpp>
// # include <boost/date_time/posix_time/posix_time.hpp>


// Using asio's asynchronous functionality means having a callback function that will be called when an asynchronous operation completes. 
void print (const boost::system::error_code &ec){
  std::cout<<"From callback function: Hello world!"<<std::endl;
}

int main(){

  boost::asio::io_service io; 

  boost::asio::deadline_timer t_1(io, boost::posix_time::seconds(5));
  t_1.async_wait(print);


  boost::asio::deadline_timer t_2(io, boost::posix_time::seconds(10));
  t_2.async_wait(print);

  std::cout<<"From main body before control handed over to io: Hello world!"<<std::endl;
  
  //  1. asio library provides a guarantee that callback handlers will only be called from threads that are currently calling io_service::run(). 
  // 2. The advantage of async_wait() is that the function call returns immediately instead of waiting five seconds.
  // 3. async_wait() function expects a handler function (or function object) with the signature void(const boost::system::error_code&)
  io.run();
  // 1. The io_service::run() function will also continue to run while there is still "work" to do.
  // 2. run() actually blocks the current thread only. Execution therefore stops at the call of run().
  std::cout<<"From main body after control handed over to io: Hello world!"<<std::endl;

  return 0;
}


