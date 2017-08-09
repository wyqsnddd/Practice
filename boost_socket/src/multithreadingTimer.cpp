# include <iostream>
# include <boost/asio.hpp>
# include <boost/bind.hpp>
# include <boost/thread.hpp>
# include <boost/date_time/posix_time/posix_time.hpp>

// Motivation: 
//  asio library provides a guarantee that callback handlers will only be called from threads that are currently calling io_service::run(). Consequently, calling io_service::run() from only one thread ensures that callback handlers cannot run concurrently.

// Goal: concurrent running handlers

// Approach:  a pool of threads calling io_service::run().
// Notice: as this approach allows handlers to run concurrently, we need a method of synchronisation when handlers might be accessing a shared, thread-unsafe resource.
// By wrapping the handlers using the same boost::asio::strand, we are ensuring that they cannot execute concurrently.
// Notice that using strand is slightly better than a single thread.  

class printer{
public: 

  printer(boost::asio::io_service& io)
    :counter_(0), 
     timer1_(io, boost::posix_time::seconds(1)),
     timer2_(io, boost::posix_time::seconds(1)),
     strand_(io)
  {

    // Key:  When initiating the asynchronous operations, each callback handler is "wrapped" using the boost::asio::strand object.
    timer1_.async_wait(
		       strand_.wrap(
				    boost::bind(&printer::print1,this)
				    )
		       );

    timer2_.async_wait(
		       strand_.wrap(
				    boost::bind(&printer::print2,this)
				    )
		       );
    
  }

  ~printer(){
    std::cout << "Final count is " << counter_ <<std::endl;
  }

  void print1(){
    if (counter_ < 5)
      {
	std::cout << "timer 1: "<< counter_ << std::endl;
	++counter_;

	timer1_.expires_at(timer1_.expires_at() + boost::posix_time::seconds(1));
	timer1_.async_wait(
			  boost::bind(&printer::print1,  // boost::asio::placeholders::error is NOT needed anymore. 
				      this) // Since all non-static class member functions have an implicit this parameter, we need to bind this to the function
			  );
      }

  }
  void print2(){
    if (counter_ < 5)
      {
	std::cout << "timer 2: "<<  counter_ << std::endl;
	++counter_;

	timer2_.expires_at(timer2_.expires_at() + boost::posix_time::seconds(1));
	timer2_.async_wait(
			  boost::bind(&printer::print2,  // boost::asio::placeholders::error is NOT needed anymore. 
				      this) // Since all non-static class member functions have an implicit this parameter, we need to bind this to the function
			  );
      }

  }


private:
  int counter_; // this is the shared resources in this case
  boost::asio::deadline_timer timer1_;
  boost::asio::deadline_timer timer2_;
  boost::asio::strand strand_;// An boost::asio::strand guarantees that, for those handlers that are dispatched through it, an executing handler will be allowed to complete before the next one is started. This is guaranteed irrespective of the number of threads that are calling io_service::run().
};


int main(){

  boost::asio::io_service io;

  printer p(io);
  boost::thread t (boost::bind(
			       &boost::asio::io_service::run, // function address  
			       &io  // pointer to the object 
			       ));
  io.run();// to be called from two threads: the main thread and one additional thread 't'.
  t.join();

  return 0;
}
