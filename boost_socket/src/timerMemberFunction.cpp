# include <iostream>
# include <boost/asio.hpp>
# include <boost/bind.hpp>
# include <boost/date_time/posix_time/posix_time.hpp>

// Goal:  use timer as a member and async wait on a member function. 

// Approach: code timer as an object member  
// Notice:  const boost::system::error_code& is a must for a free handler function NOT a member function.  
// boost::asio::placeholders::error placeholder is not specified here, as the print member function does not accept an error object as a parameter.
class printer{
  
public: 

  printer(boost::asio::io_service& io)
    :timer_(io, boost::posix_time::seconds(1)),
    counter_(0)
  {
    // start the timer at construction
    timer_.async_wait(boost::bind(&printer::print,
				  this)
		      );


  }

  ~printer(){
    std::cout<<"Final count is: "<< counter_ << std::endl;
  }
  void print(){
    if (counter_ < 5)
      {
	std::cout << counter_ << std::endl;
	++counter_;

	timer_.expires_at(timer_.expires_at() + boost::posix_time::seconds(1));
	timer_.async_wait(
			  boost::bind(&printer::print,  // boost::asio::placeholders::error is NOT needed anymore. 
				      this) // Since all non-static class member functions have an implicit this parameter, we need to bind this to the function
			  );
      }
    
  }
private: 
  boost::asio::deadline_timer timer_;
  int counter_;
};


int main(){

  boost::asio::io_service io;

  printer p(io);

  io.run();
  
  return 0;
}
