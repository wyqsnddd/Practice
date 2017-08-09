# include <iostream>
# include <boost/asio.hpp>
# include <boost/bind.hpp>

// Goal: a repeating timer. 
// Approach: modified within the handler function. specified by arguments to the handler function 
// Notice: no explicit call to ask the io_service to stop.

void print(const boost::system::error_code&, 
	   boost::asio::deadline_timer * timer,  // change the timer waiting time.  
	   int* counter //  uses a counter to stop running when the timer fires for the sixth time.
	   ) {

  if(*counter < 5){
    std::cout<<*counter<<std::endl;
    (*counter)++;
    timer->expires_at(timer->expires_at() + boost::posix_time::seconds(1));
    timer->async_wait(
		      boost::bind(print, 
				  boost::asio::placeholders::error,
				  timer,
				  counter)
		      
		      );
  }  

}

int main(){

  boost::asio::io_service io; 

  int count = 0;

  boost::asio::deadline_timer sampleTimer(io, boost::posix_time::seconds(3));

  sampleTimer.async_wait(
			 boost::bind(print, 
				     boost::asio::placeholders::error,// named placeholder to match the parameter list.
				     &sampleTimer, 
				     &count
				     )
			 );
  io.run();
  
  std::cout << "Final count is " << count << "\n";


 return 0;
 
}
