# include <iostream>
// Provides IO functionality.
# include <boost/asio.hpp> 
// Provides timer. 
# include <boost/date_time/posix_time/posix_time.hpp>



int main(){

  boost::asio::io_service io; // first argument for timer object. 

  boost::asio::deadline_timer t(io, boost::posix_time::seconds(5));

  t.wait();

  std::cout<<"Hello, World!"<<std::endl;


  return 0;
}
