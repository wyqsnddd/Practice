#include <iostream>
class test_sample{
 public:
  test_sample(){
    a = 1;
    b = 2;
    c = 3;
  }
  ~test_sample(){
  }

  void print_a() {
    std::cout<<"a is: "<<a<<std::endl;
  }
  void print_b() {
    std::cout<<"b is: "<<b<<std::endl;
  }
  void print_c() {
    std::cout<<"c is: "<<c<<std::endl;
  }
  
 private:
  int a;      // Items will point to the dynamically allocated array
  int b;      // number of items currently in the list
  int c;      // the current size of the array
  
};


class SampleWorkerThread_1 : public workerThread
{
private: 
  test_sample * m_input;
  
public:
  SampleWorkerThread_1(test_sample * input){
    m_input = input;
  }
  ~SampleWorkerThread_1(){
  }
  unsigned virtual executeThis()
  {
    m_input->print_a();

    // Instead of sleep() we could do anytime consuming work here.
    //Using ThreadPools is advantageous only when the work to be done is really time consuming. (atleast 1 or 2 seconds)
    sleep(0.01);
 
    return(0);
  }
};
class SampleWorkerThread_2 : public workerThread
{
private: 
  test_sample * m_input;
  
public:
  SampleWorkerThread_2(test_sample * input){
    m_input = input;
  }
  ~SampleWorkerThread_2(){
  }

  unsigned virtual executeThis()
  {

    m_input->print_b();

    sleep(0.01);
 
    return(0);
  }
};
class SampleWorkerThread_3 : public workerThread
{
private: 
  test_sample * m_input;
  
public:
  SampleWorkerThread_3(test_sample * input){
    m_input = input;
  }
  ~SampleWorkerThread_3(){
  }

  unsigned virtual executeThis()
  {
    m_input->print_c();
    
    sleep(0.01);
 
    return(0);
  }
};

