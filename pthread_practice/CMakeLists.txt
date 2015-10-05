cmake_minimum_required(VERSION 2.4.6)
if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
  add_definitions("-g -fPIC  -Wall ")
endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)


#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH bin/)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


find_package (Threads)
include_directories({PROJECT_SOURCE_DIR}/include)

set(corelib pthread)

set (programs   pthread_creation_termination  
                pthread_struct_pass
		pthread_join
		pthread_stack_size
		pthread_mutex
		pthread_condition
		semaphore_1
		semaphore_producer_consumer
  )

foreach (program ${programs})
  add_executable(${program} src/${program}.cpp)
  #target_link_libraries(${program} ${corelib} )
  target_link_libraries(${program} ${CMAKE_THREAD_LIBS_INIT})  
endforeach(program)


