cmake_minimum_required(VERSION 2.4.6)
if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
  add_definitions("-g -fPIC  -Wall -std=gnu++11")
endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)


#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH bin/)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



#FIND_PACKAGE(Boost 1.50.0 COMPONENTS thread date_time program_options filesystem system REQUIRED)
# set(BOOST_LIBS_ME ${Boost_THREAD_LIBRARY}
#   ${Boost_SYSTEM_LIBRARY} 
#   ${Boost_FILESYSTEM_LIBRARY} 
#   ${Boost_PROGRAM_OPTIONS_LIBRARY} 
#   ${Boost_DATE_TIME_LIBRARY})

include_directories(/usr/local/include/)

set(BOOST_LIBS_ME
  # /usr/local/lib/libboost_date_time.so
  # /usr/local/lib/libboost_thread.so
  # /usr/local/lib/libboost_system.so
  )

set(corelib pthread)

set (programs   
  example_one
  )

foreach (program ${programs})
  add_executable(${program} src/${program}.cpp)
  target_link_libraries(${program}    ${BOOST_LIBS_ME}) 
endforeach(program)


