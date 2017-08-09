cmake_minimum_required(VERSION 2.4.6)
if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
  add_definitions("-g -fPIC  -Wall ")
endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)


#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



link_directories(/usr/lib/)
include_directories(/usr/include/)

# -------------------------------------------------------------------------
# boost method 1 
# FIND_PACKAGE(Boost COMPONENTS timer thread date_time program_options filesystem system REQUIRED)
FIND_PACKAGE(Boost COMPONENTS  thread date_time program_options filesystem system REQUIRED)
set(boost_libs_me 
  ${Boost_THREAD_LIBRARY}
  ${Boost_SYSTEM_LIBRARY} 
  ${Boost_FILESYSTEM_LIBRARY} 
  ${Boost_PROGRAM_OPTIONS_LIBRARY} 
  ${Boost_DATE_TIME_LIBRARY}
  )

# boost method 2 
# set(boost_libs_me
 #    /usr/local/lib/libboost_date_time.so
 #    /usr/local/lib/libboost_date_time.a
 #    /usr/local/lib/libboost_timer.so
 #    /usr/local/lib/libboost_timer.a
 #    /usr/local/lib/libboost_thread.so
 #    /usr/local/lib/libboost_thread.a
 #    /usr/local/lib/libboost_system.so
 #    /usr/local/lib/libboost_system.a
 #    /usr/local/lib/libboost_iostreams.so
 #    # /usr/lib/libboost_filesystem.so
 #    # /usr/lib/libboost_python.so
 #    )
# -------------------------------------------------------------------------

# pthread which is crucial. 
set(corelib pthread)

set (programs   
  boostDeadlineTimer 
  asynTimer
  async_arguments
  timerMemberFunction
  multithreadingTimer
  twoThreadsAsyncTimer
  )

foreach (program ${programs})
  add_executable(${program} src/${program}.cpp)
  target_link_libraries(${program}    ${boost_libs_me}) 
  target_link_libraries(${program}    ${corelib}) 
endforeach(program)


