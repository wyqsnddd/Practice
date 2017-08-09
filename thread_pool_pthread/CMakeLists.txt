cmake_minimum_required(VERSION 2.4.6)
if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
  #add_definitions("-g -fPIC  -Wall  -O4  ")
  add_definitions("-g -fPIC  -Wall  -O4 -std=gnu++0x  ")
endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)


#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH bin/)
#set the default path for built libraries to the "lib" directory
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib/)
set(LIBRARY_OUTPUT_PATH lib/)


find_package (Threads)
include_directories(${PROJECT_SOURCE_DIR}/include/)
# message("PROJECT_SOURCE_DIR}/include/")
# message(${PROJECT_SOURCE_DIR}/include/)
set(corelib )


add_library(threadPool SHARED src/threadPool.cpp)
target_link_libraries(threadPool ${CMAKE_THREAD_LIBS_INIT})  
#target_link_libraries(${program} pthread)  
#target_link_libraries(${program} ${corelib} )


set (programs   
  main
  )
# message("${LIBRARY_OUTPUT_PATH}libthreadPool.a")
# message(${LIBRARY_OUTPUT_PATH}libthreadPool.a)

foreach (program ${programs})
  add_executable(${program} src/${program}.cpp)
#  target_link_libraries(${program} ${corelib} )
#  target_link_libraries(${program} ${CMAKE_THREAD_LIBS_INIT})  
#  target_link_libraries(${program} pthread)  
  target_link_libraries(${program} threadPool)  
endforeach(program)


