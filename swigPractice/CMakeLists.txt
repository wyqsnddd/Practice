project(swig_test)

# required cmake version
cmake_minimum_required(VERSION 2.8)

# packages
find_package(SWIG REQUIRED)
include(cmake/UseSWIG.cmake)
# The SWIG package defines the following macros: SWIG_ADD_MODULE and SWIG_LINK_LIBRARIES.
message("SWIG_FOUND:")
message(${SWIG_FOUND})

find_package(PythonLibs)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_SWIG_FLAGS "")

# set_source_files_properties(example.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(example.i PROPERTIES SWIG_FLAGS "-includeall")
swig_add_module(example python example.i example.c)
swig_link_libraries(example ${PYTHON_LIBRARIES})


# add_executable(example example)