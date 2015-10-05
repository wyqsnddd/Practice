# CMakeLists.txt for G4CU project
project(test-cuda-thrust-gdb)

# required cmake version
cmake_minimum_required(VERSION 2.8)

# packages
find_package(CUDA)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 --ptxas-options=-v -g -G)

# cuda_compile(HELPER_O helper.cu)
# cuda_compile(DRIVER_O driver.cu OPTIONS -G)

# cuda_add_executable(driver ${HELPER_O} ${DRIVER_O})

cuda_compile(move_array_cu move_array.cu )
cuda_add_executable(move_array ${move_array_cu})

cuda_compile(printDeviceProperties_cu printDeviceProperties.cu )
cuda_add_executable(printDeviceProperties ${printDeviceProperties_cu})

cuda_compile(increment_array_cu incrementArray.cu )
cuda_add_executable(increment_array ${increment_array_cu})





cuda_compile(reverse_array_multiblock_cu reverse_array_multiblock.cu )
cuda_add_executable(reverse_array_multiblock ${reverse_array_multiblock_cu})

cuda_compile(reverse_array_multiblock_fast_cu reverse_array_multiblock_fast.cu )
cuda_add_executable(reverse_array_multiblock_fast ${reverse_array_multiblock_fast_cu})

add_executable(variables_bytes_check variables_bytes_check.cpp)