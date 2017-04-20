cmake_minimum_required(VERSION 3.7)
project(OpenMPI_Lab2)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_COMPILER /opt/openmpi/bin/mpicc)
set(CMAKE_CXX_COMPILER /opt/openmpi/bin/mpic++)

#include_directories({MPI_INCLUDE_PATH})

set(SOURCE_FILES integral.cpp odd_even_sort.cpp matrix_multiply.cpp)
add_executable(OpenMPI_Lab2 ${SOURCE_FILES})