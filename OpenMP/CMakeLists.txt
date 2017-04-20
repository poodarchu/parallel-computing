cmake_minimum_required(VERSION 3.7)
project(OpenMP_Lab1)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)
set(CMAKE_C_COMPILER /usr/local/bin/gcc-6)
# enable openmp using -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(OpenMP_Lab1 ${SOURCE_FILES})