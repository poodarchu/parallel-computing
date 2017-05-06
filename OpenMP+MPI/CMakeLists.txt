cmake_minimum_required(VERSION 3.7)
project(PC_Lab4)

#set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(CMAKE_CXX_COMPILER /usr/bin/mpic++)
set(CMAKE_C_COMPILER /usr/bin/mpicc)

set(SOURCE_FILES main.cpp hybrid_matrix_multiply.cpp bak/matrix_multiply.cpp bak/odd_even_trans_sort.cpp trapezoidal_integral.cpp bak/version.cpp odd_even_sort.c hybird_pi.c matrix_multipy.c)
add_executable(PC_Lab4 ${SOURCE_FILES})