cmake_minimum_required(VERSION 2.10)
project(kalman_filter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCE_FILES kalman_filter_example.cc)
add_executable(kalman_example ${SOURCE_FILES})
