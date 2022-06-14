cmake_minimum_required(VERSION 2.10)
project(kalman_filter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(Eigen3)
IF (${Eigen3_FOUND})
   MESSAGE(STATUS "Use Eigen.")
   set(SOURCE_FILES kalman_filter_example.cc)
   include_directories(${EIGEN3_INCLUDE_DIR})
ELSE (${Eigen3_FOUND})
   MESSAGE(STATUS "Use Matrix.")
   set(SOURCE_FILES kalman_filter_no_eigen_example.cc)
ENDIF (${Eigen3_FOUND})

add_executable(kalman_example ${SOURCE_FILES})
