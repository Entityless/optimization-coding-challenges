cmake_minimum_required(VERSION VERSION 3.15 FATAL_ERROR)
project(OptQ LANGUAGES CXX)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fPIC -fopenmp -pthread")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -fopenmp -pthread")
endif()

add_subdirectory(tests)

