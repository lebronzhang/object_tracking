cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(librealsense_tracking)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${INCLUDE_PATH})
link_directories(${PCL_LIBRARRY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (librealsense_tracking librealsense_tracking.cpp define.cpp Sampling.cpp Ppfe.cpp Icp.cpp)
target_link_libraries (librealsense_tracking ${PCL_LIBRARIES})