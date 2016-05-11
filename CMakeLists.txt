cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(librealsense_tracking)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARRY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (librealsense_tracking librealsense_tracking.cpp)
target_link_libraries (librealsense_tracking ${PCL_LIBRARIES})