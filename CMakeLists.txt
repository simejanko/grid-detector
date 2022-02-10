cmake_minimum_required(VERSION 3.21)
project(opencv_toy_project)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(opencv_toy_project main.cpp square-measurer/detector.cpp square-measurer/measurer.cpp)
target_link_libraries( opencv_toy_project ${OpenCV_LIBS} )

