cmake_minimum_required(VERSION 3.21)
project(grid_detector)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(grid_detector examples/opencv_video.cpp grid-detector/detector.cpp)
target_link_libraries(grid_detector ${OpenCV_LIBS} )

