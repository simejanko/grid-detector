cmake_minimum_required(VERSION 3.21)
project(grid_detector)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(video examples/video.cpp grid-detector/detector.cpp)
target_link_libraries(video ${OpenCV_LIBS} )

