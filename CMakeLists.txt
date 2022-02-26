cmake_minimum_required(VERSION 3.21)
project(grid_detector)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_libraries(${OpenCV_LIBS})

add_subdirectory(grid_detector)
link_libraries(grid_detector)

add_subdirectory(examples)
