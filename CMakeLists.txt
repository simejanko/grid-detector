cmake_minimum_required(VERSION 3.21)
project(square_grid_measurer)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(square_grid_measurer main.cpp grid-detector/detector.cpp)
target_link_libraries(square_grid_measurer ${OpenCV_LIBS} )

