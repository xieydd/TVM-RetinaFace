cmake_minimum_required(VERSION 3.5)

SET(OpenCV_DIR /usr/local/lib/)
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
add_executable(tvm_retinaface  main.cpp)
target_link_libraries(tvm_retinaface retinaface ${OpenCV_LIBS})
