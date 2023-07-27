# project name
PROJECT(opencv_test)
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${opencv_test_SOURCE_DIR}/bin)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# name of executable file and path of source file
# add_executable(opencv_test src/main.cpp)
# target_link_libraries(opencv_test ${OpenCV_LIBS})
# add_executable(opencv_test_1 src/main1.cpp)
# target_link_libraries(opencv_test_1 ${OpenCV_LIBS})
add_executable(rs_OpenCV_test src/main_rs.cpp)
target_link_libraries(rs_OpenCV_test ${OpenCV_LIBS} realsense2)


# opencv libraries


