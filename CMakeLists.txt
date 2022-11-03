cmake_minimum_required(VERSION 3.10)

#Set the project name
project(CountingIndividualObjects)
enable_testing()

#C++ 11
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED True)


#Remove to remove debug symbols.
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS}  )


#Task 1
add_executable(kmeans src/kmeans_segmentation.cpp src/histogram.cpp src/blob_detection.cpp)
target_link_libraries(kmeans ${OpenCV_LIBS})