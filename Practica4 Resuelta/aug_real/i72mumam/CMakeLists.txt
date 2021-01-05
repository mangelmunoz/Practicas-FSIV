CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(aug_real)
ENABLE_LANGUAGE(CXX)

FIND_PACKAGE(OpenCV 	REQUIRED )
LINK_LIBRARIES(${OpenCV_LIBS})
include_directories ("${OpenCV_INCLUDE_DIRS}")

add_executable(aug_real aug_real.cpp common.cpp common.hpp)
add_executable(calibrate_camera calibrate_camera.cpp common.cpp common.hpp)

 
