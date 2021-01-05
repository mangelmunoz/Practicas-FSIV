CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(usm_enhance)
ENABLE_LANGUAGE(CXX)

FIND_PACKAGE(OpenCV 	REQUIRED )
LINK_LIBRARIES(${OpenCV_LIBS})
include_directories ("${OpenCV_INCLUDE_DIRS}")

add_executable(usm_enhance usm_enhance.cc)

 
