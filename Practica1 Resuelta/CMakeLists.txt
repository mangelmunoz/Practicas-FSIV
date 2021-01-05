CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(Realzar_primer_plano_)
FIND_PACKAGE(OpenCV 	REQUIRED )
LINK_LIBRARIES(${OpenCV_LIBS})
include_directories ("${OpenCV_INCLUDE_DIRS}")

add_executable(Realzar_primer_plano Ej1.cpp)

#add_executable(pract1 pract1.cc)
#add_executable(pract2 pract2.cc)
#...

 
