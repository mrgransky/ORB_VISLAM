cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(vo)

IF(NOT CMAKE_BUILD_TYPE)
	#SET(CMAKE_BUILD_TYPE Debug)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("-- BUILD TYPE :" ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
find_package(OpenGL 2.0)
find_package(GLUT REQUIRED)
find_package(PCL 1.2 REQUIRED)

include_directories( include
						${PCL_INCLUDE_DIRS}
						${OPENGL_INCLUDE_DIRS} 
						${GLUT_INCLUDE_DIRS} 
						${Pangolin_INCLUDE_DIRS}
						${EIGEN3_INCLUDE_DIR})
set(LIBS 
${OpenCV_LIBS} 
${PCL_COMMON_LIBRARIES}
${PCL_IO_LIBRARIES}
${PCL_LIBRARIES}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OPENGL_LIBRARIES}
${GLUT_LIBRARY}
-lglut 
-lGLU 
-lGL
)

add_executable(${PROJECT_NAME}_kitti 
				src/run_kitti.cpp
				src/System.cpp
				src/AbsolutePose.cpp
				src/Vision.cpp
				src/Visualizer.cpp
				)
target_link_libraries(${PROJECT_NAME}_kitti ${LIBS})
