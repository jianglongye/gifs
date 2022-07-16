cmake_minimum_required(VERSION 3.10)

# Find CGAL
find_package(CGAL REQUIRED HINTS /usr/lib/x86_64-linux-gnu/cmake/CGAL)

# set the project name
project(IntersectionDetection)

# add the executable
add_executable(intersection intersection.cpp)
target_link_libraries(intersection CGAL)
# target_link_libraries(intersection CGAL -static)
