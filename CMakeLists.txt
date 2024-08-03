cmake_minimum_required(VERSION 3.0.2)
project(nlopt_sample)

find_package(catkin REQUIRED COMPONENTS)
find_package(NLopt REQUIRED)

catkin_package(
)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${NLOPT_INCLUDE_DIRS}
)

add_executable(sample1 src/sample1.cpp)
target_link_libraries(sample1 ${NLOPT_LIBRARIES})

