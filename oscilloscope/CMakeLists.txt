cmake_minimum_required(VERSION 3.14)
project(oscilloscope LANGUAGES CXX)

add_library(Dwf INTERFACE)
target_include_directories(Dwf INTERFACE "/usr/digilent")
target_link_libraries(Dwf INTERFACE dwf)

add_executable(program src/main.cpp)
target_link_libraries(program PRIVATE Dwf)

find_package(Eigen3 REQUIRED)
target_link_libraries(program PRIVATE Eigen3::Eigen)

target_compile_features(program PRIVATE cxx_std_20)
target_compile_options(program PRIVATE -Wall -Wextra -pedantic)
