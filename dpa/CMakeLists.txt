cmake_minimum_required(VERSION 3.22)
project(dpa LANGUAGES CXX)

add_executable(program src/main.cpp)

target_compile_features(program PRIVATE cxx_std_20)
target_compile_options(program PRIVATE -Wall -Wextra -pedantic)

find_package(Eigen3 REQUIRED)
target_link_libraries(program PRIVATE Eigen3::Eigen)
