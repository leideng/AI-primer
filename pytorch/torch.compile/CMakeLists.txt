cmake_minimum_required(VERSION 3.10)
project(torch_compile_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(torch_compile_cpp_demo torch_compile_cpp_demo.cpp)

# Set compiler flags
if(MSVC)
    target_compile_options(torch_compile_cpp_demo PRIVATE /W4)
else()
    target_compile_options(torch_compile_cpp_demo PRIVATE -Wall -Wextra -O2)
endif()

# Optional: Enable debug information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(torch_compile_cpp_demo PRIVATE -g)
endif()

# Print configuration
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}") 