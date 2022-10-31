cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 20)

project(DeepPP)

set(Headers
    include/activation_function.h
    include/network.h
)
set(Sources
    src/network.cpp
)

add_library(${PROJECT_NAME} ${Sources} ${Headers})

add_subdirectory(lib/eigen)
target_include_directories(${PROJECT_NAME} 
    PUBLIC include/
    PUBLIC lib/eigen
)

enable_testing()
add_subdirectory(lib/googletest)
add_subdirectory(lib/benchmark)
add_subdirectory(test)
add_subdirectory(benchmark)


