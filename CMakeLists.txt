cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 14)

project(DeepPP)

set(Headers
    include/activation_function.h
    include/layer.h
    include/network.h
)
set(Sources
    src/layer.cc 
    src/network.cc
)

add_library(${PROJECT_NAME} ${Sources} ${Headers})

add_subdirectory(lib/eigen)
target_include_directories(${PROJECT_NAME} 
    PUBLIC include/
    PUBLIC lib/eigen
)

enable_testing()
add_subdirectory(lib/googletest)
add_subdirectory(test)

