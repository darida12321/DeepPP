cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)

project(DeepPP)

set(Headers
    include/activation_function.h
    include/layer.h
    include/network.h
    include/image.h
)
set(Sources
    src/layer.cpp 
    src/network.cpp
    src/image.cpp
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


