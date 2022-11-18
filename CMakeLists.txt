cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)

project(DeepPP)

add_compile_options(
    -Wall -Ofast
)

set(Headers
    include/runtime/cost_function.h
    include/runtime/activation_function.h
    include/runtime/network.h
    include/runtime/mnist_imageset.h

    include/templates/network.h
    include/templates/activation_function.h
    include/templates/cost_function.h
    include/templates/template_helpers.h
    include/templates/mnist_imageset.h
)
set(Sources
    src/cost_function.cpp
    src/activation_function.cpp
    src/network.cpp
    src/mnist_imageset.cpp
)
add_library(${PROJECT_NAME} ${Sources} ${Headers})
add_executable(${PROJECT_NAME}_Demo src/demo_mnist.cpp)

add_subdirectory(lib/eigen)
target_include_directories(${PROJECT_NAME} 
    PUBLIC include/
    PUBLIC lib/eigen
)
target_include_directories(${PROJECT_NAME}_Demo 
    PUBLIC include/
    PUBLIC lib/eigen
)

enable_testing()
add_subdirectory(lib/googletest)
add_subdirectory(lib/benchmark)
add_subdirectory(test)
add_subdirectory(benchmark)


