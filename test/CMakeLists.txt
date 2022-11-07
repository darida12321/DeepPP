cmake_minimum_required(VERSION 3.22.1)

project(DeepPPTest)

set(Sources
    test_forwardprop.cpp
    test_backprop.cpp
    test_activation_function.cpp
    test_mnist.cpp
)

add_executable(${PROJECT_NAME} ${Sources})
target_link_libraries(${PROJECT_NAME} PUBLIC
    gtest_main
    DeepPP
)

add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${PROJECT_NAME}
)
