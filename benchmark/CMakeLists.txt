cmake_minimum_required(VERSION 3.22.1)

project(DeepPPBenchmark)

set(Sources 
	network_benchmark.cpp
	activation_function_benchmark.cpp
	cost_function_benchmark.cpp
)

add_executable(${PROJECT_NAME} ${Sources})
target_link_libraries(${PROJECT_NAME} PUBLIC
	benchmark_main
	DeepPP
)