cmake_minimum_required(VERSION 3.20.0)
project(universe_cuda_best_practice VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)

add_subdirectory(01_parallel_reduction)
