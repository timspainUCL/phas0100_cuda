cmake_minimum_required(VERSION 3.10) # Guaranteed to not have to use find_package(CUDA) beyond version 3.10

project(cuda_memory LANGUAGES CXX CUDA)

add_executable(memory memory.cu)
