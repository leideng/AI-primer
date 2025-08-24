#!/bin/bash

echo "Compiling CUDA Hello World program..."

nvcc -o cuda_hello_world cuda_hello_world.cpp

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running the program..."
    echo
    ./cuda_hello_world
else
    echo "Compilation failed!"
    echo "Make sure you have CUDA Toolkit installed and nvcc is in your PATH."
fi 