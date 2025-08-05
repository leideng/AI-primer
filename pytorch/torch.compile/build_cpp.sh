#!/bin/bash

# Build script for torch.compile C++ demonstration

echo "Building torch.compile C++ demonstration..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can now run: ./torch_compile_cpp_demo"
else
    echo "Build failed!"
    exit 1
fi 