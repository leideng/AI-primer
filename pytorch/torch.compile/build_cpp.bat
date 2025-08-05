@echo off
REM Build script for torch.compile C++ demonstration (Windows)

echo Building torch.compile C++ demonstration...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release

REM Build the project
echo Building project...
cmake --build . --config Release

REM Check if build was successful
if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo You can now run: torch_compile_cpp_demo.exe
) else (
    echo Build failed!
    exit /b 1
) 