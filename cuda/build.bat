@echo off
echo Compiling CUDA Hello World program...

nvcc -o cuda_hello_world.exe cuda_hello_world.cpp

if %ERRORLEVEL% EQU 0 (
    echo Compilation successful!
    echo Running the program...
    echo.
    cuda_hello_world.exe
) else (
    echo Compilation failed!
    echo Make sure you have CUDA Toolkit installed and nvcc is in your PATH.
)

pause 