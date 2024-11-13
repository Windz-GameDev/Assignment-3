# CUDA Programming Projects

This repository contains several CUDA programming projects designed to demonstrate the performance benefits of GPU computing over traditional CPU implementations. Each project focuses on a specific computational task, showcasing the use of CUDA for parallel processing.
I completed this project while taking the COP6616 Parallel Computing class at the University of North Florida.

## Projects Overview

### 1. Answer general questions about Parallel Computing. There is no source code for this question.

### 2. Euclidean Distance Calculation

This project implements a CUDA program to compute the Euclidean distance between two N-dimensional points. The program compares the performance of serial (CPU) and parallel (GPU/CUDA) implementations, running multiple tests with different array sizes and averaging the results.

#### Key Features:

- Dynamically allocated arrays for both host and device.
- Prints the number of CUDA-enabled devices and properties of Device 0.
- Initializes host arrays with random numbers.
- Uses a CUDA kernel to compute squared differences and perform parallel reduction.
- Analyzes speedup provided by the GPU implementation over varying input sizes.

### Dependencies

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C/C++ compiler

## Running the Program

### Building the Projects

1. Clone the repository:

```bash
git clone https://github.com/Windz-GameDev/COP6616_Parallel_Computing_GPU_Programming_Asssignment
```

2. CD into the Question 2 directory:

Example:

```bash
cd Question_2
```

3.  Compile the program if you make any changes:

```bash
nvcc EuclideanDistanceSpeedup.cu -o EuclideanDistanceSpeedup
```

4. Run your compiled executable

```bash
./euclidean_distance
```

### 3. Grayscale Image Conversion

This project implements a CUDA program to convert a color image to grayscale using the Colorimetric method. The program compares the performance between GPU and CPU implementations.

#### Key Features:

- Uses PyCUDA for GPU implementation
- Implements the Colorimetric method (weighted RGB conversion)
- Compares performance between GPU and CPU implementations
- Includes original and resulting images for comparison

#### Implementation Details:

- Uses 32x32 thread blocks for efficient GPU utilization
- Implements memory coalescing for improved performance
- Handles images of arbitrary dimensions
- Converts timing measurements to milliseconds for precise comparison

#### Dependencies:

- PyCUDA
- OpenCV (cv2)
- NumPy
- CUDA Toolkit

#### Running the Program:

1. Ensure all dependencies are installed
2. Navigate to the Question 3 directory:

```bash
cd Question_3
```

3. Place your input image in the same directory and name it `input.png` (multiple image formats are supported)
4. Run the program:

```python
python GrayscaleConverter.py
```

#### Files in the Question_3 folder:

- `GrayscaleConverter.py`: Main program file
- `input.png`: Example input image
- `output_cpu.png`: CPU-processed output
- `output_gpu.png`: GPU-processed output

### 4. Matrix Multiplication

This project implements matrix multiplication using GPU/CUDA. The program is designed to handle matrices consisting of floating-point values and compares the performance of GPU and CPU implementations.

#### Key Features:

- Supports floating-point matrices.
- Runs experiments with varying matrix sizes.
- Analyzes speedup and how it is affected by matrix size.
- Explains differences between GPU and CPU implementations.

## Results and Discussion

Each project includes a detailed analysis of the performance improvements achieved through GPU acceleration.
Excel Graphs and discussions are provided to illustrate the speedup and efficiency of CUDA implementations compared to traditional CPU methods.
