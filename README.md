# CUDA Programming Projects

This repository contains several CUDA programming projects designed to demonstrate the performance benefits of GPU computing over traditional CPU implementations. Each project focuses on a specific computational task, showcasing the use of CUDA for parallel processing.

**Note:** I completed this project while taking the COP6616 Parallel Computing class at the University of North Florida.

## Projects Overview

### 1. Answer general questions about Parallel Computing.

There is no source code for this question. Answers are provided in **Assignment_3_Analysis.pdf** located in the root directory.

### 2. Euclidean Distance Calculation

This project implements a CUDA program to compute the Euclidean distance between two N-dimensional points. The program compares the performance of serial (CPU) and parallel (GPU/CUDA) implementations, running multiple tests with different array sizes and averaging the results.

#### Key Features:

- **Dynamic Memory Allocation:** Arrays for both host and device are dynamically allocated. Host arrays are allocated using `malloc`, and device arrays using `cudaMalloc`.
- **Device Information:** Prints the number of CUDA-enabled devices and at least three interesting properties of Device 0, including the device name, maximum threads per block, and maximum grid size.
- **Initialization:** Host arrays are initialized with random integers within a small range (0 to 99).
- **Data Transfer:** Uses `cudaMemcpy` to copy host input arrays to the device.
- **Parallel Execution:** Calls a CUDA kernel that computes the square of the difference of the components for each dimension, performs a parallel reduction to sum these values across each block, sums the partial sums of each block on the CPU, and then finally takes the square root of the sum to compute the Euclidean distance.
- **Performance Analysis:** Runs experiments using varying sizes of inputs. Graphs and results can be found in the analysis pdf.

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

- **PyCUDA Implementation:** Uses PyCUDA for the GPU implementation, making it accessible and easier to integrate with Python libraries.
- **Colorimetric Method:** Implements the standard Colorimetric method (weighted RGB conversion) for accurate grayscale conversion.
- **Performance Comparison:** Compares execution times between GPU and CPU implementations to evaluate speedup.
- **Image Handling:** Works with multiple image types as input and outputs the grayscale images for visual comparison.

#### Implementation Details:

- **Parallelism:** Each pixel conversion is independent, allowing for massive parallelism on the GPU.
- **Thread Blocks:** Utilizes 32x32 thread blocks for efficient GPU utilization if max thread per block is 1024 on your hardware.
- **Memory Coalescing:** Ensures that RGB values are accessed in a coalesced manner, improving memory access performance.
- **Timing Measurements:** Converts timing measurements to milliseconds for precise comparison between CPU and GPU execution times.

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

- **Floating-Point Matrices:** Supports matrices with floating-point values between [0, 1].
- **Variable Matrix Sizes:** Runs experiments with varying matrix sizes, from very small (2x2) to large (2048x2048), to observe how performance scales.
- **Performance Analysis:** Analyzes the speedup achieved by the GPU implementation and how it is affected by matrix size in the analysis pdf.
- **Comparative Explanation:** Explains how the GPU implementation differs from the CPU-based implementation, including detailed examples and diagrams in the analysis pdf.

#### Implementation Details:

- **CPU Implementation:**
  - **Algorithm:** Uses a standard triple nested loop algorithm for matrix multiplication.
  - **Order of Access:** Matrices are stored and accessed in row-major order.
  - **Time Complexity:** \(O(N^3)\), which becomes significant for large \(N\).
- **GPU Implementation:**
  - **Kernel Function (`gpuMatrixMultiply`):** Each thread computes one element of the result matrix.
  - **Thread Mapping:**
    - **Global Index Calculation:** Utilizes block and thread indices to compute the global row and column for each thread.
    - **Edge Case Handling:** Includes bounds checking to prevent out-of-bounds memory access.
  - **Block and Grid Dimensions:**
    - **Block Size:** Uses a 32x32 block size to optimize for GPU architecture (if you have a total of 1024 threads per block for your hardware).
    - **Grid Size:** Calculated based on matrix size and block dimensions to ensure coverage of the entire matrix.
- **Performance Results:**
  - Observes that GPU speedup increases with matrix size.
  - Notes that for small matrices, GPU may be slower due to kernel launch overhead.
  - Provides detailed a graph of speedup and analysis, included in **Assignment_3_Analysis.pdf**.

#### Running the Program:

1. Navigate to the `Question_4` directory:

```
bash cd Question_4
```

2. Compile the program if you make any changes:

```bash
nvcc MatrixMultiplication.cu -o MatrixMultiplication
```

3. Run the executable:

```bash
./MatrixMultiplication
```

4. Observe the output, which includes:

   - Matrix size.
   - Average CPU time over multiple runs.
   - Average GPU time over multiple runs.
   - Speedup factor.
   - Verification of correctness for small matrix sizes.

## Results and Discussion

Each project includes a detailed analysis of the performance improvements achieved through GPU acceleration. Graphs and discussions are provided in the **Assignment_3_Analysis.pdf** file to illustrate the speedup and efficiency of CUDA implementations compared to traditional CPU methods.

- **Question 1 Analysis:** Answers to the general questions about parallel computing are included in **Assignment_3_Analysis.pdf**.
- **Experimental Results:** Includes graphs plotting speedup versus input sizes, and observations on performance trends.
- **Discussion:** Explores why speedup varies with input size and architectural advantages of GPUs for certain tasks.
