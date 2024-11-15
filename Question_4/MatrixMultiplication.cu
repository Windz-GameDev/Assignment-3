#include <stdio.h>        // Standard Input / Output functions (printf)
#include <stdlib.h>       // General-purpose standard library functions (random number generation, malloc(), etc)
#include <cuda_runtime.h> // Cuda Runtime API needed for CUDA functionality
#include <time.h>         // Time-related functions, used for seeding random number generator

#define NUM_RUNS 10 // Define a constant for the number of times each matrix size will be multiplied to get an average time.

// Define an array of integers containing the sizes of matrices to be tested. These represent square matrices 256x256, 512x512, etc.
const int matrixSizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

// Calculate the number of elements in the matrixSizes array dynamically.
const int NUM_SIZES = sizeof(matrixSizes) / sizeof(matrixSizes[0]);

/**
 * @brief Initializes a size * size matrix with random float values between 0 and 1.
 *
 * @param matrix Pointer to the matrix to be initialized.
 * @param size Row and column length of the square matrix.
 */
void initializeMatrix(float *matrix, int size)
{
    // The loop iterates size * size times, covering all elements of the matrix.
    for (int i = 0; i < size * size; i++)
    {
        // For each element, a random float value is generated using rand().
        // We scale it between 0 and 1 by dividing by RAND_MAX.
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * @brief Performs matrix multiplication on the CPU.
 *
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param C Pointer to the output matrix.
 * @param size Row and column length of the square matrices.
 */
void cpuMatrixMultiply(float *A, float *B, float *C, int size)
{
    // Iterate through each row of the result matrix
    for (int i = 0; i < size; i++)
    {
        // For each row in result matrix, iterate through each column
        for (int j = 0; j < size; j++)
        {

            // Dot product for current element will be accumulated here
            float sum = 0.0f;
            // For each element (i, j) in the result matrix calculate it's value as the dot product
            // of the ith row in matrix A and the jth column in matrix B.
            // This is done by multiplying each corresponding kth element in the ith row and jth column
            // and adding their product to a sum.
            for (int k = 0; k < size; k++)
            {
                // i * size gets us to the ith row, then adding k gets us to the current element in matrix A being multiplied in the dot product.
                // Size * k gets us to the current row of the element being multiplied in the dot product in matrix B, then adding j ensures we are moving through the jth column's rows.
                sum += A[i * size + k] * B[k * size + j];
            }
            // Once each corresponding element in the ith row and jth column has been multiplied and added to the sum, store it at (i, j) in the result matrix.
            // i * size gets the first column in the ith row, then adding j gets us to the jth column in that row.
            C[i * size + j] = sum;
        }
    }
}

/**
 * @brief CUDA kernel function to perform matrix multiplication on the GPU.
 *
 * Each thread calulates one element of the result matrix that has a value of the dot product
 * of it's corresponding row and column in the A and B matrices respectively.
 *
 * @param A Pointer to the first input matrix on the device.
 * @param B Pointer to the second input matrix on the device.
 * @param C Pointer to the output matrix on the device.
 * @param size Row and column length of the square matrices.
 */
__global__ void gpuMatrixMultiply(float *A, float *B, float *C, int size)
{

    // Get the row of the result element this thread will be working on
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Get the column of the result element this thread will be working on
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure this thread's global index corresponding to a valid element index in the result matrix
    // This is because the final block may have extra indexes if the problem size isn't evenly divisible by the block size
    if (row < size && col < size)
    {
        // Keep track of the sum for the dot product
        float sum = 0.0f;

        // Go through each corresponding kth element element in the current row in the A matrix and current column in the B matrix, multiply them, and accumulate their products in a sum.
        for (int k = 0; k < size; k++)
        {
            sum += A[row * size + k] * B[k * size + col];
        }

        // Store the sum in the appropiate location in the result matrix.
        // The element's row and column matches the row in matrix A, and column in matrix B respectively whose dot product was calculated.
        C[row * size + col] = sum;
    }
}

/**
 * @brief Prints a square matrix to the console.
 *
 * This function takes a pointer to a 1D array representing a square matrix
 * and its size, and prints it in a 2D format to the console.
 *
 * @param matrix Pointer to the 1D array representing the square matrix.
 * @param size The number of rows (and columns) in the square matrix.
 */
void printMatrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Compares two matrices element-wise.
 *
 * This function compares two matrices of the same size element-wise and returns
 * true if all elements are within a specified tolerance, false otherwise.
 *
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @param size The size of the square matrices.
 * @param tolerance The maximum allowed difference between elements.
 * @return true if the matrices match within the tolerance, false otherwise.
 */
bool compareMatrices(float *A, float *B, int size, float tolerance)
{
    for (int i = 0; i < size * size; i++)
    {
        if (fabs(A[i] - B[i]) > tolerance)
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Main function that orchestrates the matrix multiplication benchmark.
 *
 * @param argc Number of command-line arguments (not used in this program).
 * @param argv Array of command-line argument strings (not used in this program).
 * @return 0 on successful execution.
 */
int main(int argc, char **argv)
{
    // Called to seed the random number generator with the current time, ensuring different random sequences each time the program is run.
    srand(time(NULL));

    // Declare the prop variable that will hold our CUDA device properties
    cudaDeviceProp prop;

    // Retrieve the properties of the CUDA device with index 0, and store them in the prop variable called prop
    cudaGetDeviceProperties(&prop, 0);

    // Get the maximum number of threads per block based on device
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    // The maximum block size for the x and y dimensions is calculated as the square root of the maximum threads per block.
    int maxBlockSizeX = sqrt(maxThreadsPerBlock);
    int maxBlockSizeY = maxBlockSizeX;

    // Print maximum threads per block and the chosen block size for the GPU kernel
    printf("Max threads per block: %d\n", maxThreadsPerBlock);
    printf("Using block size: %dx%d\n", maxBlockSizeX, maxBlockSizeY);

    // Declare variables for the block size and grid size using the CUDA dim3 type.
    // The block sizes for the x and y dimensions are set to the values determined earlier.
    dim3 blockSize(maxBlockSizeX, maxBlockSizeY);
    dim3 gridSize;

    // Declare pointers for the matrices on the host.
    float *h_A, *h_B, *h_C;

    // Declare pointer for CPU result, h_C will store GPU result on host
    float *h_C_cpu;

    // Declare pointers for matrices on the device.
    float *d_A, *d_B, *d_C;

    // Declare event variables and timing variables for measuring GPU execution time
    cudaEvent_t start, stop;
    float gpuTime, cpuTime;

    // Start of a loop that iterates over each matrix size in the matrixSizes array.
    for (int i = 0; i < NUM_SIZES; i++)
    {
        // Current matrix size is stored in the size variable
        int size = matrixSizes[i];

        // The grid size in each dimension is calculated by dividing
        // the matrix size (width and height are the same) by the
        // block size of that dimension and rounding up to the nearest integer.
        gridSize.x = (size + blockSize.x - 1) / blockSize.x;
        gridSize.y = (size + blockSize.y - 1) / blockSize.y;

        // Allocate memory on the host for input matrices h_A and h_B.
        h_A = (float *)malloc(size * size * sizeof(float));
        h_B = (float *)malloc(size * size * sizeof(float));

        // Allocate memory on the host for the device execution output matrix h_C. (Device output will be moved here)
        h_C = (float *)malloc(size * size * sizeof(float));

        // Allocate memory on the host for the host execution output matrix h_C_cpu
        h_C_cpu = (float *)malloc(size * size * sizeof(float));

        // Allocate memory on the CUDA device for the input matrices d_A and D_B using cudaMalloc.
        cudaMalloc(&d_A, size * size * sizeof(float));
        cudaMalloc(&d_B, size * size * sizeof(float));

        // Allocate memory on the CUDA device for the output matrix d_C.
        cudaMalloc(&d_C, size * size * sizeof(float));

        // Initialize the input matrices h_A and h_B with random values using the initializeMatrix function.
        initializeMatrix(h_A, size);
        initializeMatrix(h_B, size);

        // Copy the input matrices from the host memory to the device memory using cudaMemcpy with the cudaMemcpyHostToDevice flag.
        cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);

        // CPU computation
        cpuTime = 0.0f;

        // Matrix multiplication for this matrix size is performed 10 times on CPU.
        for (int j = 0; j < NUM_RUNS; j++)
        {
            // Record current clock time before calling matrix multiply function.
            clock_t begin = clock();

            // Multiply matrices h_A and h_B serially and store the product in h_C_cpu
            cpuMatrixMultiply(h_A, h_B, h_C_cpu, size);

            // Record the current clock time after function completes
            clock_t end = clock();

            // Add elapsed time to the cpuTime variable to get total time taken so far across all runs.
            cpuTime += (float)(end - begin) / CLOCKS_PER_SEC; // Calculate difference between the end and start times in clock ticks and divide by clock ticks per second to convert to seconds.
        }

        // Get the average CPU execution time for this matrix size by dividing the total time taken by the number of runs.
        cpuTime /= NUM_RUNS;
        cpuTime = cpuTime * 1000; // Convert to miliseconds

        // Print matrices for small size to verify results
        if (size == 2)
        {
            printf("Matrix A:\n");
            printMatrix(h_A, size);
            printf("Matrix B:\n");
            printMatrix(h_B, size);
            printf("CPU Result:\n");
            printMatrix(h_C_cpu, size);
        }

        // GPU computation
        gpuTime = 0.0f;
        // Matrix multiplication for this matrix size is performed 10 times on GPU.
        for (int j = 0; j < NUM_RUNS; j++)
        {
            // Create CUDA start and stop event objects using cudaEventCreate and have the start and stop variables declared earlier store the references to them.
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Record the start event using cudaEventRecord to mark the beginning of the GPU kernel execution.
            // The second argument specifies the CUDA stream to associate with the event, 0 represents the default stream.
            cudaEventRecord(start, 0);

            // Launch the gpuMatrixMultiply kernel with the specified grid and block size, with the device matrices and matrix size as arguments.
            // The kernel performs the matrix multiplication on the GPU
            gpuMatrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);

            // The stop event is recorded using cudaEventRecord to mark the end of the GPU kernel execution
            // 0 specifies to associate the default CUDA stream with the event.
            cudaEventRecord(stop, 0);

            // Call to ensure the stop event has completed.
            // Synchronize the host (CPU) with the stop event using the cudaEventSynchronize function.
            // Ensures the host has waited until the stop event has been recorded, and that all GPU operations have been completed.
            cudaEventSynchronize(stop);

            // Declare variable of type float to store elapsed time for this run
            float tempTime;

            // Calculate the elapsed time between the start and stop events using cudaEventElapsedTime and store it in tempTime.
            cudaEventElapsedTime(&tempTime, start, stop); // Time is returned in milliseconds

            // Add this run time to the total gpuTime across all iterations
            gpuTime += tempTime;

            // Destroy the CUDA event objects using cudaEventDestroy and free up resources
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Get average GPU execution time for this matrix size
        gpuTime /= NUM_RUNS; // No need to convert to milliseconds, cudaEventElapsed time gives ms already.

        // Move GPU product matrix to host
        cudaMemcpy(h_C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

        // Print matrices for small size to verify results
        if (size == 2)
        {
            printf("GPU Result:\n");
            printMatrix(h_C, size);
        }

        // Compare CPU and GPU results for sizes <= 1024
        if (size <= 1024)
        {
            bool match = compareMatrices(h_C, h_C_cpu, size, 1e-4);
            printf("CPU and GPU results for %dx%d matrix %s\n", size, size, match ? "match" : "do not match");
        }
        else
        {
            printf("CPU and GPU results for %dx%d matrix not compared (size > 1024) due to performance reasons and potential floating-point precision limitations\n", size, size);
        }

        // Print the matrix size, average CPU time, average GPU time, and the speedup achieved by the GPU over the CPU for the current matrix size.
        printf("Matrix size: %dx%d\n", size, size);
        printf("Average CPU time: %.2f ms\n", cpuTime);
        printf("Average GPU time: %.2f ms\n", gpuTime);
        printf("Average Speedup: %.2fx\n", cpuTime / gpuTime);
        printf("\n");

        // Free the allocated memory on the host using free
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_cpu);

        // Free the allocated memory on the device using cudaFree
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // Return 0 to indicate successful execution
    return 0;
}