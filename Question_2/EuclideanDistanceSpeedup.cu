/*
 * This program compares the performance of serial (CPU) and parallel (GPU/CUDA)
 * implementations of Euclidean distance calculation between two N-dimensional points.
 * It runs multiple tests with different array sizes and averages the results.
 */

// Include standard C libraries and the CUDA runtime library
#include <stdio.h>        // For input/output operations
#include <stdlib.h>       // For memory allocation and random number generation
#include <math.h>         // For mathematical functions like pow() and sqrt()
#include <time.h>         // For time functions and clock measurements
#include <cuda_runtime.h> // For CUDA functions and types

#define NUM_RUNS 10 // Number of times to run each test for averaging

// Define global variables
const int sizes[] = {10000, 50000, 100000, 500000, 1000000, 10000000}; // Array of different sizes to test
int numSizes = sizeof(sizes) / sizeof(sizes[0]);                       // Length of sizes array

double *hostA; // First point coordinates
double *hostB; // Second point coordinates

/**
 * @brief Initialize arrays with random numbers between 0 and 99
 *
 * @param size The size of the arrays to initialize.
 */
void initArrays(int size)
{
    for (int i = 0; i < size; ++i)
    {
        hostA[i] = (double)(rand() % 100);
        hostB[i] = (double)(rand() % 100);
    }
}

/**
 * @brief Calculates the Euclidean distance using serial computation.
 *
 * This function computes the Euclidean distance between two arrays A and B using
 * serial computation.
 *
 * @param size The size of the input arrays.
 * @return The Euclidean distance between the input arrays.
 */
double serialEuclideanDistance(int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; ++i)
    {
        sum += pow(hostA[i] - hostB[i], 2); // Calculate squared difference
    }
    return sqrt(sum); // Return square root of sum
}

/**
 * @brief CUDA kernel for partial sum calculation.
 *
 * This kernel calculates the squared differences between corresponding elements of
 * input arrays A and B and stores them in the partialSums array. Each
 * element in the partialSums array is the sum of all computations performed by
 * threads in a single block, where the element index corresponds to the block index.
 * Adding all these partial sum elements after will provide the total sum of squared differences.
 *
 * @param A Pointer to the first input array on the device.
 * @param B Pointer to the second input array on the device.
 * @param partialSums Pointer to the output array on the device.
 * @param size The size of the input arrays.
 */
__global__ void partialSumKernel(double *A, double *B, double *partialSums, int size)
{

    // Store intermediate results within each block
    extern __shared__ double sharedData[];

    // Thread index within the block
    unsigned int tid = threadIdx.x;

    // Global index of the thread
    unsigned int i = tid + blockDim.x * blockIdx.x;

    // Each thread calculates the squared difference for its assigned element and stores it in shared memory.
    // If i exceeds the array size, we store 0 to avoid going out of bounds.
    sharedData[tid] = (i < size) ? pow(A[i] - B[i], 2) : 0;

    // Ensure all threads in a block have completed their writes to shared memory before proceeding
    __syncthreads();

    // This loop performs the reduction, it halves the number of threads with each iteration
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {

        // Threads with indices less than s add the value from the corresponding partner thread at tid + s
        if (tid < s)
        {
            sharedData[tid] += sharedData[tid + s];
        }
        // Makes sure all threads complete this iteration before moving to the next, needed to prevent element sums from being skipped or race conditions
        __syncthreads();
    }

    // Once the reduction is complete, the first thread in each block writes the result of the reduction to the partialSums array.
    // This is the sum of the squared differences for this block
    if (tid == 0)
    {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

/**
 * @brief Cleans up allocated memory.
 *
 * This function frees the memory allocated for hostA and hostB arrays.
 */
void cleanup()
{
    free(hostA);
    free(hostB);
}

int main()
{

    // Print the number of CUDA-enabled hardware devices
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    printf("Number of CUDA-enabled devices: %d\n", numDevices);

    // Print properties of Device 0
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Device Name: %s\n", props.name);
    printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("Max Grid Size: %d x %d x %d\n\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);

    // Seed the random number generator once at the beginning
    srand(time(NULL));

    // Find the maximum size in the sizes array
    int maxSize = 0;
    for (int i = 0; i < numSizes; i++)
    {
        if (sizes[i] > maxSize)
        {
            maxSize = sizes[i];
        }
    }

    // Allocate memory for the arrays
    hostA = (double *)malloc(sizeof(double) * maxSize);
    hostB = (double *)malloc(sizeof(double) * maxSize);

    // Loop through different array sizes
    for (int i = 0; i < numSizes; i++)
    {
        int size = sizes[i];

        // Arrays to store timing and distance results for each run
        double serialTimes[NUM_RUNS];
        double cudaTimes[NUM_RUNS];
        double serialDistances[NUM_RUNS];
        double cudaDistances[NUM_RUNS];

        // Perform multiple runs for each size for averaging
        for (int run = 0; run < NUM_RUNS; run++)
        {
            // Initialize arrays with random numbers for run
            initArrays(size);

            // Serial (CPU) version
            clock_t start = clock();
            double serialDistance = serialEuclideanDistance(size);
            clock_t end = clock();
            serialTimes[run] = (double)(end - start) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
            serialDistances[run] = serialDistance;

            // CUDA (GPU) version
            double *d_A, *d_B, *d_partialSums; // Device arrays

            // Allocate memory on GPU
            cudaMalloc(&d_A, sizeof(double) * size);
            cudaMalloc(&d_B, sizeof(double) * size);
            cudaMalloc(&d_partialSums, sizeof(double) * size);

            // Copy data from host to device
            cudaMemcpy(d_A, hostA, sizeof(double) * size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, hostB, sizeof(double) * size, cudaMemcpyHostToDevice);

            // Set block to 256, a multiple of the fixed size of warps
            int blockSize = 256;

            // Calculate to ensure all elements are processed even if size is not a multiple of blockSize
            int numBlocks = (size + blockSize - 1) / blockSize;

            // Specify the amount of shared memory to allocate per block
            size_t sharedMemSize = blockSize * sizeof(double);

            // Launch kernel and measure time, calculating partial sums of squared differences for each block
            start = clock();
            partialSumKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_A, d_B, d_partialSums, size);

            // Make sure device computation has finished for all blocks
            cudaDeviceSynchronize();

            // Copy results back to host and calculate final distance
            double *partialSums = (double *)malloc(sizeof(double) * numBlocks);
            cudaMemcpy(partialSums, d_partialSums, sizeof(double) * numBlocks, cudaMemcpyDeviceToHost);

            // Sum up partial results of each block
            double cudaSum = 0.0;
            for (int j = 0; j < numBlocks; ++j)
            {
                cudaSum += partialSums[j];
            }

            // Calculate the final Euclidean distance by taking the square root of the sum of the squared differences
            double cudaDistance = sqrt(cudaSum);

            // Record timing and distance
            end = clock();
            cudaTimes[run] = (double)(end - start) / CLOCKS_PER_SEC * 1000; // Convert to miliseconds
            cudaDistances[run] = cudaDistance;

            // Free GPU memory
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_partialSums);
            free(partialSums);
        }

        // Calculate averages for array size
        double avgSerialTime = 0.0;
        double avgCudaTime = 0.0;
        double avgSerialDistance = 0.0;
        double avgCudaDistance = 0.0;

        for (int run = 0; run < NUM_RUNS; run++)
        {
            avgSerialTime += serialTimes[run];
            avgCudaTime += cudaTimes[run];
            avgSerialDistance += serialDistances[run];
            avgCudaDistance += cudaDistances[run];
        }

        avgSerialTime /= NUM_RUNS;
        avgCudaTime /= NUM_RUNS;
        avgSerialDistance /= NUM_RUNS;
        avgCudaDistance /= NUM_RUNS;

        // Calculate average speedup
        double averageSpeedup = avgSerialTime / avgCudaTime;

        // Print average results for array size
        printf("Size: %d\n", size);
        printf("Average Serial Time: %.2f ms\n", avgSerialTime);
        printf("Average CUDA Time: %.2f ms\n", avgCudaTime);
        printf("Average Speedup: %.2fx\n", averageSpeedup);
        printf("Average Serial Distance: %.4f\n", avgSerialDistance);
        printf("Average CUDA Distance: %.4f\n", avgCudaDistance);

        // Print individual run results Note: Only used for testing distance results
        /**printf("\nIndividual Run Results:\n");
        for (int run = 0; run < NUM_RUNS; run++)
        {
            printf("Run %d:\n", run + 1);
            printf("Serial Time: %.2f ms\n", serialTimes[run]);
            printf("CUDA Time: %.2f ms\n", cudaTimes[run]);
            printf("Serial Distance: %.4f\n", serialDistances[run]);
            printf("CUDA Distance: %.4f\n", cudaDistances[run]);
            printf("\n");
        }**/
        printf("\n");
    }

    // Clean up and exit
    cleanup();
    return 0;
}