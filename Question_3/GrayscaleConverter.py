from pycuda.compiler import SourceModule # Provide source module class for compiling C code
import pycuda.autoinit # Automatically initialize the CUDA driver and context
import pycuda.driver as cuda # Provide CUDA driver functions for memory management and kernel launches
import numpy as np # Used for array manipulation and image processing
import cv2 # OpenCV library for image I/O and processing
import time # Used for measuring execution time


mod = SourceModule("""                 
    /**
    * @brief Converts an RGB image to grayscale using the standard luminance formula.
    * 
    * This kernel calculates the intensity of each pixel in the new grayscale image
    * based on the RGB values of that pixel in the original image. It uses the ITU-R BT.601 standard
    * coefficients for luma calculation: 0.299 for red, 0.587 for green, and 0.114 for blue.
    * 
    * Unsigned char an 8-bit unsigned integer, ranging 0 from to 255, making it suitable for storing
    * pixel values.
    *
    * @param rgb Pointer to the input RGB image data (flattened 3D array)
    * @param gray Pointer to the output grayscale image data (flattened 2D array)
    * @param width Width of the input image
    * @param height Height of the input image
    */     
    __global__ void rgb_to_gray(unsigned char *rgb, unsigned char *gray, int width, int height)
    {
        // Get the column of the element the thread will be working on
        int col = threadIdx.x + blockIdx.x * blockDim.x;
                   
        // Get the row of the element the thread will be working on
        int row = threadIdx.y + blockIdx.y * blockDim.y;

        // If the element x and y indices are valid, calculate the gray pixel from the original rgb values
        if (col < width && row < height) {
            
            // Get the pixel this thread is calculating for the grayscale output
            // gray is a flattened 2D array with width * height indexes
            int grayOffset = row * width + col;
                   
            // Get the first RGB channel for this pixel in the original image by multiplying by 3 since each pixel has 3 channels
            // rgb is a flattened 3D array with width * height * 3 indexes      
            int rgbOffset = grayOffset * 3;
                   
            // Get the red intensity value the new gray pixel had in the original image
            unsigned char r = rgb[rgbOffset + 0];
                   
            // Get the green intensity value the new gray pixel had in the original image
            unsigned char g = rgb[rgbOffset + 1];
                   
            // Get the blue intensity value the new gray pixel had in the original image
            unsigned char b = rgb[rgbOffset + 2];
                   
            // We calculate the grayscale intensity using the luminosity  method, as it is the most accurate.
            // It uses a weighted average of the red, green, and blue intensities to calculate the new gray value.
            // We add 0.5f to conver the value to the nearest intensity before converting the float to an integer.
            gray[grayOffset] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b + 0.5f);
        }
    }
""")

# Read input image
img = cv2.imread('input.png')

# CPU Code:
# Start measuring CPU time
start_cpu = time.time()  

# Convert image to grayscale serially using cv2 library 
gray_img_cpu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Stop measuring CPU time
end_cpu = time.time()

# Get CPU time taken in milliseconds and print it
cpu_time = (end_cpu - start_cpu) * 1000
print(f"CPU execution time: {cpu_time:.2f} ms")   

# GPU Code:
# Extract width and height of the image, ignoring the 3 RGB channels
height, width, _ = img.shape
                   
# Create empty Numpy array to store output grayscale image
gray_img = np.empty((height, width), dtype=np.uint8)

# Allocate GPU memory for the input RGB image
rgb_img_gpu = cuda.mem_alloc(img.nbytes)

# Copy the image data from host to device memory
cuda.memcpy_htod(rgb_img_gpu, img)

# AAllocate GPU memory for the output grayscale image
gray_img_gpu = cuda.mem_alloc(gray_img.nbytes)

# Retrieve a reference to the rgb_to_gray kernel function from the compiled CUDA module
rgb_to_gray = mod.get_function("rgb_to_gray")

# The block size is set to (32, 32, 1) meaning each block has 32 x 32 threads
block_dim = (32, 32, 1)

# The grid size is calculated based on the image dimensions and block size
grid_dim_x = (width + block_dim[0] - 1) // block_dim[0] # Choose number of blocks in the x dimension to ensure there are enough threads in the x dimension (blocks.x * blockdim.x) to cover the full width for one row.
grid_dim_y = (height + block_dim[1] - 1) // block_dim[1] # Choose number of blocks in the y dimension to ensure there are enough threads in the y dimension (blocks.y * blockdim.y) to cover the full height for one column.
grid_dim = (grid_dim_x, grid_dim_y, 1) # Multiplying the grid_dim_x times grid_dim_y for the number of blocks ensures each row and column is fully covered.

# Start measuring GPU time
start_gpu = time.time()

# Call the GPU kernel to get the grayscale image
rgb_to_gray(rgb_img_gpu, gray_img_gpu, np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)

# Copy result from device memory to host memory
cuda.memcpy_dtoh(gray_img, gray_img_gpu)

# Stop measuring GPU time
end_gpu = time.time()

# Get time taken for GPU in ms
gpu_time = (end_gpu - start_gpu) * 1000

# Print GPU time and speedup over CPU 
print(f"GPU execution time: {gpu_time:.2f} ms")
speedup = cpu_time / gpu_time
print(f"Speedup: {speedup:.2f}x")

# Save output
cv2.imwrite('output_cpu.png', gray_img_cpu)
cv2.imwrite('output_gpu.png', gray_img)