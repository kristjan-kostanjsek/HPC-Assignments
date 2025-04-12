// little bit more advanced parallel - uses partial histograms for luminance but doesn't use Blelloch reduction for cumulative histogram

// nvcc  -diag-suppress 550 -O2 -lm histeq_par_1.cu -o histeq_par_1
// ./histeq_par_1 in_images/720x480.png out_images/test.png

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

__device__ unsigned char clamp(float x) {
    return (unsigned char)(fmaxf(0.0f, fminf(255.0f, x)));
}

__global__ void global_rgb_to_yuv_histogram(unsigned char *image, int width, int height, int cpp, int *d_lum_hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    if (idx >= size) return;

    int i = idx * cpp;

    unsigned char R = image[i];
    unsigned char G = image[i + 1];
    unsigned char B = image[i + 2];

    float Y = 0.299f * R + 0.587f * G + 0.114f * B;
    float U = -0.14713f * R - 0.28886f * G + 0.436f * B + 128.0f;
    float V = 0.615f * R - 0.51499f * G - 0.10001f * B + 128.0f;

    unsigned char y = clamp(Y);
    unsigned char u = clamp(U);
    unsigned char v = clamp(V);

    image[i + 0] = y;
    image[i + 1] = u;
    image[i + 2] = v;

    atomicAdd(&d_lum_hist[y], 1);
}

__global__ void basic_shared_rgb_to_yuv_histogram(unsigned char *image, int width, int height, int cpp, int *d_lum_hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int size = width * height;

    // Declare shared memory for partial histogram (local to each block)
    __shared__ int local_hist[256];

    // Initialize shared memory histogram to zero
    if (tid < 256) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // Process pixels and update shared histogram
    if (idx < size) {
        int i = idx * cpp;

        unsigned char R = image[i];
        unsigned char G = image[i + 1];
        unsigned char B = image[i + 2];

        float Y = 0.299f * R + 0.587f * G + 0.114f * B;
        float U = -0.14713f * R - 0.28886f * G + 0.436f * B + 128.0f;
        float V = 0.615f * R - 0.51499f * G - 0.10001f * B + 128.0f;

        unsigned char y = clamp(Y);
        unsigned char u = clamp(U);
        unsigned char v = clamp(V);

        image[i + 0] = y;
        image[i + 1] = u;
        image[i + 2] = v;

        // Atomic add to block shared histogram
        atomicAdd(&local_hist[y], 1);
    }

    __syncthreads();

    // After local histogram is ready, a few threads combine it to global
    if (tid < 256) {
        atomicAdd(&d_lum_hist[tid], local_hist[tid]);
    }
}

__global__ void bin_shared_rgb_to_yuv_histogram(unsigned char *image, int width, int height, int cpp, int *d_lum_hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bin_index = tid % 4;
    int size = width * height;

    __shared__ int local_hist[256][4];

    // Initialize all entries of shared histogram (1024 total)
    int row = tid / 4;
    int col = tid % 4;
    if (row < 256) {
        local_hist[row][col] = 0;
    }

    __syncthreads();

    if (idx < size) {
        int i = idx * cpp;
        unsigned char R = image[i];
        unsigned char G = image[i + 1];
        unsigned char B = image[i + 2];

        float Y = 0.299f * R + 0.587f * G + 0.114f * B;

        unsigned char y = clamp(Y);

        image[i + 0] = y;
        image[i + 1] = clamp(-0.14713f * R - 0.28886f * G + 0.436f * B + 128.0f);
        image[i + 2] = clamp(0.615f * R - 0.51499f * G - 0.10001f * B + 128.0f);

        atomicAdd(&local_hist[y][bin_index], 1);
    }

    __syncthreads();

    // Reduce privatized histogram
    if (tid < 256) {
        int bin_value = local_hist[tid][0] + local_hist[tid][1] +
                        local_hist[tid][2] + local_hist[tid][3];
        atomicAdd(&d_lum_hist[tid], bin_value);
    }
}

__global__ void bitmap_shared_rgb_to_yuv_histogram(unsigned char *image, int width, int height, int cpp, int *d_lum_hist) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int size = width * height;

    // Shared memory: one 32-bit bitmap per warp per luminance bin
    __shared__ unsigned int bitmap[256][32]; // 256 bins, 32 warps

    // Init shared memory (each thread clears one cell)
    if (tid < 256 * 32) {
        int bin = tid / 32;
        int w = tid % 32;
        bitmap[bin][w] = 0;
    }

    __syncthreads();

    // Step 1: Process pixel
    if (idx < size) {
        int i = idx * cpp;

        unsigned char R = image[i];
        unsigned char G = image[i + 1];
        unsigned char B = image[i + 2];

        float Y = 0.299f * R + 0.587f * G + 0.114f * B;
        float U = -0.14713f * R - 0.28886f * G + 0.436f * B + 128.0f;
        float V =  0.615f * R - 0.51499f * G - 0.10001f * B + 128.0f;

        unsigned char y = clamp(Y);
        unsigned char u = clamp(U);
        unsigned char v = clamp(V);

        // Write back YUV
        image[i + 0] = y;
        image[i + 1] = u;
        image[i + 2] = v;

        // Mark vote in bitmap
        atomicOr(&bitmap[y][warp_id], 1 << lane_id);
    }

    __syncthreads();

    // Step 2: Reduce bitmap and write to global histogram
    if (tid < 256) {
        int bin = tid;
        int count = 0;
        for (int w = 0; w < 32; ++w) {
            count += __popc(bitmap[bin][w]);
        }
        atomicAdd(&d_lum_hist[bin], count);
    }
}

__global__ void big_shared_rgb_to_yuv_histogram(unsigned char *image, int width, int height, int cpp, int *d_lum_hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;  // 0-1023
    int lane_id = tid % 32; // 0-31 (warp lane)
    int warp_id = tid / 32; // 0-31 (for 1024-thread block)
    int size = width * height;

    __shared__ int block_hist[32][256]; 

    for (int i = 0; i < 8; i++) {
        block_hist[warp_id][lane_id*8 + i] = 0; // Initialize shared histogram to zero
    }
    
    __syncthreads();

    if (idx < size) {
        // RGB to YUV conversion (unchanged)
        int i = idx * cpp;
        unsigned char R = image[i];
        unsigned char G = image[i + 1];
        unsigned char B = image[i + 2];
        float Y = 0.299f * R + 0.587f * G + 0.114f * B;
        unsigned char y = clamp(Y);

        // Update YUV (unchanged)
        image[i] = y;
        image[i+1] = clamp(-0.14713f*R - 0.28886f*G + 0.436f*B + 128.0f);
        image[i+2] = clamp(0.615f*R - 0.51499f*G - 0.10001f*B + 128.0f);

        atomicAdd(&block_hist[warp_id][y], 1);
    } 
    __syncthreads();

    // Step 2: Merge warp histograms (coalesced, 1 atomic per bin)
    if (tid < 256) {
        int bin_value = 0;
        for (int i = 0; i < 32; i++) {
            bin_value += block_hist[i][tid];
        }
        atomicAdd(&d_lum_hist[tid], bin_value);
    }
}

__global__ void compute_cumulative_histogram_kernel(int *lum_hist, int *cum_hist, int *min_luminance) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int min_lum = 0;
        cum_hist[0] = lum_hist[0];

        for (int i = 1; i < 256; i++) {
            cum_hist[i] = cum_hist[i - 1] + lum_hist[i];
            
            // Find minimum non zero value
            if (cum_hist[i - 1] > 0 && min_lum == 0) {
                min_lum = cum_hist[i - 1];
            }
        }

        *min_luminance = min_lum;
    }
}

__global__ void imp_compute_cumulative_histogram_kernel(int *lum_hist, int *cum_hist, int *min_luminance) {
    // Shared memory for intermediate scan calculations
    // Size 256 to match the histogram bins (0-255)
    __shared__ int temp[256];  

    // Thread ID within the block (0-255)
    int tid = threadIdx.x;
    
    // Phase 1: Load input into shared memory
    // Each thread loads one histogram bin
    temp[tid] = lum_hist[tid];
    __syncthreads();  // Ensure all threads have loaded data
    
    // Phase 2: Up-sweep (reduction) phase
    // Builds a binary tree of partial sums
    for (int k = 1; k < 256; k <<= 1) {  // k doubles each iteration: 1, 2, 4, 8, 16, 32, 64, 128, 256
        // Only threads at even multiples of 2k participate
        if ((tid+1) % (k<<1) == 0) {
            // Add the value k positions away to current position
            temp[tid] += temp[tid-k];
        }
        __syncthreads();  // Sync after each tree level
    }

    // Phase 3: Down-sweep phase
    // Propagates partial sums back down the tree
    // Down-sweep phase
    for (int k = 8; k >= 1; k--) { // k: 8, 7, ... 1
        int stride = 1 << k;                 // 2^k
        int half_stride = stride >> 1;       // 2^{k-1}
        
        // Each thread handles one element
        int i = tid * stride;
        int idx = i - 1 + half_stride + stride;
        
        if (idx < 256) {
            temp[idx] += temp[i - 1 + stride];
        }
        __syncthreads();
    }

    // Phase 5: Store final results
    cum_hist[tid] = temp[tid];

    if (tid == 0) {
        // Find minimum non-zero value in the cumulative histogram
        int min_lum = 0;
        for (int i = 0; i < 256; i++) {
            if (cum_hist[i] > 0 && min_lum == 0) {
                min_lum = cum_hist[i];
                break;
            }
        }
        *min_luminance = min_lum;
    }
}

__global__ void compute_new_luminance_kernel(int *cum_hist, int *new_lum, int width, int height, int *min_luminance) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread index
    if (idx >= 256) return;  // Each thread processes one luminance value (0-255)

    int cum_lum = cum_hist[idx];
    int up = cum_lum - *min_luminance;
    int down = (width * height) - *min_luminance;
    float division = (float)up / down;  // Avoid integer division
    int new_luminance = floorf(division * 255);

    new_lum[idx] = new_luminance;  // Store the result
}

__global__ void assign_new_lum_convert_YUV_to_RGB(unsigned char *image, int width, int height, int cpp, int *new_lum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    if (idx >= size) return;

    int l = (int)roundf(image[idx * cpp]);
    image[idx * cpp] = new_lum[l]; // assign new luminance

    // Convert YUV back to RGB
    unsigned char Y = image[idx * cpp + 0];
    unsigned char U = image[idx * cpp + 1];
    unsigned char V = image[idx * cpp + 2];

    image[idx * cpp + 0] = fminf(255.0f, fmaxf(0.0f, Y + 1.402f * (V - 128.0f)));
    image[idx * cpp + 1] = fminf(255.0f, fmaxf(0.0f, Y - 0.344136f * (U - 128.0f) - 0.714136f * (V - 128.0f)));
    image[idx * cpp + 2] = fminf(255.0f, fmaxf(0.0f, Y + 1.772f * (U - 128.0f)));
}

void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("USAGE: histeq_par_1 input_image output_image\n");
        return 1;
    }

    char *input = argv[1];
    char *output = argv[2];

    int width, height, cpp;
    unsigned char *h_image = stbi_load(input, &width, &height, &cpp, 0);
    if (!h_image) {
        printf("Failed to load image %s\n", input);
        return 1;
    }

    size_t img_size = width * height * cpp * sizeof(unsigned char);
    size_t hist_size = 256 * sizeof(int);

    // Allocate device memory
    unsigned char *d_image;
    int *d_lum_hist;

    
    checkCuda(cudaMalloc(&d_image, img_size), "Allocating image");
    checkCuda(cudaMalloc(&d_lum_hist, hist_size), "Allocating luminance histogram");
    checkCuda(cudaMemset(d_lum_hist, 0, hist_size), "Clearing luminance histogram");
    int *d_min_luminance;
    checkCuda(cudaMalloc(&d_min_luminance, sizeof(int)), "Allocating min luminance");
    int *d_cum_hist;
    checkCuda(cudaMalloc(&d_cum_hist, hist_size), "Allocating cumulative histogram");
    int *d_new_lum;
    checkCuda(cudaMalloc(&d_new_lum, hist_size), "Allocating new luminance table");

    /*
    int h_lum_hist[256];
    for (int i = 0; i < 256; i++) {
        h_lum_hist[i] = i;
    }*/


    // Use CUDA events to measure execution time
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_hist, stop_hist;
    cudaEvent_t start_cumulative, stop_cumulative;
    cudaEvent_t start_lum, stop_lum;
    cudaEvent_t start_assign, stop_assign;

    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_hist);
    cudaEventCreate(&stop_hist);
    cudaEventCreate(&start_cumulative);
    cudaEventCreate(&stop_cumulative);
    cudaEventCreate(&start_lum);
    cudaEventCreate(&stop_lum);
    cudaEventCreate(&start_assign);
    cudaEventCreate(&stop_assign);

    // Copy image to device
    checkCuda(cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice), "Copying image to device");
    //checkCuda(cudaMemcpy(d_lum_hist, h_lum_hist, hist_size, cudaMemcpyHostToDevice), "Copying luminance histogram to device");

    // Launch kernel
    int threadsPerBlock = 1024; // TODO change threads per block here
    int numPixels = width * height;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    // Start total timer
    cudaEventRecord(start_total);

    // 1. and 2. conversion to YUV and luminance histogram computation (simple method)
    cudaEventRecord(start_hist);
    bin_shared_rgb_to_yuv_histogram<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height, cpp, d_lum_hist);
    cudaEventRecord(stop_hist);
    
    // 2. Cumulative Histogram
    cudaEventRecord(start_cumulative);
    imp_compute_cumulative_histogram_kernel<<<1, 256>>>(d_lum_hist, d_cum_hist, d_min_luminance);
    cudaEventRecord(stop_cumulative);
    
    // 3. New Luminance Values
    cudaEventRecord(start_lum);
    compute_new_luminance_kernel<<<1, 256>>>(d_cum_hist, d_new_lum, width, height, d_min_luminance);
    cudaEventRecord(stop_lum);

    // 4. Assign Luminance + YUV→RGB
    cudaEventRecord(start_assign);
    assign_new_lum_convert_YUV_to_RGB<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height, cpp, d_new_lum);
    cudaEventRecord(stop_assign);
    checkCuda(cudaGetLastError(), "Kernel launch");

    // Final sync
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // Copy result back to host (CPU)
    checkCuda(cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost), "Copying image back to host");
    
    /* cumulative hist testing
    int h_cum_hist[256];
    int h_min_luminance;

    // Copy device data to host
    cudaMemcpy(h_lum_hist, d_lum_hist, 256*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum_hist, d_cum_hist, 256*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_min_luminance, d_min_luminance, sizeof(int), cudaMemcpyDeviceToHost);

    // Print histogram
    printf("Luminance and cum Histogram:\n");
    for (int i = 0; i < 256; i++) {
        printf("Bin %3d\t| Count %d\t\t\t| Cum %d\n", i, h_lum_hist[i], h_cum_hist[i]);
    }*/

    float total_ms = 0, hist_ms = 0, cum_ms = 0, lum_ms = 0, assign_ms = 0;

    cudaEventElapsedTime(&hist_ms, start_hist, stop_hist);
    cudaEventElapsedTime(&cum_ms, start_cumulative, stop_cumulative);
    cudaEventElapsedTime(&lum_ms, start_lum, stop_lum);
    cudaEventElapsedTime(&assign_ms, start_assign, stop_assign);
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    printf("STATS, %.5f, %.5f, %.5f, %.5f, %.5f\n", hist_ms, cum_ms, lum_ms, assign_ms, total_ms);

    // Save output
    if (!stbi_write_png(output, width, height, cpp, h_image, width * cpp)) {
        printf("Failed to save image %s\n", output);
    }

    // Free resources
    cudaFree(d_image);
    cudaFree(d_lum_hist);
    cudaFree(d_min_luminance);
    cudaFree(d_cum_hist);
    cudaFree(d_new_lum);
    stbi_image_free(h_image);

    return 0;
}