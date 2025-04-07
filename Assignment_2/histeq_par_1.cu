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

__global__ void rgb_to_yuv_histogram(unsigned char *image, int width, int height, int cpp, int *d_lum_hist) {
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

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);

    // Copy image to device
    checkCuda(cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice), "Copying image to device");

    // Launch kernel
    int threadsPerBlock = 256;
    int numPixels = width * height;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    // 1. and 2. conversion to YUV and luminance histogram computation (simple method)
    rgb_to_yuv_histogram<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height, cpp, d_lum_hist);
    checkCuda(cudaGetLastError(), "Kernel launch");

    // 3. cumulative histogram computation (simple, sequential method)
    compute_cumulative_histogram_kernel<<<1, 1>>>(d_lum_hist, d_cum_hist, d_min_luminance);

    // 4. computing new luminance values
    compute_new_luminance_kernel<<<1, 256>>>(d_cum_hist, d_new_lum, width, height, d_min_luminance);

    // 5. and 6. assign new luminances to each pixel and convert YUV to RGB
    assign_new_lum_convert_YUV_to_RGB<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height, cpp, d_new_lum);

    // Copy result back to host (CPU)
    checkCuda(cudaMemcpy(h_image, d_image, img_size, cudaMemcpyDeviceToHost), "Copying image back to host");

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f seconds \n", milliseconds / 1000.0f);

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