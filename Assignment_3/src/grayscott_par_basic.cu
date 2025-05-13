// nvcc -diag-suppress 550 -O2 -lm grayscott_par_1.c -o grayscott_par_1
// ./grayscott_par_1 256 5000 1 0.16 0.08 0.060 0.062

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// just for testing (saving the final image)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// initialize U and V grids (2D) with a square in the middle
void initUV(float *U, float *V, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            U[i * n + j] = 1.0f;
            V[i * n + j] = 0.0f;
        }
    }
    int r = n / 8;
    for (int i = n / 2 - r; i < n / 2 + r; i++) {
        for (int j = n / 2 - r; j < n / 2 + r; j++) {
            U[i * n + j] = 0.75f;
            V[i * n + j] = 0.25f;
        }
    }
}

// save float array as an image, just for testing
void save_grayscale_image(const char* filename, float* data, int n) {
    // allocate a buffer for 8-bit pixels
    unsigned char* pixels = (unsigned char*)malloc(n * n * sizeof(unsigned char));
    // convert float [0, 1] to unsigned char [0, 255]
    for (int i = 0; i < n * n; i++) {
        pixels[i] = (unsigned char)(data[i] * 255.0f);
    }
    // save as PNG
    stbi_write_png(filename, n, n, 1, pixels, n);
    free(pixels);
}

__global__ void gray_scott_kernel(float* U, float* V, float* U_new, float* V_new, int n, float dt, float du, float dv, float f, float k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        // Wrap-around indices using modulus
        int up    = (i - 1 + n) % n;
        int down  = (i + 1) % n;
        int left  = (j - 1 + n) % n;
        int right = (j + 1 + n) % n;

        float center_u = U[i * n + j];
        float center_v = V[i * n + j];

        float laplacian_u = U[up * n + j] + U[down * n + j] + U[i * n + left] + U[i * n + right] - 4.0f * center_u;
        float laplacian_v = V[up * n + j] + V[down * n + j] + V[i * n + left] + V[i * n + right] - 4.0f * center_v;

        float UV_square = center_u * center_v * center_v;

        U_new[i * n + j] = center_u + dt * (-UV_square + f * (1.0f - center_u) + du * laplacian_u);
        V_new[i * n + j] = center_v + dt * ( UV_square - (f + k) * center_v + dv * laplacian_v);
    }
}

// here is where the magic happens, baby
float gray_scott(float* U, float* V, float* U_new, float* V_new, int n, int steps, float dt, float du, float dv, float f, float k) {
    // Device pointers
    float *d_U, *d_V, *d_U_new, *d_V_new;
    int size = n * n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_U, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_U_new, size);
    cudaMalloc(&d_V_new, size);

    // Copy initial data to device
    cudaMemcpy(d_U, U, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(32, 32);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);
    for (int cur_step = 0; cur_step < steps; cur_step++) {
        gray_scott_kernel<<<gridSize, blockSize>>>(d_U, d_V, d_U_new, d_V_new, n, dt, du, dv, f, k);

        // Swap pointers (on device)
        float* tmp;

        tmp    = d_U;
        d_U    = d_U_new;
        d_U_new= tmp;

        tmp    = d_V;
        d_V    = d_V_new;
        d_V_new= tmp;
    }
    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(U, d_U, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_U_new);
    cudaFree(d_V_new);

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds;
}

int main(int argc, char *argv[]) {

    if (argc < 8) {
        printf("USAGE: grayscott_seq width steps time_step diff_u diff_v feed_factor kill_rate\n");
        exit(EXIT_FAILURE);
    }

    int n = atoi(argv[1]); // grid width
    int steps = atoi(argv[2]); // number of steps of the algorithm
    float dt = atof(argv[3]); // time step size
    float du = atof(argv[4]); // diffusion rate for u
    float dv = atof(argv[5]); // diffusion rate for v
    float f = atof(argv[6]); // feed rate
    float k = atof(argv[7]); // kill rate

    // allocate memory for the grids
    int grid_size = n * n;
    float *U      = (float*) malloc(grid_size * sizeof(float));
    float *V      = (float*) malloc(grid_size * sizeof(float));
    float *U_new  = (float*) malloc(grid_size * sizeof(float));
    float *V_new  = (float*) malloc(grid_size * sizeof(float));

    // initialize U and V
    initUV(U, V, n);

    // gray scott function, where everything happens basically
    float milliseconds = gray_scott(U, V, U_new, V_new, n, steps, dt, du, dv, f, k);

    // optionally visualize the end result (V grid)
    save_grayscale_image("V_end.png", V, n);

    // free resources
    free(U);
    free(V);
    free(U_new);
    free(V_new);

    // print out the time
    printf("Kernel Execution time is: %0.3f seconds \n", milliseconds / 1000.0f);

    return 0;
}