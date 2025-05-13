#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>

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

__global__ void gray_scott_kernel(float* U, float* V, float* U_new, float* V_new,
                                  int n, int local_rows, float dt, float du, float dv, float f, float k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip ghost rows (assumes 1 ghost row at top and bottom)
    if (i < local_rows && j < n) {
        int up    = (i - 1 + local_rows) % local_rows;
        int down  = (i + 1) % local_rows;
        int left  = (j - 1 + n) % n;
        int right = (j + 1) % n;

        float center_u = U[i * n + j];
        float center_v = V[i * n + j];

        float laplacian_u = U[up * n + j] + U[down * n + j] +
                            U[i * n + left] + U[i * n + right] -
                            4.0f * center_u;

        float laplacian_v = V[up * n + j] + V[down * n + j] +
                            V[i * n + left] + V[i * n + right] -
                            4.0f * center_v;

        float UV_square = center_u * center_v * center_v;

        U_new[i * n + j] = center_u + dt * (-UV_square + f * (1.0f - center_u) + du * laplacian_u);
        V_new[i * n + j] = center_v + dt * ( UV_square - (f + k) * center_v + dv * laplacian_v);
    }
}


float gray_scott_mpi(float* U, float* V, int n, int steps, 
                    float dt, float du, float dv, float f, float k) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Device setup
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(rank % device_count);

    const int half_rows = n / 2; // 128 for n=256
    const int local_rows = half_rows + 2; // 130 for n=256 (2 ghost rows)
    const size_t single_row_size = n * sizeof(float);
    const size_t local_size = n * local_rows * sizeof(float);

    // Allocate device memory
    float *d_U, *d_V, *d_U_new, *d_V_new;
    cudaMalloc(&d_U, local_size);
    cudaMalloc(&d_V, local_size);
    cudaMalloc(&d_U_new, local_size);
    cudaMalloc(&d_V_new, local_size);

    // Initialize device memory with proper ghost rows
    if (rank == 0) {
        // Rank 0 gets rows [0..half_rows-1] plus:
        // - Ghost row 0: last row of global grid (periodic boundary)
        // - Ghost row half_rows+1: first row of rank 1's data
        cudaMemcpy(d_U + n, U, half_rows * single_row_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V + n, V, half_rows * single_row_size, cudaMemcpyHostToDevice);
        // Set top ghost row (global periodic boundary)
        cudaMemcpy(d_U, U + (n-1)*n, single_row_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V + (n-1)*n, single_row_size, cudaMemcpyHostToDevice);
    } else {
        // Rank 1 gets rows [half_rows..n-1] plus:
        // - Ghost row 0: last row of rank 0's data
        // - Ghost row half_rows+1: first row of global grid (periodic boundary)
        cudaMemcpy(d_U + n, U + half_rows*n, half_rows * single_row_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V + n, V + half_rows*n, half_rows * single_row_size, cudaMemcpyHostToDevice);
        // Set bottom ghost row (global periodic boundary)
        cudaMemcpy(d_U + (half_rows+1)*n, U, single_row_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V + (half_rows+1)*n, V, single_row_size, cudaMemcpyHostToDevice);
    }

    // Allocate boundary exchange buffers
    float *send_mid = (float*)malloc(2 * n * sizeof(float));
    float *recv_mid = (float*)malloc(2 * n * sizeof(float));
    float *send_global = (float*)malloc(2 * n * sizeof(float));
    float *recv_global = (float*)malloc(2 * n * sizeof(float));

    // Kernel configuration
    dim3 block(32, 32);
    dim3 grid((n + block.x - 1)/block.x, (local_rows + block.y - 1)/block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int step = 0; step < steps; step++) {
        gray_scott_kernel<<<grid, block>>>(d_U, d_V, d_U_new, d_V_new, 
                                         n, local_rows, dt, du, dv, f, k);
        cudaDeviceSynchronize();

        // Prepare boundary data
        if (rank == 0) {
            // Mid-boundary: send last real row to rank 1
            cudaMemcpy(send_mid, d_U_new + half_rows*n, single_row_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(send_mid + n, d_V_new + half_rows*n, single_row_size, cudaMemcpyDeviceToHost);
            // Global boundary: send first real row to rank 1's bottom ghost
            cudaMemcpy(send_global, d_U_new + n, single_row_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(send_global + n, d_V_new + n, single_row_size, cudaMemcpyDeviceToHost);
        } else {
            // Mid-boundary: send first real row to rank 0
            cudaMemcpy(send_mid, d_U_new + n, single_row_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(send_mid + n, d_V_new + n, single_row_size, cudaMemcpyDeviceToHost);
            // Global boundary: send last real row to rank 0's top ghost
            cudaMemcpy(send_global, d_U_new + half_rows*n, single_row_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(send_global + n, d_V_new + half_rows*n, single_row_size, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();

        // Exchange boundaries (use different tags)
        MPI_Sendrecv(send_mid, 2*n, MPI_FLOAT, 1-rank, 100,
                    recv_mid, 2*n, MPI_FLOAT, 1-rank, 100,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(send_global, 2*n, MPI_FLOAT, 1-rank, 200,
                    recv_global, 2*n, MPI_FLOAT, 1-rank, 200,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Apply received boundaries
        if (rank == 0) {
            // Mid-boundary goes to bottom ghost row
            cudaMemcpy(d_U_new + (half_rows+1)*n, recv_mid, single_row_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_V_new + (half_rows+1)*n, recv_mid + n, single_row_size, cudaMemcpyHostToDevice);
            // Global boundary goes to top ghost row
            cudaMemcpy(d_U_new, recv_global, single_row_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_V_new, recv_global + n, single_row_size, cudaMemcpyHostToDevice);
        } else {
            // Mid-boundary goes to top ghost row
            cudaMemcpy(d_U_new, recv_mid, single_row_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_V_new, recv_mid + n, single_row_size, cudaMemcpyHostToDevice);
            // Global boundary goes to bottom ghost row
            cudaMemcpy(d_U_new + (half_rows+1)*n, recv_global, single_row_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_V_new + (half_rows+1)*n, recv_global + n, single_row_size, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();

        // Swap pointers
        float* tmp = d_U; d_U = d_U_new; d_U_new = tmp;
        tmp = d_V; d_V = d_V_new; d_V_new = tmp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copy back results (excluding ghost rows)
    if (rank == 0) {
        cudaMemcpy(U, d_U + n, half_rows * single_row_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(V, d_V + n, half_rows * single_row_size, cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(U + half_rows*n, d_U + n, half_rows * single_row_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(V + half_rows*n, d_V + n, half_rows * single_row_size, cudaMemcpyDeviceToHost);
    }

    // Gather final results on rank 0
    if (rank == 1) {
        MPI_Send(U + half_rows*n, half_rows * n, MPI_FLOAT, 0, 300, MPI_COMM_WORLD);
        MPI_Send(V + half_rows*n, half_rows * n, MPI_FLOAT, 0, 300, MPI_COMM_WORLD);
    } else if (rank == 0) {
        MPI_Recv(U + half_rows*n, half_rows * n, MPI_FLOAT, 1, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(V + half_rows*n, half_rows * n, MPI_FLOAT, 1, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Cleanup
    cudaFree(d_U); cudaFree(d_V); cudaFree(d_U_new); cudaFree(d_V_new);
    free(send_mid); free(recv_mid); free(send_global); free(recv_global);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return milliseconds;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 8) {
        if (rank == 0) {
            printf("USAGE: mpirun -np 2 ./grayscott_mpi width steps time_step diff_u diff_v feed_factor kill_rate\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    int steps = atoi(argv[2]);
    float dt = atof(argv[3]);
    float du = atof(argv[4]);
    float dv = atof(argv[5]);
    float f = atof(argv[6]);
    float k = atof(argv[7]);

    float *U = NULL, *V = NULL;
    if (rank == 0) {
        U = (float*)malloc(n * n * sizeof(float));
        V = (float*)malloc(n * n * sizeof(float));
        initUV(U, V, n);
    } else {
        U = (float*)malloc(n * n * sizeof(float));
        V = (float*)malloc(n * n * sizeof(float));
        initUV(U, V, n);
    }

    float milliseconds = gray_scott_mpi(U, V, n, steps, dt, du, dv, f, k);

    if (rank == 0) {
        save_grayscale_image("V_end.png", V, n);
        printf("Execution time: %.3f seconds\n", milliseconds / 1000.0f);
    }

    free(U);
    free(V);
    MPI_Finalize();
    return 0;
}