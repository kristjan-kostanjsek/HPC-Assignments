#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

// just for testing (saving the final image)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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


// initialize U and V grids (2D) with a square in the middle
// TODO: add offset parameter to allow for different initial conditions
void initUV(float *U, float *V, int n, int local_rows, int rank, int size) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            U[i * n + j] = 1.0f;
            V[i * n + j] = 0.0f;
        }
    }

    int r = n / 8;
    int real_local_rows = n / size;
    int boundary_rows = (local_rows - real_local_rows)/2;
    int global_row_offset = rank * real_local_rows;

    for (int i = 0; i < real_local_rows; i++) {
        int global_i = i + global_row_offset;
        if (global_i >= n/2 - r && global_i < n/2 + r) {
            for (int j = n/2 - r; j < n/2 + r; j++) {
                U[(i + boundary_rows) * n + j] = 0.75f;
                V[(i + boundary_rows) * n + j] = 0.25f;
            }
        }
    }
}

// calculate laplacian for U or V
float calc_laplacian(float* data, int n, int local_rows, int i, int j) {
    // Wrap-around indices using modulus
    int up    = i - 1;
    int down  = i + 1;
    int left  = (j - 1 + n) % n;
    int right = (j + 1) % n;

    float center = data[i * n + j];
    return data[up * n + j] + data[down * n + j] + data[i * n + left] + data[i * n + right] - 4.0f * center;
}

// here is where the magic happens, baby
void gray_scott(float* U, float* V, float* U_new, float* V_new,
                int n, int local_rows, int n_boundary_rows, int steps, float dt, float du, float dv, float f, float k,
                int rank, int size, MPI_Comm comm) {

    float *recvbuf_up_U = (float*) malloc(n_boundary_rows * n * sizeof(float));
    float *recvbuf_down_U = (float*) malloc(n_boundary_rows * n * sizeof(float));
    float *recvbuf_up_V = (float*) malloc(n_boundary_rows * n * sizeof(float));
    float *recvbuf_down_V = (float*) malloc(n_boundary_rows * n * sizeof(float));

    int up_rank = (rank - 1 + size) % size;
    int down_rank = (rank + 1) % size;

    // main simulation loop
    for (int step = 0; step < steps; step++) {
        int remainder = step % n_boundary_rows; // 
        // Exchange halos every n_boundary_rows steps
        if (remainder == 0) {
            // Exchange U halos
            MPI_Sendrecv(U + (n * n_boundary_rows), n*n_boundary_rows, MPI_FLOAT, up_rank, 0,
                        recvbuf_down_U, n*n_boundary_rows, MPI_FLOAT, down_rank, 0, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(U + (local_rows-(n_boundary_rows*2)) * n, n*n_boundary_rows, MPI_FLOAT, down_rank, 1,
                        recvbuf_up_U, n*n_boundary_rows, MPI_FLOAT, up_rank, 1, comm, MPI_STATUS_IGNORE);

            // Exchange V halos
            MPI_Sendrecv(V + (n * n_boundary_rows), n*n_boundary_rows, MPI_FLOAT, up_rank, 2,
                        recvbuf_down_V, n*n_boundary_rows, MPI_FLOAT, down_rank, 2, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(V + (local_rows-(n_boundary_rows*2)) * n, n*n_boundary_rows, MPI_FLOAT, down_rank, 3,
                        recvbuf_up_V, n*n_boundary_rows, MPI_FLOAT, up_rank, 3, comm, MPI_STATUS_IGNORE);
            
            // Copy all received boundary rows
            for (int r = 0; r < n_boundary_rows; r++) {
                // Copy upper boundaries (from up_rank to our top rows)
                memcpy(U + r * n, recvbuf_up_U + r * n, n * sizeof(float));
                memcpy(V + r * n, recvbuf_up_V + r * n, n * sizeof(float));
                
                // Copy lower boundaries (from down_rank to our bottom rows)
                memcpy(U + (local_rows - n_boundary_rows + r) * n, 
                      recvbuf_down_U + r * n, n * sizeof(float));
                memcpy(V + (local_rows - n_boundary_rows + r) * n,
                      recvbuf_down_V + r * n, n * sizeof(float));
            }
        }

        // Compute simulation step
        for (int i = 1+remainder; i < local_rows-(1+remainder); i++) {
            for (int j = 0; j < n; j++) {
                float lap_u = calc_laplacian(U, n, local_rows, i, j);
                float lap_v = calc_laplacian(V, n, local_rows, i, j);
                float u = U[i*n + j];
                float v = V[i*n + j];
                float uvv = u * v * v;

                U_new[i*n + j] = u + dt * (-uvv + f*(1-u) + du * lap_u);
                V_new[i*n + j] = v + dt * ( uvv - (f + k) * v + dv * lap_v);
            }
        }

        // Swap U and U_new, V and V_new
        float *tmp;
        tmp = U; U = U_new; U_new = tmp;
        tmp = V; V = V_new; V_new = tmp;
    }
    free(recvbuf_up_U); 
    free(recvbuf_down_U);
    free(recvbuf_up_V);
    free(recvbuf_down_V);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //printf("Rank %d of %d processes\n", rank, size);

    if (argc < 8) {
        if (rank == 0)
            printf("USAGE: grayscott_mpi width steps dt du dv f k\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = atoi(argv[1]); // grid width
    int steps = atoi(argv[2]); // number of steps of the algorithm
    float dt = atof(argv[3]); // time step size
    float du = atof(argv[4]); // diffusion rate for u
    float dv = atof(argv[5]); // diffusion rate for v
    float f = atof(argv[6]); // feed rate
    float k = atof(argv[7]); // kill rate

    if (n % size != 0) {
        if (rank == 0)
            printf("Grid size must be divisible by number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // allocate memory for the grids
    int n_boundary_rows = 5;
    int local_rows = n/size + n_boundary_rows * 2; // +2 for ghost rows

    float *U     = (float*) malloc(local_rows * n * sizeof(float));
    float *V     = (float*) malloc(local_rows * n * sizeof(float));
    float *U_new = (float*) malloc(local_rows * n * sizeof(float));
    float *V_new = (float*) malloc(local_rows * n * sizeof(float));

    // initialize U and V
    initUV(U, V, n, local_rows, rank, size);

    // gray scott function, where everything happens basically
    double t0 = MPI_Wtime();
    gray_scott(U, V, U_new, V_new, n, local_rows, n_boundary_rows, steps, dt, du, dv, f, k, rank, size, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Elapsed time: %f seconds\n", t1 - t0);

    // Gather V grids to rank 0
    float* V_global = NULL;
    if (rank == 0) V_global = (float*) malloc(n*n*sizeof(float));

    MPI_Gather(V + n_boundary_rows * n, (local_rows-(n_boundary_rows*2))*n, MPI_FLOAT, V_global, (local_rows-(n_boundary_rows*2))*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // optionally visualize the end result (V grid)
    if (rank == 0) {
        save_grayscale_image("V_extended_rows_end_mpi.png", V_global, n);
        free(V_global);
    }

    // free resources
    free(U); free(V); free(U_new); free(V_new);
    MPI_Finalize();
    return 0;
}