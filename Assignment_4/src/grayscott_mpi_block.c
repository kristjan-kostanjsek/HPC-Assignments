#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void save_grayscale_image(const char* filename, float* data, int n) {
    unsigned char* pixels = (unsigned char*)malloc(n * n * sizeof(unsigned char));
    for (int i = 0; i < n * n; i++) {
        pixels[i] = (unsigned char)(data[i] * 255.0f);
    }
    stbi_write_png(filename, n, n, 1, pixels, n);
    free(pixels);
}

void initUV(float *U, float *V, int n, int block_rows, int block_cols, 
            int coords[2], int dims[2], int n_halo) {
    // Initialize entire block to default values
    for (int i = 0; i < block_rows; i++) {
        for (int j = 0; j < block_cols; j++) {
            U[i * block_cols + j] = 1.0f;
            V[i * block_cols + j] = 0.0f;
        }
    }

    // Calculate global coordinates of this block
    int block_size_i = n / dims[0];
    int block_size_j = n / dims[1];
    int global_start_i = coords[0] * block_size_i;
    int global_start_j = coords[1] * block_size_j;

    // Add square in the center (adjusting for halo regions)
    int r = n / 8;
    for (int i = 0; i < block_size_i; i++) {
        int global_i = global_start_i + i;
        if (global_i >= n/2 - r && global_i < n/2 + r) {
            for (int j = 0; j < block_size_j; j++) {
                int global_j = global_start_j + j;
                if (global_j >= n/2 - r && global_j < n/2 + r) {
                    // Adjust for halo offset in local array
                    U[(i + n_halo) * block_cols + (j + n_halo)] = 0.75f;
                    V[(i + n_halo) * block_cols + (j + n_halo)] = 0.25f;
                }
            }
        }
    }
}

float calc_laplacian(float* data, int block_cols, int i, int j) {
    int up = i - 1;
    int down = i + 1;
    int left = j - 1;
    int right = j + 1;

    float center = data[i * block_cols + j];
    return data[up * block_cols + j] + data[down * block_cols + j] + 
           data[i * block_cols + left] + data[i * block_cols + right] - 4.0f * center;
}

void exchange_halos(float *U, float *V, int block_rows, int block_cols, 
                   int n_halo, int neighbors[4], MPI_Datatype col_type, 
                   MPI_Comm grid_comm) {
    MPI_Request reqs[8];
    MPI_Status stats[8];

    // Exchange rows (top/bottom)
    MPI_Isend(U + n_halo * block_cols + n_halo, block_cols - 2*n_halo, MPI_FLOAT, 
              neighbors[0], 0, grid_comm, &reqs[0]); // Send top
    MPI_Irecv(U + 0 * block_cols + n_halo, block_cols - 2*n_halo, MPI_FLOAT, 
              neighbors[0], 1, grid_comm, &reqs[1]); // Recv top

    MPI_Isend(U + (block_rows - n_halo - 1) * block_cols + n_halo, block_cols - 2*n_halo, MPI_FLOAT, 
              neighbors[1], 1, grid_comm, &reqs[2]); // Send bottom
    MPI_Irecv(U + (block_rows - 1) * block_cols + n_halo, block_cols - 2*n_halo, MPI_FLOAT, 
              neighbors[1], 0, grid_comm, &reqs[3]); // Recv bottom

    // Exchange columns (left/right)
    MPI_Isend(U + n_halo * block_cols + n_halo, 1, col_type, 
              neighbors[2], 2, grid_comm, &reqs[4]); // Send left
    MPI_Irecv(U + n_halo * block_cols + 0, 1, col_type, 
              neighbors[2], 3, grid_comm, &reqs[5]); // Recv left

    MPI_Isend(U + n_halo * block_cols + (block_cols - n_halo - 1), 1, col_type, 
              neighbors[3], 3, grid_comm, &reqs[6]); // Send right
    MPI_Irecv(U + n_halo * block_cols + (block_cols - 1), 1, col_type, 
              neighbors[3], 2, grid_comm, &reqs[7]); // Recv right

    MPI_Waitall(8, reqs, stats);
}

void gray_scott(float* U, float* V, float* U_new, float* V_new,
               int n, int block_rows, int block_cols, int n_halo,
               int steps, float dt, float du, float dv, float f, float k,
               int neighbors[4], MPI_Datatype col_type, MPI_Comm grid_comm) {

    for (int step = 0; step < steps; step++) {
        // Exchange halos every step for accuracy
        exchange_halos(U, V, block_rows, block_cols, n_halo, neighbors, col_type, grid_comm);
        exchange_halos(V, U, block_rows, block_cols, n_halo, neighbors, col_type, grid_comm);

        // Compute interior points
        for (int i = n_halo; i < block_rows - n_halo; i++) {
            for (int j = n_halo; j < block_cols - n_halo; j++) {
                float lap_u = calc_laplacian(U, block_cols, i, j);
                float lap_v = calc_laplacian(V, block_cols, i, j);
                float u = U[i * block_cols + j];
                float v = V[i * block_cols + j];
                float uvv = u * v * v;

                U_new[i * block_cols + j] = u + dt * (-uvv + f*(1-u) + du * lap_u);
                V_new[i * block_cols + j] = v + dt * (uvv - (f + k) * v + dv * lap_v);
            }
        }

        // Swap pointers
        float *tmp = U; U = U_new; U_new = tmp;
        tmp = V; V = V_new; V_new = tmp;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 8) {
        if (rank == 0)
            printf("USAGE: %s width steps dt du dv f k\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = atoi(argv[1]);
    int steps = atoi(argv[2]);
    float dt = atof(argv[3]);
    float du = atof(argv[4]);
    float dv = atof(argv[5]);
    float f = atof(argv[6]);
    float k = atof(argv[7]);

    // Create 2D Cartesian grid
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {1, 1}; // Periodic boundary conditions
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    // Get neighbor ranks
    enum {UP, DOWN, LEFT, RIGHT};
    int neighbors[4];
    MPI_Cart_shift(grid_comm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(grid_comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);

    // Block dimensions (with halo)
    int n_halo = 1;
    int block_size_i = n / dims[0];
    int block_size_j = n / dims[1];
    int block_rows = block_size_i + 2 * n_halo;
    int block_cols = block_size_j + 2 * n_halo;

    // Create column datatype for halo exchange
    MPI_Datatype col_type;
    MPI_Type_vector(block_size_i, 1, block_cols, MPI_FLOAT, &col_type);
    MPI_Type_commit(&col_type);

    // Allocate memory
    float *U = (float*)malloc(block_rows * block_cols * sizeof(float));
    float *V = (float*)malloc(block_rows * block_cols * sizeof(float));
    float *U_new = (float*)malloc(block_rows * block_cols * sizeof(float));
    float *V_new = (float*)malloc(block_rows * block_cols * sizeof(float));

    // Initialize
    initUV(U, V, n, block_rows, block_cols, coords, dims, n_halo);

    // Run simulation
    double t0 = MPI_Wtime();
    gray_scott(U, V, U_new, V_new, n, block_rows, block_cols, n_halo,
               steps, dt, du, dv, f, k, neighbors, col_type, grid_comm);
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Elapsed time: %f seconds\n", t1 - t0);

    // Gather results to rank 0
    float* V_global = NULL;
    if (rank == 0) V_global = (float*)malloc(n * n * sizeof(float));

    // Create subarray type for sending
    MPI_Datatype subarray;
    int sizes[2] = {block_rows, block_cols};
    int subsizes[2] = {block_size_i, block_size_j};
    int starts[2] = {n_halo, n_halo};
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
    MPI_Type_commit(&subarray);

    // On root process, create a type to receive into the correct position in global array
    MPI_Datatype recv_type;
    if (rank == 0) {
        int global_sizes[2] = {n, n};
        int global_subsizes[2] = {block_size_i, block_size_j};
        int global_starts[2] = {coords[0] * block_size_i, coords[1] * block_size_j};
        MPI_Type_create_subarray(2, global_sizes, global_subsizes, global_starts, 
                                MPI_ORDER_C, MPI_FLOAT, &recv_type);
        MPI_Type_commit(&recv_type);
    }

    // Gather blocks - root uses recv_type, others use subarray
    if (rank == 0) {
        // First put root's own data
        for (int i = 0; i < block_size_i; i++) {
            for (int j = 0; j < block_size_j; j++) {
                V_global[(coords[0] * block_size_i + i) * n + (coords[1] * block_size_j + j)] = 
                    V[(i + n_halo) * block_cols + (j + n_halo)];
            }
        }
        
        // Then receive from others
        for (int src = 1; src < size; src++) {
            int src_coords[2];
            MPI_Cart_coords(grid_comm, src, 2, src_coords);
            
            int recv_sizes[2] = {n, n};
            int recv_subsizes[2] = {block_size_i, block_size_j};
            int recv_starts[2] = {src_coords[0] * block_size_i, src_coords[1] * block_size_j};
            MPI_Datatype recv_subarray;
            MPI_Type_create_subarray(2, recv_sizes, recv_subsizes, recv_starts, 
                                    MPI_ORDER_C, MPI_FLOAT, &recv_subarray);
            MPI_Type_commit(&recv_subarray);
            
            MPI_Recv(V_global, 1, recv_subarray, src, 0, grid_comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&recv_subarray);
        }
    } else {
        MPI_Send(V + n_halo * block_cols + n_halo, 1, subarray, 0, 0, grid_comm);
    }

    if (rank == 0) MPI_Type_free(&recv_type);

    // Save image
    if (rank == 0) {
        save_grayscale_image("V_block_end_mpi.png", V_global, n);
        free(V_global);
    }

    // Cleanup
    MPI_Type_free(&col_type);
    MPI_Type_free(&subarray);
    free(U); free(V); free(U_new); free(V_new);
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}