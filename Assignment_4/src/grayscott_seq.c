// gcc grayscott_seq.c -o grayscott_seq -lm -fopenmp -O2
// ./grayscott_seq 256 5000 1 0.16 0.08 0.060 0.062

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

// calculate laplacian for U or V
float calc_laplacian(float* data, int n, int i, int j) {
    // Wrap-around indices using modulus
    int up    = (i - 1 + n) % n;
    int down  = (i + 1) % n;
    int left  = (j - 1 + n) % n;
    int right = (j + 1) % n;

    float center = data[i * n + j];
    return data[up * n + j] + data[down * n + j] + data[i * n + left] + data[i * n + right] - 4.0f * center;
}

// here is where the magic happens, baby
void gray_scott(float* U, float* V, float* U_new, float* V_new, int n, int steps, float dt, float du, float dv, float f, float k) {
    for (int cur_step = 0; cur_step < steps; cur_step ++) {
        // for each grid cell do, what must be done...
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // calculate laplacians for U and V
                float laplacian_u = calc_laplacian(U, n, i, j);
                float laplacian_v = calc_laplacian(V, n, i, j);
                // store center values for U and V, so we don't have to look them up 30 times
                float center_u = U[i * n + j];
                float center_v = V[i * n + j];
                // calculate UV square
                float UV_square = center_u * center_v * center_v;
                // calculate new values and assign them to U_new and V_new
                U_new[i * n + j] = center_u + dt * (- UV_square + f * (1 - center_u) + du * laplacian_u);
                V_new[i * n + j] = center_v + dt * (UV_square - (f + k) * center_v + dv * laplacian_v);
            }
        }
        // switch U and U_new, V and V_new
        float* tmp;

        tmp   = U;
        U     = U_new;
        U_new = tmp;

        tmp   = V;
        V     = V_new;
        V_new = tmp;
    }
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
    double start_time = omp_get_wtime();
    gray_scott(U, V, U_new, V_new, n, steps, dt, du, dv, f, k);
    double end_time = omp_get_wtime();

    // optionally visualize the end result (V grid)
    save_grayscale_image("V_end.png", V, n);

    // free resources
    free(U);
    free(V);
    free(U_new);
    free(V_new);

    printf("Elapsed time: %f seconds\n", end_time - start_time);
    return 0;
}