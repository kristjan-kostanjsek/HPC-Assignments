#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

// Sobel operator kernels
int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

// Function to clamp pixel positions
int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Compute energy using Sobel operator
void compute_energy(unsigned char *image_in, int width, int height, int cpp, float **energy_map) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gx_sum = 0, gy_sum = 0;

            // Compute gradients for each channel separately
            for (int c = 0; c < cpp; c++) {
                float Gx_total = 0, Gy_total = 0;

                // Apply Sobel operator
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int nx = clamp(x + j, 0, width - 1);
                        int ny = clamp(y + i, 0, height - 1);
                        int index = (ny * width + nx) * cpp + c;

                        Gx_total += image_in[index] * Gx[i + 1][j + 1];
                        Gy_total += image_in[index] * Gy[i + 1][j + 1];
                    }
                }

                gx_sum += Gx_total * Gx_total;
                gy_sum += Gy_total * Gy_total;
            }

            // Compute final energy as the average of all channels
            energy_map[y][x] = sqrt(gx_sum + gy_sum) / cpp;
        }
    }
}

void cummulative_energy(int width, int height, float **energy_map) {
    for (int y = 1; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float previous_min = energy_map[y - 1][0]; // Directly above
            if (x > 0)
                fmin(previous_min, energy_map[y - 1][x - 1]);  // Top-left
            if (x < width - 1)
                fmin(previous_min, energy_map[y - 1][x + 1]);  // Top-right
            energy_map[y][x] += previous_min;
        }
    }
}

void find_cheapest_path(int width, int height, float **energy_map) {
    // Find the min value in the last row of energy_map
    int min_x = 0;
    float min_value = energy_map[height-1][0];
    for (int x = 1; x < width; x++) {
        if (energy_map[height-1][x] < min_value) {
            min_value = energy_map[height-1][x];
            min_x = x;
        }
    }
    // Mark the seam with -1 values
    energy_map[height-1][min_x] = -1;
    for (int y = height - 2; y >= 0; y--) {
        int best_x = min_x; // Directly above
        if (min_x > 0 && energy_map[y][min_x - 1] < energy_map[y][best_x]) // Top left
            best_x = min_x - 1;
        if (min_x < width - 1 && energy_map[y][min_x + 1] < energy_map[y][best_x]) // Top right
            best_x = min_x + 1;
        energy_map[y][best_x] = -1;
    }
}

// copies the input image to output image without the seam
void remove_seam_copy(unsigned char *image_in, unsigned char *image_out, int width, int height, int cpp, float **energy_map) {
    int in_index = 0;
    int out_index = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (energy_map[y][x] == -1) {
                in_index += cpp;
                continue; // Skip the seam pixel
            }
            for (int c = 0; c < cpp; c++) {
                image_out[out_index] = image_in[in_index];
                in_index ++;
                out_index ++;
            }
        }
    }
}

int main(int argc, char *argv[]) {

    // Too few arguments
    if (argc < 3) {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);

    // 1. COMPUTING ENERGY FOR EVERY PIXEL

    // Allocate memory for the 2D energy map
    float **energy_map = (float **)malloc(height * sizeof(float *));
    for (int i = 0; i < height; i++) {
        energy_map[i] = (float *)malloc(width * sizeof(float));
    }

    // Compute energy map
    compute_energy(image_in, width, height, cpp, energy_map);

    // 2. IDENTIFYING THE VERTICAL SEAM

    // Calculate the cummulative energy for the energy map
    cummulative_energy(width, height, energy_map);

    // Find the cheapest path in the cummulative energy map and mark it with -1s
    find_cheapest_path(width, height, energy_map);

    // 3. REMOVE SEAM FROM IMAGE (Copy the image to new image, without the seam)
    
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(datasize);

    remove_seam_copy(image_in, image_out, width, height, cpp, energy_map);

    // Save the image
    if (!stbi_write_png(image_out_name, width - 1, height, cpp, image_out, (width - 1) * cpp)) {
        printf("Error saving image to %s\n", image_out_name);
        exit(EXIT_FAILURE);
    }

    // Release the memory
    free(image_in);
    free(image_out);
    free(energy_map);

    return 0;
}