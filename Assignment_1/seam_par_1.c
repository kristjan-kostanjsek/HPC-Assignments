// gcc -o seam_seq seam_seq.c -lm -fopenmp -O2

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

// Function to clamp pixel positions
int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

void compute_energy(unsigned char *image_in, int width, int height, int cpp, float **energy_map) {
    // Precompute Sobel kernels
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // PARALLELIZATION: Collapse the y and x loops, unfold the Sobel kernel loop, remove redundant monotonic sqrt and cpp scaling
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gx_sum = 0, gy_sum = 0;

            // Parallelize and reduce over channels
            //#pragma omp parallel for reduction(+:gx_sum, gy_sum) // breaks the code, probably cuz of the nested parallel for
            for (int c = 0; c < cpp; c++) {
                float Gx_total = 0, Gy_total = 0;

                // Apply Sobel operator
                #pragma unroll
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int nx = x + j;
                        int ny = y + i;

                        // Handle edge cases by skipping out-of-bounds pixels
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                            continue;
                        }

                        int index = (ny * width + nx) * cpp + c;
                        Gx_total += image_in[index] * Gx[i + 1][j + 1];
                        Gy_total += image_in[index] * Gy[i + 1][j + 1];
                    }
                }
 
                gx_sum += Gx_total * Gx_total;
                gy_sum += Gy_total * Gy_total;
            }

            // Compute final energy (division by cpp moved outside the loop)
            //energy_map[y][x] = sqrt(gx_sum + gy_sum) / cpp;
            energy_map[y][x] = gx_sum + gy_sum; 
            // sqrt is monotonic and cpp is a constant used just for scaling, if
            // we are searching for minimums, we can skip them as they don't affect the result
        }
    }
}

void cummulative_energy(int width, int height, float **energy_map) {
    // PARALLELIZATION 2 (can be improved with triangles)
    for (int y = 1; y < height; y++) {
        //#pragma omp parallel for
        for (int x = 0; x < width; x++) {
            // Compute the minimum of the three neighboring values
            //float previous_min = energy_map[y - 1][x]; // Directly above
            float tmp_min = energy_map[y - 1][x];
            float up_left = energy_map[y - 1][clamp(x - 1, 0, width - 1)];
            float up_right = energy_map[y - 1][clamp(x + 1, 0, width - 1)];
            tmp_min = (tmp_min < up_left) ? tmp_min : up_left;
            tmp_min = (tmp_min < up_right) ? tmp_min : up_right;
            energy_map[y][x] += tmp_min;
        }
        //#pragma omp barrier // synchronize threads
    }
}

void imp_cummulative_energy(int width, int height, float **energy_map) {
    const int T_H = 8; // Triangle height (adjust as needed)
    const int T_W = T_H * 2; // Triangle width (adjust as needed)

    for (int s = 0; s <= height / T_H; s++) { // horizontal stripes
        #pragma omp parallel for // parallelize the triangles in the stripe
        for (int i = 0; i <= width / T_W; i++) { // i is the triangle in the stripe
            for (int j = 0; j < T_H; j++) { // j is the row in the triangle
                int y = s * T_H + j+1;
                if (y >= height) continue; // Skip if y is out of bounds

                int x_1 = T_W * i + j;
                if (x_1 < 0) x_1 = 0;
                int x_2 = T_W * i + T_W - j;
                if (x_2 > width) x_2 = width;

                for (int x = x_1; x < x_2; x++) { // x_1 and x_2 are the bounds of the triangle on the current row
                    //if (x < 0 || x >= width) continue; // Skip if x is out of bounds

                    float tmp_min = energy_map[y - 1][x];
                    float up_left = energy_map[y - 1][clamp(x - 1, 0, width - 1)];
                    float up_right = energy_map[y - 1][clamp(x + 1, 0, width - 1)];
                    tmp_min = (tmp_min < up_left) ? tmp_min : up_left;
                    tmp_min = (tmp_min < up_right) ? tmp_min : up_right;
                    energy_map[y][x] += tmp_min;
                }
            }
        }
        #pragma omp barrier // synchronize threads in the stripe - down triangles

        #pragma omp parallel for // parallelize the triangles in the stripe
        for (int i = 0; i <= width / T_W; i++) { // i is the triangle in the stripe
            for (int j = 1; j < T_H; j++) { // j is the row in the triangle
                int y = s * T_H + j+1;
                if (y >= height) continue; // Skip if y is out of bounds

                int x_1 = T_W * i - j;
                if (x_1 < 0) x_1 = 0;
                int x_2 = T_W * i + j;
                if (x_2 > width) x_2 = width;

                for (int x = x_1; x < x_2; x++) { // x_1 and x_2 are the bounds of the triangle on the current row
                    float tmp_min = energy_map[y - 1][x];
                    float up_left = energy_map[y - 1][clamp(x - 1, 0, width - 1)];
                    float up_right = energy_map[y - 1][clamp(x + 1, 0, width - 1)];
                    tmp_min = (tmp_min < up_left) ? tmp_min : up_left;
                    tmp_min = (tmp_min < up_right) ? tmp_min : up_right;
                    energy_map[y][x] += tmp_min;
                }
            }
        }
        #pragma omp barrier // synchronize threads in the stripe - fill spots between triangles
    }
}

bool* imp_find_cheapest_path(int width, int height, float **energy_map, int num_threads) {
    int num_strips = num_threads; // Number of strips = number of threads
    int strip_width = width / num_strips;   // Width of each strip

    bool *seam_mask = (bool *)calloc(width * height, sizeof(bool));

    //#pragma omp parallel for // parallelize the strips
    for (int s = 0; s < num_strips; s++) {
        int strip_start = s * strip_width;
        int strip_end = (s == num_strips - 1) ? width : (s + 1) * strip_width;

        // Find the minimum value in the last row of the strip
        int min_x = strip_start;
        float min_value = energy_map[height - 1][strip_start];
        for (int x = strip_start + 1; x < strip_end; x++) {
            if (energy_map[height - 1][x] < min_value) {
                min_value = energy_map[height - 1][x];
                min_x = x;
            }
        }

        // Store the positions in the strip's seam array
        seam_mask[width * (height - 1) + min_x] = true;
        for (int y = height - 2; y >= 0; y--) {
            int best_x = min_x; // priority above left
            if (min_x > strip_start && energy_map[y][min_x - 1] < energy_map[y][best_x]) // Directly above
                best_x = min_x - 1;
            if (min_x < strip_end - 1 && energy_map[y][min_x + 1] < energy_map[y][best_x]) // Top right
                best_x = min_x + 1;
            //printf("strip: %d x: %d y: %d\n", s, best_x, y);
            seam_mask[width * y + best_x] = true;
            min_x = best_x;
        }
    }

    return seam_mask;
}

void imp_remove_seam_copy(unsigned char *image_in, unsigned char *image_out, int width, int height, int cpp, bool *seam_mask, int num_threads) {
    // Copy the input image to the output image, skipping the marked pixels
    //#pragma omp parallel for // parallel for each row
    int num_strips = num_threads; // Number of strips = number of threads
    int strip_width = width / num_strips;   // Width of each strip

    //#pragma omp parallel for // parallelize the strips
    for (int s = 0; s < num_strips; s++) {
        int strip_start = s * strip_width;
        int strip_end = (s == num_strips - 1) ? width : (s + 1) * strip_width;
        for (int y = 0; y < height; y++) {
            int removed = 0;
            for (int x = strip_start; x < strip_end; x++) {
                if (seam_mask[y * width + x]) {
                    //printf("strip: %d x: %d y: %d value: %d\n", s, x, y, seam_mask[y * width + x]);
                    //in_index += cpp; // Skip the seam pixel
                    removed++;
                    continue;
                }
                //#pragma unfold
                for (int c = 0; c < cpp; c++) {
                    int index = (y * width + x) * cpp + c;
                    //printf("index: %d ", index);
                    //printf("index_out: %d\n", (index-(cpp*y)));
                    image_out[index-(cpp*(y+removed+s))] = image_in[index];
                }
            }
        }
    }
    // Free the remove_mask
    free(seam_mask);
}

void visualize(float **energy_map, int width, int height, const char *output_filename) {
    // Find the minimum and maximum values in the energy map
    float min_energy = energy_map[0][0];
    float max_energy = energy_map[0][0];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (energy_map[y][x] < min_energy) min_energy = energy_map[y][x];
            if (energy_map[y][x] > max_energy) max_energy = energy_map[y][x];
        }
    }

    // Normalize the energy map to the 0-255 range
    unsigned char *gray_image = (unsigned char *)malloc(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float normalized_value = (energy_map[y][x] - min_energy) / (max_energy - min_energy);
            gray_image[y * width + x] = (unsigned char)(normalized_value * 255.0f);
        }
    }

    // Save the grayscale image
    if (!stbi_write_png(output_filename, width, height, 1, gray_image, width)) {
        printf("Error saving energy map image\n");
    }

    // Free memory
    free(gray_image);
}

int main(int argc, char *argv[]) {
    // Too few arguments
    if (argc < 4) {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];
    int num_pixels_to_remove;

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);
    num_pixels_to_remove = atoi(argv[3]); // Number of pixels to remove

    // Validate the number of pixels to remove
    if (num_pixels_to_remove <= 0) {
        printf("Error: num_pixels_to_remove must be a positive integer.\n");
        exit(EXIT_FAILURE);
    }

    // Load image from file
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    int num_threads = 1;//omp_get_max_threads(); // Get the number of available threads
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    printf("Number of pixels to remove: %d\n", num_pixels_to_remove);

    // Allocate memory for the 2D energy map
    float *energy_data = (float *)malloc(height * width * sizeof(float));
    float **energy_map = (float **)malloc(height * sizeof(float *));
   
    for (int i = 0; i < height; i++) {
        energy_map[i] = &energy_data[i * width];  // Point each row pointer to the correct offset
    }

    // Allocate space for the output image
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    const int REPETITIONS = num_pixels_to_remove / num_threads; // Number of seams removed from the image

    // Starting time of the algorithm
    double start_time = omp_get_wtime();

    // SEAM CARVING ALGORITHM
    for (int rep = 0; rep < REPETITIONS; rep++) {
        //printf("rep: %d\n", rep);
        // 1. COMPUTING ENERGY FOR EVERY PIXEL

        // Compute energy map
        compute_energy(image_in, width - rep, height, cpp, energy_map);

        // 2. IDENTIFYING THE VERTICAL SEAM

        // Calculate the cummulative energy for the energy map
        cummulative_energy(width - rep, height, energy_map);
        //imp_cummulative_energy(width - rep, height, energy_map);

        // Find the cheapest path in the cummulative energy map and store it in the seam array
        bool* seam_mask = imp_find_cheapest_path(width - rep, height, energy_map, num_threads);

        // 3. REMOVE SEAM FROM IMAGE (Copy the image to new image, without the seam)
        imp_remove_seam_copy(image_in, image_out, width - rep, height, cpp, seam_mask, num_threads);

        if (rep != REPETITIONS - 1) {
            // output image becomes new input image and vice versa (swap image_in and image_out)
            unsigned char *temp = image_out;
            image_out = image_in;
            image_in = temp;
        }
    }

    //print_energy_map(energy_map, height, width - REPETITIONS);

    // End timing the computation
    double end_time = omp_get_wtime();

    // Print out the ammount of time needed
    printf("Time needed: %f s\n",end_time-start_time);

    // test debug image
    visualize(energy_map, width - REPETITIONS*num_threads, height, "energy_map.png");

    // Save the image
    if (!stbi_write_png(image_out_name, width - REPETITIONS*num_threads, height, cpp, image_out, (width - REPETITIONS) * cpp)) {
        printf("Error saving image to %s\n", image_out_name);
        exit(EXIT_FAILURE);
    }

    // Release the memory
    free(image_in);
    free(image_out);
    free(energy_map);

    return 0;
}