// gcc histeq_seq.c -o histeq_seq -lm -fopenmp -O2
// ./histeq_seq in_images/720x480.png out_images/test.png

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void print_histogram(int *hist) { // Just for testing, can be deleted
    printf("Bin  | Count\n");
    printf("-----|-------\n");
    for (int l = 0; l < 256; l++) {
        printf("%3d  | %d\n", l, hist[l]);
    }
}

void RGB_to_YUV_compute_luminance_histogram(unsigned char *image, int width, int height, int cpp, int *lum_hist, int offset) {
    for (int i = 0; i < width * height; i++) {
        // convert RGB to YUV
        unsigned char R = image[i * cpp + 0];
        unsigned char G = image[i * cpp + 1];
        unsigned char B = image[i * cpp + 2];

        // formula from the instructions + clamping values to valid range (0-255)
        image[i * cpp + 0] = fmax(0.0f, fmin(255.0f, 0.299f * R + 0.587f * G + 0.114f * B));
        image[i * cpp + 1] = fmax(0.0f, fmin(255.0f, -0.14713f * R - 0.28886f * G + 0.436f * B + 128.0f));
        image[i * cpp + 2] = fmax(0.0f, fmin(255.0f, 0.615f * R - 0.51499f * G - 0.10001f * B + 128.0f));
        
        // compute luminance histogram
        int l = (int)roundf(image[i * cpp + offset]); 
        lum_hist[l]++;
    }
}

int compute_cumulative_histogram(int *lum_hist, int *cum_hist) {
    int min_luminance = 0;
    cum_hist[0] = lum_hist[0];
    for (int i = 1; i < 256; i++) {
        cum_hist[i] = cum_hist[i - 1] + lum_hist[i];

        // Find the minimum non-zero luminance value in the histogram
        if (cum_hist[i-1] > 0 && min_luminance == 0) {
            min_luminance = cum_hist[i-1];
        }
    }

    return min_luminance; // return the minimum luminance value

}

void compute_new_luminance(int *cum_hist, int *new_lum, int width, int height, int min_luminance) {
    // Compute the new luminance values based on the cumulative histogram
    for (int lum = 0; lum < 256; lum++) {
        int cum_lum = cum_hist[lum];
        int up = cum_lum - min_luminance;
        int down = (width * height) - min_luminance;
        float division = (float)up/down; // the (float) is necessary to avoid integer division
        int new_luminance = floorf(division * 255);
        new_lum[lum] = new_luminance;
    }
}

void assign_new_lum_convert_YUV_to_RGB(unsigned char *image, int width, int height, int cpp, int *new_lum) {
    for (int i = 0; i < width * height; i++) {
        int l = (int)roundf(image[i * cpp]); // luminance value of the pixel currently
        image[i * cpp] = new_lum[l]; // use old luminance value as index to get new luminance value

        // convert YUV back to RGB
        unsigned char Y = image[i * cpp + 0];
        unsigned char U = image[i * cpp + 1];
        unsigned char V = image[i * cpp + 2];

        // formula from the instructions + clamping values to valid range (0-255)
        image[i * cpp + 0] = fmax(0.0f, fmin(255.0f, Y + 1.402f * (V - 128.0f)));
        image[i * cpp + 1] = fmax(0.0f, fmin(255.0f, Y - 0.344136f * (U - 128.0f) - 0.714136f * (V - 128.0f)));
        image[i * cpp + 2] = fmax(0.0f, fmin(255.0f, Y + 1.772f * (U - 128.0f)));
    }
}

void util_print_histogram(int *hist) { // Just for testing, can be deleted
    printf("Bin  | Count\n");
    printf("-----|-------\n");
    for (int l = 0; l < 256; l++) {
        printf("%3d  | %d\n", l, hist[l]);
    }
}

void util_print_hist_for_plot(int *hist) { // Just for testing, can be deleted
    printf("[");
    for (int l = 0; l < 256; l++) {
        printf("%d, ", hist[l]);
    }
    printf("]\n\n");
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("USAGE: histeq_seq input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image = stbi_load(image_in_name, &width, &height, &cpp, 0);

    if (image == NULL) {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);

    // (c)allocate space for luminance histogram, cumulative histogram and array for new luminance values
    int *lum_hist = (int *)calloc(256, sizeof(int));
    int *cum_hist = (int *)calloc(256, sizeof(int));
    int *new_lum = (int *)calloc(256, sizeof(int));

    int *R_hist = (int *)calloc(256, sizeof(int));
    int *new_R_hist = (int *)calloc(256, sizeof(int));
    int min_lum = 0;

    // print old R histogram for plotting
    //compute_luminance_histogram(image, width, height, cpp, R_hist, 0); // to change to G or B, change offset to 1 or 2
    //util_print_hist_for_plot(R_hist); // Just for testing, can be deleted

    // start timer
    double start_time = omp_get_wtime();

    // 1. and 2. convert RGB to YUV and compute luminance histogram
    RGB_to_YUV_compute_luminance_histogram(image, width, height, cpp, lum_hist, 0);
    
    // 3. compute cumulative histogram
    min_lum = compute_cumulative_histogram(lum_hist, cum_hist);
    //util_print_histogram(cum_hist); // Just for testing, can be deleted

    // 4. calculate new pixel luminances
    compute_new_luminance(cum_hist, new_lum, width, height, min_lum);

    // 5. and 6. assign new luminances to each pixel and convert YUV to RGB
    assign_new_lum_convert_YUV_to_RGB(image, width, height, cpp, new_lum);

    // stop timer
    double end_time = omp_get_wtime();

    // print new R histogram for plotting
    //compute_luminance_histogram(image, width, height, cpp, new_R_hist, 0);
    //util_print_hist_for_plot(new_R_hist); // Just for testing, can be deleted

    // Save the image
    if (!stbi_write_png(image_out_name, width, height, cpp, image, width * cpp)) {
        printf("Error saving image to %s\n", image_out_name);
        exit(EXIT_FAILURE);
    }

    // Release the memory
    free(image);
    free(lum_hist);
    free(cum_hist);

    // Print out the ammount of time needed
    printf("Time needed: %f s\n",end_time-start_time);

    return 0;
}
