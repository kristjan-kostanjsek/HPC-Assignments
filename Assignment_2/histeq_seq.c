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

void RGB_to_YUV(unsigned char *image, int width, int height, int cpp) {
    for (int i = 0; i < width * height; i++) {
        unsigned char R = image[i * cpp + 0];
        unsigned char G = image[i * cpp + 1];
        unsigned char B = image[i * cpp + 2];

        // formula from the instructions + clamping values to valid range (0-255)
        image[i * cpp + 0] = fmax(0.0f, fmin(255.0f, 0.299f * R + 0.587f * G + 0.114f * B));
        image[i * cpp + 1] = fmax(0.0f, fmin(255.0f, -0.14713f * R - 0.28886f * G + 0.436f * B + 128.0f));
        image[i * cpp + 2] = fmax(0.0f, fmin(255.0f, 0.615f * R - 0.51499f * G - 0.10001f * B + 128.0f));
    }
}

void YUV_to_RGB(unsigned char *image, int width, int height, int cpp) {
    for (int i = 0; i < width * height; i++) {
        unsigned char Y = image[i * cpp + 0];
        unsigned char U = image[i * cpp + 1];
        unsigned char V = image[i * cpp + 2];

        // formula from the instructions + clamping values to valid range (0-255)
        image[i * cpp + 0] = fmax(0.0f, fmin(255.0f, Y + 1.402f * (V - 128.0f)));
        image[i * cpp + 1] = fmax(0.0f, fmin(255.0f, Y - 0.344136f * (U - 128.0f) - 0.714136f * (V - 128.0f)));
        image[i * cpp + 2] = fmax(0.0f, fmin(255.0f, Y + 1.772f * (U - 128.0f)));
    }
}

void compute_luminance_histogram(unsigned char *image, int width, int height, int cpp, int *lum_hist) {
    for (int i = 0; i < width * height; i++) {
        int l = (int)roundf(image[i * cpp]);
        lum_hist[l]++;
    }
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

    // (c)allocate space for luminance histogram, cumulative histogram
    int *lum_hist = (int *)calloc(256, sizeof(int));
    int *cum_hist = (int *)calloc(256, sizeof(int));
    
    // start timer
    double start_time = omp_get_wtime();

    // 1. convert to YUV
    RGB_to_YUV(image, width, height, cpp);

    // 2. compute luminance histogram
    compute_luminance_histogram(image, width, height, cpp, lum_hist);

    // 3. compute cumulative histogram
    // TODO

    // 4. calculate new pixel luminances
    // TODO

    // 5. assign new luminances to each pixel
    // TODO

    // 6. convert image back to RGB
    YUV_to_RGB(image, width, height, cpp);

    // stop timer
    double end_time = omp_get_wtime();

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
