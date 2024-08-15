#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <dirent.h>

// Function declarations
void Gaussian_Blur(int M, int N);
void Sobel(int M, int N);
int initialize_kernel();
void read_image(const char* filename, int M, int N);
void write_image2(const char* filename, unsigned char* output_image, int M, int N);
void openfile(const char* filename, FILE** finput);
int getint(FILE* fp);

// Dynamic arrays for image processing
unsigned char* frame1 = NULL; // Input image
unsigned char* filt = NULL; // Output filtered image
unsigned char* gradient = NULL; // Output image

const signed char Mask[5][5] = {
    {2,4,5,4,2} ,
    {4,9,12,9,4},
    {5,12,15,12,5},
    {4,9,12,9,4},
    {2,4,5,4,2}
};

const signed char GxMask[3][3] = {
    {-1,0,1} ,
    {-2,0,2},
    {-1,0,1}
};

const signed char GyMask[3][3] = {
    {-1,-2,-1} ,
    {0,0,0},
    {1,2,1}
};

char header[100];

int main() {
    DIR *d;
    struct dirent *dir;

    d = opendir("input_images");
    if (!d) {
        fprintf(stderr, "Could not open input_images directory\n");
        return 1;
    }

    while ((dir = readdir(d)) != NULL) {
        // Check if it's a regular file and has a .pgm extension
        if (dir->d_type == DT_REG && strstr(dir->d_name, ".pgm")) {
            char input_image_path[1024];
            snprintf(input_image_path, sizeof(input_image_path), "input_images/%s", dir->d_name);

            int M, N;
            // Open file to determine the dimensions (M and N)
            FILE* finput = fopen(input_image_path, "rb");
            if (!finput) {
                fprintf(stderr, "Could not open file: %s\n", input_image_path);
                continue;
            }

            fscanf(finput, "%s", header);
            M = getint(finput); // Get width (M)
            N = getint(finput); // Get height (N)
            fclose(finput);

            // Allocate memory dynamically for the current image size
            frame1 = (unsigned char*)malloc(N * M);
            filt = (unsigned char*)malloc(N * M);
            gradient = (unsigned char*)malloc(N * M);

            if (!frame1 || !filt || !gradient) {
                fprintf(stderr, "Memory allocation failed\n");
                return 1;
            }

            // Generate output filenames
            char output_blur_path[1024];
            char output_edge_path[1024];
            snprintf(output_blur_path, sizeof(output_blur_path), "output_images/%s_blur.pgm", dir->d_name);
            snprintf(output_edge_path, sizeof(output_edge_path), "output_images/%s_edge.pgm", dir->d_name);

            read_image(input_image_path, M, N); // Read image

            Gaussian_Blur(M, N); // Apply Gaussian Blur (reduce noise)
            Sobel(M, N); // Apply Sobel edge detection

            write_image2(output_blur_path, filt, M, N); // Save blurred image
            write_image2(output_edge_path, gradient, M, N); // Save edge detection image

            // Free dynamically allocated memory
            free(frame1);
            free(filt);
            free(gradient);
        }
    }

    closedir(d);
    return 0;
}

void Gaussian_Blur(int M, int N) {
    int row, col, rowOffset, colOffset;
    int newPixel;
    unsigned char pix;
    const unsigned short int size = 2;

    /*---------------------- Gaussian Blur ---------------------------------*/
    for (row = 0; row < N; row++) {
        for (col = 0; col < M; col++) {
            newPixel = 0;
            for (rowOffset = -size; rowOffset <= size; rowOffset++) {
                for (colOffset = -size; colOffset <= size; colOffset++) {

                    if ((row + rowOffset < 0) || (row + rowOffset >= N) || (col + colOffset < 0) || (col + colOffset >= M))
                        pix = 0;
                    else
                        pix = frame1[M * (row + rowOffset) + col + colOffset];

                    newPixel += pix * Mask[size + rowOffset][size + colOffset];
                }
            }
            filt[M * row + col] = (unsigned char)(newPixel / 159);
        }
    }
}

void Sobel(int M, int N) {
    int row, col, rowOffset, colOffset;
    int Gx, Gy;

    /*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
    for (row = 1; row < N - 1; row++) {
        for (col = 1; col < M - 1; col++) {

            Gx = 0;
            Gy = 0;

            /* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
            for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
                for (colOffset = -1; colOffset <= 1; colOffset++) {

                    Gx += filt[M * (row + rowOffset) + col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
                    Gy += filt[M * (row + rowOffset) + col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
                }
            }

            gradient[M * row + col] = (unsigned char)sqrt(Gx * Gx + Gy * Gy); /* Calculate gradient strength */
        }
    }
}

void read_image(const char* filename, int M, int N) {
    int c;
    FILE* finput;
    int i, j, temp;

    printf("\nReading %s image from disk ...", filename);
    finput = NULL;
    openfile(filename, &finput);

    if ((header[0] == 'P') && (header[1] == '5')) { // If P5 image

        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                temp = getc(finput);
                frame1[M * j + i] = (unsigned char)temp;
            }
        }
    }
    else if ((header[0] == 'P') && (header[1] == '2')) { // If P2 image
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                if (fscanf(finput, "%d", &temp) == EOF)
                    exit(EXIT_FAILURE);

                frame1[M * j + i] = (unsigned char)temp;
            }
        }
    }
    else {
        printf("\nProblem with reading the image");
        exit(EXIT_FAILURE);
    }

    fclose(finput);
    printf("\nImage successfully read from disk\n");
}

void write_image2(const char* filename, unsigned char* output_image, int M, int N) {
    FILE* foutput;
    int i, j;

    printf("  Writing result to disk ...\n");

    foutput = fopen(filename, "wb");
    if (foutput == NULL) {
        fprintf(stderr, "Unable to open file %s for writing\n", filename);
        exit(-1);
    }

    fprintf(foutput, "P2\n");
    fprintf(foutput, "%d %d\n", M, N);
    fprintf(foutput, "%d\n", 255);

    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            fprintf(foutput, "%3d ", output_image[M * j + i]);
            if (i % 32 == 31) fprintf(foutput, "\n");
        }
        if (M % 32 != 0) fprintf(foutput, "\n");
    }
    fclose(foutput);
}

void openfile(const char* filename, FILE** finput) {
    int x0, y0, x;

    *finput = fopen(filename, "rb");
    if (*finput == NULL) {
        fprintf(stderr, "Unable to open file %s for reading\n", filename);
        exit(-1);
    }

    fscanf(*finput, "%s", header);

    x0 = getint(*finput); // This is M (width)
    y0 = getint(*finput); // This is N (height)
    printf("\t Header is %s, while x=%d, y=%d", header, x0, y0);

    x = getint(*finput); /* Read and throw away the range info */
}

int getint(FILE* fp) {
    int c, i, firstchar;

    c = getc(fp);
    while (1) {
        if (c == '#') {
            char cmt[256], *sp;
            sp = cmt;
            firstchar = 1;
            while (1) {
                c = getc(fp);
                if (firstchar && c == ' ') firstchar = 0;
                else {
                    if (c == '\n' || c == EOF) break;
                    if ((sp - cmt) < 250) *sp++ = c;
                }
            }
            *sp++ = '\n';
            *sp = '\0';
        }

        if (c == EOF) return 0;
        if (c >= '0' && c <= '9') break;

        c = getc(fp);
    }

    i = 0;
    while (1) {
        i = (i * 10) + (c - '0');
        c = getc(fp);
        if (c == EOF) return i;
        if (c < '0' || c > '9') break;
    }
    return i;
}
