#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <math.h>

// Function declarations
void initialize(unsigned int M, unsigned int N);
void routine1(float alpha, float beta, unsigned int M);
void routine2(float alpha, float beta, unsigned int N);
void routine1_vec(float alpha, float beta, unsigned int M);
void routine2_vec(float alpha, float beta, unsigned int N);
void check_correctness_routine1(float alpha, float beta, unsigned int M);
void check_correctness_routine2(float alpha, float beta, unsigned int N);

float* y;
float* z;
float* y_ref;
double* x;
double* w;
double* w_ref;
double** A;

int main(int argc, char* argv[]) {

    // Default sizes
    unsigned int M = 1024 * 512;
    unsigned int N = 8192;

    // Accept input sizes if provided
    if (argc == 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    // Allocate memory with alignment
    y = (float*)_aligned_malloc(M * sizeof(float), 64);
    z = (float*)_aligned_malloc(M * sizeof(float), 64);
    y_ref = (float*)_aligned_malloc(M * sizeof(float), 64);

    x = (double*)_aligned_malloc(N * sizeof(double), 64);
    w = (double*)_aligned_malloc(N * sizeof(double), 64);
    w_ref = (double*)_aligned_malloc(N * sizeof(double), 64);

    A = (double**)_aligned_malloc(N * sizeof(double*), 64);
    for (unsigned int i = 0; i < N; i++) {
        A[i] = (double*)_aligned_malloc(N * sizeof(double), 64);
    }

    initialize(M, N);

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); // Start timer

    for (t = 0; t < 1; t++)
        routine1_vec(alpha, beta, M);

    run_time = omp_get_wtime() - start_time; // End timer
    printf("\n Time elapsed is %f secs \n", run_time);

    // Check correctness of routine1
    check_correctness_routine1(alpha, beta, M);

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); // Start timer

    for (t = 0; t < 1; t++)
        routine2_vec(alpha, beta, N);

    run_time = omp_get_wtime() - start_time; // End timer
    printf("\n Time elapsed is %f secs \n", run_time);

    // Check correctness of routine2
    check_correctness_routine2(alpha, beta, N);

    // Clean up
    _aligned_free(y);
    _aligned_free(z);
    _aligned_free(y_ref);
    _aligned_free(x);
    _aligned_free(w);
    _aligned_free(w_ref);
    for (unsigned int i = 0; i < N; i++) {
        _aligned_free(A[i]);
    }
    _aligned_free(A);

    return 0;
}

void initialize(unsigned int M, unsigned int N) {

    unsigned int i, j;

    // Initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013;
        }

    // Initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01;
        w[i] = (i % 5) - 0.002;
        w_ref[i] = w[i];
    }

    // Initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
        y_ref[i] = y[i];
    }
}

void routine1(float alpha, float beta, unsigned int M) {

    unsigned int i;

    for (i = 0; i < M; i++)
        y_ref[i] = y_ref[i] - alpha + beta - z[i];

}

void routine2(float alpha, float beta, unsigned int N) {

    unsigned int i, j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w_ref[i] += beta * x[j] + alpha * A[i][j] * x[j];

}

void routine1_vec(float alpha, float beta, unsigned int M) {

    unsigned int i;
    __m128 vec_alpha = _mm_set1_ps(alpha);
    __m128 vec_beta = _mm_set1_ps(beta);

    for (i = 0; i < M; i += 4) {
        __m128 vec_y = _mm_loadu_ps(&y[i]);
        __m128 vec_z = _mm_loadu_ps(&z[i]);
        __m128 result = _mm_sub_ps(_mm_add_ps(vec_y, vec_beta), _mm_add_ps(vec_alpha, vec_z));
        _mm_storeu_ps(&y[i], result);
    }
}

void routine2_vec(float alpha, float beta, unsigned int N) {

    unsigned int i, j;
    __m256d vec_alpha = _mm256_set1_pd(alpha);
    __m256d vec_beta = _mm256_set1_pd(beta);

    for (i = 0; i < N; i++) {
        double w_sum = 0.0;

        for (j = 0; j < N; j += 4) {
            __m256d vec_x = _mm256_loadu_pd(&x[j]);
            __m256d vec_A = _mm256_loadu_pd(&A[i][j]);
            __m256d mul1 = _mm256_mul_pd(vec_beta, vec_x);
            __m256d mul2 = _mm256_mul_pd(vec_alpha, _mm256_mul_pd(vec_A, vec_x));
            __m256d vec_result = _mm256_add_pd(mul1, mul2);

            // Sum up elements in vec_result
            w_sum += vec_result.m256d_f64[0] + vec_result.m256d_f64[1] + vec_result.m256d_f64[2] + vec_result.m256d_f64[3];
        }

        w[i] += w_sum;
    }
}

void check_correctness_routine1(float alpha, float beta, unsigned int M) {

    routine1(alpha, beta, M);

    for (unsigned int i = 0; i < M; i++) {
        if (fabs(y[i] - y_ref[i]) > 1e-6) {
            printf("\nRoutine1_vec failed at index %d: y_vec=%f, y_ref=%f\n", i, y[i], y_ref[i]);
            return;
        }
    }

    printf("\nRoutine1_vec passed the correctness test!\n");
}

void check_correctness_routine2(float alpha, float beta, unsigned int N) {

    routine2(alpha, beta, N);

    for (unsigned int i = 0; i < N; i++) {
        if (fabs(w[i] - w_ref[i]) > 1e-6) {
            printf("\nRoutine2_vec failed at index %d: w_vec=%f, w_ref=%f\n", i, w[i], w_ref[i]);
            return;
        }
    }

    printf("\nRoutine2_vec passed the correctness test!\n");
}
