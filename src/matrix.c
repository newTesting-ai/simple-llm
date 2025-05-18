#include "matrix.h"

void matmul(const float* A, const float* B, float *C, int n, int m, int c) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            C[i*m + j] = 0.0f;
            for(int k = 0; k < c; k++) {
                C[i*m + j] += A[i*c + k] * B[k*m + j];
            }
        }
    }
}

void transpose(const float* A, float *B, int n, int m) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            B[i*n + j] = A[i*m + j];
        }
    }
}