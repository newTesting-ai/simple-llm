#ifndef MATRIX_H
#define MATRIX_H

void matmul(const float* A, const float* B, float *C, int n, int m, int c);

void transpose(const float* A, float *B, int n, int m);

#endif