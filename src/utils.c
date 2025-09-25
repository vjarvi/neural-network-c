#include <stdlib.h>
#include "utils.h"

void init_uniform_array(double *arr, int size, double a, double b) {
    for (int i = 0; i < size; ++i) {
        arr[i] = a + (b - a) * ((double)rand() / (double)RAND_MAX);
    }
}

void vec_mat_mul(double **mat, const double *vec, double *result,
                 int vec_n, int mat_n, int mat_m) {
    (void)vec_n;
    for (int i = 0; i < mat_n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < mat_m; ++j) {
            sum += mat[i][j] * vec[j];
        }
        result[i] = sum;
    }
}

void elementwise_add(const double *vec1, const double *vec2, double *result, int n) {
    for (int i = 0; i < n; ++i) result[i] = vec1[i] + vec2[i];
}

double ReLU(double x) { return x > 0.0 ? x : 0.0; }
double ReLU_d(double x) { return x > 0.0 ? 1.0 : 0.0; }