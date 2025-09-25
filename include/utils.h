#ifndef UTILS_H
#define UTILS_H

void init_uniform_array(double *arr, int size, double a, double b);

// matrix-vector multiplication to result
void vec_mat_mul(double **mat, const double *vec, double *result,
                 int vec_n, int mat_n, int mat_m);

void elementwise_add(const double *vec1, const double *vec2, double *result, int n);

double ReLU(double x);
double ReLU_d(double x);

#endif // UTILS_H