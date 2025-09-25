#ifndef NN_H
#define NN_H

#include "utils.h"

typedef struct {
    int in_dim;
    int out_dim;
    double **weights; // 2D array for all connections in this layer
    double *bias;
    double *z; // pre-activation for backprop
    double *a; // post-activation
} Linear;

typedef struct {
    int num_layers;
    Linear *layers;
    double (*activation)(double);
    double (*activation_d)(double);
    double (*activation_output)(double);
    double (*activation_output_d)(double);
} NeuralNetwork;

NeuralNetwork* init_network(int num_layers, int *layer_dims,
                            const char activation[], const char activation_output[]);

void free_network(NeuralNetwork *nn);

void forward(NeuralNetwork *nn, const double input[], double output[]);


#endif // NN_H