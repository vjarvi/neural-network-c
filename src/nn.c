#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "nn.h"
#include "utils.h"

// ...existing code...
NeuralNetwork* init_network(int num_layers, int *layer_dims,
                            const char activation[], const char activation_output[]) {
    // range for rand initialization of weights
    static const double INIT_LOWER_BOUND = -0.1;
    static const double INIT_UPPER_BOUND = 0.1;

    NeuralNetwork *nn = malloc(sizeof *nn);
    nn->num_layers = num_layers;
    nn->layers = calloc(num_layers - 1, sizeof *nn->layers);

    for (int i = 0; i < num_layers - 1; i++) {
        Linear *curr = &nn->layers[i];
        curr->in_dim = layer_dims[i];
        curr->out_dim = layer_dims[i+1];

        curr->weights = calloc(curr->out_dim, sizeof *curr->weights);
        for (int j = 0; j < curr->out_dim; j++) {
            curr->weights[j] = calloc(curr->in_dim, sizeof *curr->weights[j]);
            init_uniform_array(curr->weights[j], curr->in_dim, INIT_LOWER_BOUND, INIT_UPPER_BOUND);
        }
        curr->bias = calloc(curr->out_dim, sizeof *curr->bias);
        curr->z = malloc(curr->out_dim * sizeof(double));
        curr->a = malloc(curr->out_dim * sizeof(double));
    }

    if (strcmp(activation, "relu") == 0) {
        nn->activation = ReLU;
        nn->activation_d = ReLU_d;
    } else {
        nn->activation = NULL;
        nn->activation_d = NULL;
    }
    if (strcmp(activation_output, "relu") == 0) {
        nn->activation_output = ReLU;
        nn->activation_output_d = ReLU_d;
    } else {
        nn->activation_output = NULL;
        nn->activation_output_d = NULL;
    }
    return nn;
}

void free_network(NeuralNetwork *nn) {
    if (!nn) return;
    for (int i = 0; i < nn->num_layers - 1; ++i) {
        Linear *layer = &nn->layers[i];
        if (layer->weights) {
            for (int j = 0; j < layer->out_dim; ++j) free(layer->weights[j]);
            free(layer->weights);
        }
        free(layer->bias);
        free(layer->z);
        free(layer->a);
    }
    free(nn->layers);
    free(nn);
}

void forward(NeuralNetwork *nn, const double input[], double output[]) {
    const double *current_input = input;
    for (int i = 0; i < nn->num_layers - 1; ++i) {
        Linear *layer = &nn->layers[i];
        vec_mat_mul(layer->weights, current_input, layer->z, layer->in_dim, layer->out_dim, layer->in_dim);
        elementwise_add(layer->z, layer->bias, layer->z, layer->out_dim);

        if (i == nn->num_layers - 2 && nn->activation_output) {
            for (int j = 0; j < layer->out_dim; ++j) layer->a[j] = nn->activation_output(layer->z[j]);
        } else if (nn->activation) {
            for (int j = 0; j < layer->out_dim; ++j) layer->a[j] = nn->activation(layer->z[j]);
        } else {
            memcpy(layer->a, layer->z, layer->out_dim * sizeof(double));
        }
        current_input = layer->a;
    }

    Linear *last = &nn->layers[nn->num_layers - 2];
    if (output) memcpy(output, last->a, last->out_dim * sizeof(double));
}