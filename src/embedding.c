#include "embedding.h"
#include <stdlib.h>

static float *embedding_matrix = NULL;
static int vocab_size = 0;
static int embedding_dim = 0;
void init_embeddings(int vocab_sz, int ndim) {
    vocab_size = vocab_sz;
    embedding_dim = ndim;

    embedding_matrix = (float *) malloc(vocab_size * embedding_dim * sizeof(float));

    for(int i = 0; i < vocab_size * embedding_dim; i++) {
        embedding_matrix[i] = ((float) rand() / RAND_MAX);
    }
}

const float *get_embeddings(int token_id) {
    if(token_id < 0 || token_id >= vocab_size) {
        return NULL;
    }
    return &embedding_matrix[token_id * embedding_dim];
}