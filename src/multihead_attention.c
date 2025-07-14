#include "attention.h"
#include "multihead_attention.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#define MAX_HEADS 10

struct multihead_weights {
    float *w_q;
    float *w_k;
    float *w_v;
};

static struct multihead_weights *weights[MAX_HEADS];
static int sequence_length = 0;
static int embedding_dim = 0;
static int heads = 0;

void init_multihead_attention(int ndim, int seq_len, int nheads) {
    if (ndim % nheads != 0) {
        printf("Error: Embedding dimension not divisible by number of heads.\n");
        return;
    }
    sequence_length = seq_len;
    embedding_dim = ndim/nheads;
    heads = nheads;
    for(int j = 0; j < nheads; j++) {
        weights[j] = (struct multihead_weights *) malloc(sizeof(struct multihead_weights));

        float *w_q = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));
        float *w_k = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));
        float *w_v = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));
    
        for(int i = 0; i < embedding_dim * embedding_dim; i++) {
            w_q[i] = ((float) rand() / RAND_MAX);
            w_k[i] = ((float) rand() / RAND_MAX);
            w_v[i] = ((float) rand() / RAND_MAX);
        }
        weights[j]->w_k = w_k;
        weights[j]->w_q = w_q;
        weights[j]->w_v = w_v;
    }
}

void multihead_attention(const float* embeddings, float *output) {
    for(int i = 0; i < heads; i++) {
        float *w_k = weights[i]->w_k;
        float *w_q = weights[i]->w_q;
        float *w_v = weights[i]->w_v;
        init_weight(embedding_dim, sequence_length, w_q, w_k, w_v);
        float *Q = (float *) malloc(sequence_length * embedding_dim * sizeof(float));
        float *K = (float *) malloc(sequence_length * embedding_dim * sizeof(float));
        float *V = (float *) malloc(sequence_length * embedding_dim * sizeof(float));
        float *head_out = (float *) malloc(sequence_length * embedding_dim * sizeof(float));
        compute_qkv(embeddings, Q, K, V);
        scaled_dot_product_attention(Q, K, V, head_out);

        for(int j = 0; j < sequence_length; j++) {
            for(int k = 0; k < embedding_dim; k++) {
                output[j * (embedding_dim * heads) + i * embedding_dim + k] =
                head_out[j * embedding_dim + k];
            }
        }

        free(Q), free(K); free(V); free(head_out);
        
    }
}