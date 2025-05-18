#include "attention.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>

static float *w_q = NULL;
static float *w_k = NULL;
static float *w_v = NULL;
static int sequence_length = 0;
static int embedding_dim = 0;

void init_attention(int ndim, int seq_len) {
    sequence_length = seq_len;
    embedding_dim = ndim;

    w_q = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));
    w_k = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));
    w_v = (float *) malloc(embedding_dim * embedding_dim * sizeof(float));

    for(int i = 0; i < embedding_dim * embedding_dim; i++) {
        w_q[i] = ((float) rand() / RAND_MAX);
        w_k[i] = ((float) rand() / RAND_MAX);
        w_v[i] = ((float) rand() / RAND_MAX);
    }
}

void compute_qkv(const float* embeddings, float *Q, float *K, float *V) {
    matmul(embeddings, w_q, Q, sequence_length, embedding_dim, embedding_dim);
    matmul(embeddings, w_v, V, sequence_length, embedding_dim, embedding_dim);
    matmul(embeddings, w_k, K, sequence_length, embedding_dim, embedding_dim);
}

void softmax(float* attention_score) {
    for(int i = 0; i < sequence_length; i++) {
        float max_val = attention_score[i*sequence_length];
        for(int j = 1; j < sequence_length; j++) {
            if(attention_score[i*sequence_length + j] > max_val)
                max_val = attention_score[i*sequence_length + j];
        }

        float sum_exp = 0.0f;
        for(int j = 1; j < sequence_length; j++) {
            attention_score[i*sequence_length + j] = exp(max_val - attention_score[i*sequence_length + j]);
            sum_exp += attention_score[i*sequence_length + j];
        }


        for(int j = 1; j < sequence_length; j++) {
            attention_score[i*sequence_length + j] /= sum_exp;
        }
    }
}

void scaled_dot_product_attention(const float* Q, const float *K, const float *V, float *output) {
    float scale_factor = sqrt(embedding_dim);
    float *Kt = (float *) malloc(embedding_dim * sequence_length * sizeof(float));   
    float *attention_score = (float *) malloc(sequence_length * sequence_length * sizeof(float));   

    transpose(K, Kt, sequence_length, embedding_dim);

    matmul(Q, Kt, attention_score, sequence_length, sequence_length, embedding_dim);

    for(int i = 0; i < sequence_length; i++) {
        for(int j = 0; j < sequence_length; j++) {
            attention_score[i*sequence_length + j] /= scale_factor;
        }
    }

    softmax(attention_score);

    matmul(attention_score, V, output, sequence_length, embedding_dim, sequence_length);

}
