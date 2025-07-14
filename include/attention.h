#ifndef ATTENTION_H
#define ATTENTION_H


void init_attention(int ndim, int sequence_length);


// Computes Q, K, V matrices from the input embeddings
// Embeddings should be of size [sequence_length x ndim]
// Q, K, V should be pre-allocated to [sequence_length x ndim]
void compute_qkv(const float* embeddings, float *Q, float *K, float *V);

void softmax(float* attention_score);

void init_weight(int ndim, int seq_len, float *WQ, float *WK, float *WV);

void scaled_dot_product_attention(const float* Q, const float *K, const float *V, float *output);

#endif