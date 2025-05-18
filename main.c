#include "tokenizer.h"
#include "embedding.h"
#include "attention.h"
#include <stdio.h>
#include <stdlib.h>

#define VOCAB_SIZE 6
#define NDIM 10
#define sequence_len 6

int main() {
    char text[] = "मैं केरल की रहने वाली हूँ।";
    Tokenizer tokenizer;
    init_tokenizer(&tokenizer);
    init_embeddings(VOCAB_SIZE, NDIM);
    init_attention(NDIM, sequence_len);

    int tokens[7];
    int total_token = tokenize(&tokenizer, text, tokens, 7);
    float *embeddings = (float *)malloc(total_token * NDIM * sizeof(float));
    
    
    printf("total %d tokens generated\n", total_token);
    
    for(int i = 0; i < total_token; i++) {
        printf("%d\n", tokens[i]);
        const float *embedding = get_embeddings(tokens[i]);
        for(int j = 0; j < NDIM; j++) {
            embeddings[i*NDIM + j] = embedding[j];
        }
    }

    float *Q = (float *) malloc(sequence_len * NDIM * sizeof(float));
    float *K = (float *) malloc(sequence_len * NDIM * sizeof(float));
    float *V = (float *) malloc(sequence_len * NDIM * sizeof(float));
    float *output = (float *) malloc(sequence_len * NDIM * sizeof(float));

    compute_qkv(embeddings, Q, K, V);
    scaled_dot_product_attention(Q, K, V, output);
    
    
    for(int i = 0; i < sequence_len; i++) {
        for(int j = 0; j < NDIM; j++) {
            printf("%f ", output[i*NDIM + j]);
        }
        printf("\n");
    }
    
    printf("\n");
    char decode[256];
    detokenize(&tokenizer, tokens, total_token, decode);
    printf("Decoded Tokens are: %s\n", decode);
}