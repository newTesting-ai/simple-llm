#include "tokenizer.h"
#include "embedding.h"
#include "attention.h"
#include "multihead_attention.h"
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

    float *output = (float *) malloc(sequence_len * NDIM * sizeof(float));


    // init_multihead_attention(NDIM, sequence_len, 5);
    // multihead_attention(embeddings, output);

    float *Q = (float *) malloc(sequence_len * NDIM * sizeof(float));
    float *K = (float *) malloc(sequence_len * NDIM * sizeof(float));
    float *V = (float *) malloc(sequence_len * NDIM * sizeof(float));
    compute_qkv(embeddings, Q, K, V);
    scaled_dot_product_attention(Q, K, V, output);
    
    for(int i = 0; i < sequence_len; i++) {
        for(int j = 0; j < NDIM; j++) {
            printf("%f ", output[i*NDIM + j]);
        }
        printf("\n");
    }
    int sentence_embedding[10] = {0};
    for (int i = 0; i < 6 /* tokens */; i++) {
        for (int j = 0; j < 10; j++) {
            sentence_embedding[j] += output[i * 10 + j];
        }
    }
    for (int j = 0; j < 10; j++) {
        sentence_embedding[j] /= 6;
    }

    
    printf("\n");
    char decode[256];
    detokenize(&tokenizer, sentence_embedding, total_token, decode);
    printf("Decoded Tokens are: %s\n", decode);
}


    