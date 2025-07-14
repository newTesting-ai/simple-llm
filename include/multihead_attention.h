#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include<stdio.h>

void init_multihead_attention(int ndim, int sequence_length, int nheads);

void multihead_attention(const float* embeddings, float *output);

#endif