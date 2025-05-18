#ifndef EMBEDDING_H
#define EMBEDDING_H

void init_embeddings(int vocab_size, int ndim);

const float* get_embeddings(int token_id);

#endif