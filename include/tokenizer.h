#ifndef TOKENIZER_H
#define TOKENIZER_H

#include<stddef.h>
#include<string.h>
#define MAX_TOKENS 1024
#define MAX_VOCAB_SIZE 1000
#define MAX_TOKEN_LENGTH 32

typedef struct {
    char token[MAX_TOKEN_LENGTH];
    int id;
} Token;

typedef struct {
    Token vocab[MAX_VOCAB_SIZE];
    int vocab_size;
} Tokenizer;


void init_tokenizer(Tokenizer *tokenizer);

int tokenize(Tokenizer *tokenizer, const char *text, int *tokens, int max_tokens);

void detokenize(Tokenizer *tokenizer, const int* token, int num_tokens, char *output);


#endif