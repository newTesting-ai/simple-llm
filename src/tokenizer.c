#include "tokenizer.h"
#include<stdio.h>
#include<string.h>
void init_tokenizer(Tokenizer *tokenizer) {
    tokenizer->vocab_size = 6;
    strcpy(tokenizer->vocab[0].token, "[PAD]");
    tokenizer->vocab[0].id = 0;

    strcpy(tokenizer->vocab[1].token, "[UNK]");
    tokenizer->vocab[1].id = 1;

    strcpy(tokenizer->vocab[2].token, "मैं");
    tokenizer->vocab[2].id = 2;

    strcpy(tokenizer->vocab[3].token, "रहने");
    tokenizer->vocab[3].id = 3;
    
    strcpy(tokenizer->vocab[4].token, "वाली");
    tokenizer->vocab[4].id = 4;
    
    strcpy(tokenizer->vocab[5].token, "हूँ।");
    tokenizer->vocab[5].id = 5;
    printf("Tokenizer Initalized\n");
}

int tokenize(Tokenizer *tokenizer, const char* text, int *tokens, int max_tokens) {
    int token_count = 0;

    char *token = strtok((char *) text, " ");
    while(token != NULL && token_count < max_tokens) {
        int found = 0;
        for(int i = 0; i < tokenizer->vocab_size; i++) {
            if(strcmp(tokenizer->vocab[i].token, token) == 0) {
                tokens[token_count++] = tokenizer->vocab[i].id;
                found = 1;
                break;
            }
        }
        if(!found) {
            tokens[token_count++] = 1;
        }
        token = strtok(NULL, " ");
    }
    return token_count;
}

void detokenize(Tokenizer *tokenizer, const int* token, int num_tokens, char *output) {
    for(int i = 0; i < num_tokens; i++) {
        for(int j = 0; j < tokenizer->vocab_size; j++) {
            if(tokenizer->vocab[j].id == token[i]) {
                strcat(output, tokenizer->vocab[j].token);
                if(i < num_tokens -1) strcat(output, " ");
                break;
            }
        }
    }
}