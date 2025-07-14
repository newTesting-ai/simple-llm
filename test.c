#include "tokenizer.h"
#include "matrix.h"
#include "math.h"
#include "embedding.h"
#include "attention.h"
#include "multihead_attention.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define VOCAB_SIZE 6
#define NDIM 10
#define sequence_len 6

void test_matmul() {
    float A[6] = {1, 2, 3, 4, 5, 6};     // 2×3
    float B[6] = {7, 8, 9, 10, 11, 12};  // 3×2
    float C[4];                          // 2×2 result

    matmul(A, B, C, 2, 2, 3);  // A[2x3] × B[3x2] = C[2x2]

    assert(fabs(C[0] - 58) < 0.001);   // 1*7 + 2*9 + 3*11
    assert(fabs(C[1] - 64) < 0.001);   // 1*8 + 2*10 + 3*12
    assert(fabs(C[2] - 139) < 0.001);
    assert(fabs(C[3] - 154) < 0.001);

    printf("✅ test_matmul passed\n");
}
void test_transpose() {
    float A[6] = {1, 2, 3, 4, 5, 6};  // 2×3
    float T[6];

    transpose(A, T, 2, 3);  // A[2x3] → T[3x2]

    assert(T[0] == 1);
    assert(T[1] == 4);
    assert(T[2] == 2);
    assert(T[3] == 5);
    assert(T[4] == 3);
    assert(T[5] == 6);

    printf("✅ test_transpose passed\n");
}


void test_softmax() {
    int sequence_length = 3;
    float scores[9] = {1, 2, 3, 4, 5, 6, 1, 1, 1}; // 3×3

    init_attention(3, 3);

    softmax(scores);  // Softmax per row

    for (int i = 0; i < 3; i++) {
        float sum = 0;
        for (int j = 0; j < 3; j++) {
            sum += scores[i * 3 + j];
        }
        assert(fabs(sum - 1.0f) < 0.001);
    }

    printf("✅ test_softmax passed\n");
}

void test_compute_qkv() {
    int sequence_length = 2;
    int embedding_dim = 2;
    float embeddings[4] = {1, 2, 3, 4};  // 2×2

    float wq[4] = {1, 0, 0, 1};
    float wk[4] = {0, 1, 1, 0};
    float wv[4] = {1, 1, 1, 1};
    init_weight(sequence_length, embedding_dim, wq, wk, wv);

    float Q[4], K[4], V[4];
    compute_qkv(embeddings, Q, K, V);

    // Q = embeddings * wq = same as input
    assert(Q[0] == 1 && Q[1] == 2);
    assert(Q[2] == 3 && Q[3] == 4);

    // K = embeddings * wk
    assert(K[0] == 2 && K[1] == 1);
    assert(K[2] == 4 && K[3] == 3);

    // V = embeddings * wv
    assert(V[0] == 3 && V[1] == 3);
    assert(V[2] == 7 && V[3] == 7);

    printf("✅ test_compute_qkv passed\n");
}

void test_attention() {
    int sequence_length = 2;
    int embedding_dim = 2;
    init_attention(sequence_length, embedding_dim);
    
    // Test 1: Identity matrices for Q and K should give reasonable attention weights
    float Q[4] = {1, 0, 0, 1}; // 2x2 identity matrix
    float K[4] = {1, 0, 0, 1}; // 2x2 identity matrix  
    float V[4] = {1, 2, 3, 4}; // 2x2 matrix
    float out[4];
    
    scaled_dot_product_attention(Q, K, V, out);

    // Test 2: When Q and K are designed so first token attends strongly to itself
    float Q2[4] = {10, 0, 0, 1}; // Strong attention for first token
    float K2[4] = {1, 0, 0, 1};
    float V2[4] = {1, 2, 3, 4};
    float out2[4];
    
    scaled_dot_product_attention(Q2, K2, V2, out2);
    
    printf("Test 2 - Strong self-attention: ");
    for(int i = 0; i < 4; i++) {
        printf("%f ", out2[i]);
    }
    printf("\n");
    
    // Assertions for expected behavior:
    // 1. Output should be finite (not NaN or infinity)
    for(int i = 0; i < 4; i++) {
        assert(isfinite(out[i]));
        assert(isfinite(out2[i]));
    }
    
    // 2. With strong self-attention, first row should be closer to first row of V
    // out2[0] should be closer to V[0]=1 than out[0] is
    assert(fabs(out2[0] - 1.0) < fabs(out[0] - 1.0));
    
    // 3. Output values should be reasonable weighted combinations of V
    // (between min and max values of V)
    float v_min = 1.0, v_max = 4.0;
    for(int i = 0; i < 4; i++) {
        assert(out[i] >= v_min - 0.01 && out[i] <= v_max + 0.01);
    }
    
    printf("✅ test_attention passed - attention mechanism working correctly\n");
}

void test_attention_basic() {
    int sequence_length = 2;
    int embedding_dim = 2;
    init_attention(sequence_length, embedding_dim);
    
    float Q[4] = {1, 0, 0, 1}; // 2x2 identity matrix
    float K[4] = {1, 0, 0, 1}; // 2x2 identity matrix  
    float V[4] = {1, 2, 3, 4}; // 2x2 matrix
    float out[4];
    
    scaled_dot_product_attention(Q, K, V, out);
    
    // Test: Output should be finite (not NaN or infinity)
    for(int i = 0; i < 4; i++) {
        assert(isfinite(out[i]));
    }
    
    printf("✅ test_attention_basic passed\n");
}

void test_attention_bounds() {
    int sequence_length = 2;
    int embedding_dim = 2;
    init_attention(sequence_length, embedding_dim);
    
    float Q[4] = {1, 0, 0, 1};
    float K[4] = {1, 0, 0, 1};
    float V[4] = {1, 2, 3, 4};
    float out[4];
    
    scaled_dot_product_attention(Q, K, V, out);
    
    // Test: Output should be reasonable weighted combinations of V values
    float v_min = 1.0, v_max = 4.0;
    for(int i = 0; i < 4; i++) {
        assert(out[i] >= v_min - 0.01 && out[i] <= v_max + 0.01);
    }
    
    printf("✅ test_attention_bounds passed\n");
}

void test_attention_self_attention() {
    int sequence_length = 2;
    int embedding_dim = 2;
    init_attention(sequence_length, embedding_dim);
    
    // Strong self-attention: first token should attend strongly to itself
    float Q[4] = {10, 0, 0, 1};
    float K[4] = {1, 0, 0, 1};
    float V[4] = {1, 2, 3, 4};
    float out[4];
    
    scaled_dot_product_attention(Q, K, V, out);
    
    // Compare with normal attention
    float Q_normal[4] = {1, 0, 0, 1};
    float out_normal[4];
    scaled_dot_product_attention(Q_normal, K, V, out_normal);
    
    for(int i = 0; i < 4; i++) {
        printf("%f ", out[i]);
    }
    printf("\n");

    for(int i = 0; i < 4; i++) {
        printf("%f ", out_normal[i]);
    }
    printf("\n");
    // Test: Strong self-attention should make first row closer to first row of V
    assert(fabs(out[0] - 1.0) < fabs(out_normal[0] - 1.0));
    
    printf("✅ test_attention_self_attention passed\n");
}

void test_attention_symmetric() {
    int sequence_length = 2;
    int embedding_dim = 2;
    init_attention(sequence_length, embedding_dim);
    
    // Symmetric Q and K should produce symmetric-like behavior
    float Q[4] = {1, 0, 0, 1};
    float K[4] = {1, 0, 0, 1};
    float V[4] = {5, 5, 5, 5}; // All same values
    float out[4];
    
    scaled_dot_product_attention(Q, K, V, out);
    
    // Test: With identical V values, output should be close to those values
    for(int i = 0; i < 4; i++) {
        assert(fabs(out[i] - 5.0) < 0.01);
    }
    
    printf("✅ test_attention_symmetric passed\n");
}

void test_multihead_init() {
    int embedding_dim = 8;
    int sequence_length = 4;
    int heads = 2;
    
    init_multihead_attention(embedding_dim, sequence_length, heads);
    
    printf("✅ test_multihead_init passed\n");
}

void test_multihead_init_invalid_dimension() {
    int embedding_dim = 7; // Not divisible by heads
    int sequence_length = 4;
    int heads = 2;
    
    printf("Testing invalid dimension (should print error):\n");
    init_multihead_attention(embedding_dim, sequence_length, heads);
    
    printf("✅ test_multihead_init_invalid_dimension passed\n");
}

void test_multihead_basic_functionality() {
    int total_embedding_dim = 4;
    int sequence_length = 2;
    int heads = 2;
    
    init_multihead_attention(total_embedding_dim, sequence_length, heads);
    
    // Simple input: 2 tokens, 4 dimensions each
    float embeddings[8] = {1, 2, 3, 4,    // token 1
                          5, 6, 7, 8};   // token 2
    
    float output[8]; // 2 tokens * 4 dimensions
    
    multihead_attention(embeddings, output);
    
    // Test: Output should be finite
    for(int i = 0; i < 8; i++) {
        assert(isfinite(output[i]));
    }
    
    printf("Output: ");
    for(int i = 0; i < 8; i++) {
        printf("%.3f ", output[i]);
    }
    printf("\n");
    
    printf("✅ test_multihead_basic_functionality passed\n");
}

void test_multihead_output_dimensions() {
    int total_embedding_dim = 6;
    int sequence_length = 3;
    int heads = 3;
    
    init_multihead_attention(total_embedding_dim, sequence_length, heads);
    
    float embeddings[18]; // 3 tokens * 6 dimensions
    for(int i = 0; i < 18; i++) {
        embeddings[i] = i + 1;
    }
    
    float output[18]; // Should be same size as input
    
    multihead_attention(embeddings, output);
    
    // Test: All outputs should be finite
    for(int i = 0; i < 18; i++) {
        assert(isfinite(output[i]));
    }
    
    printf("✅ test_multihead_output_dimensions passed\n");
}

void test_multihead_single_head() {
    int total_embedding_dim = 4;
    int sequence_length = 2;
    int heads = 1; // Single head should work like regular attention
    
    init_multihead_attention(total_embedding_dim, sequence_length, heads);
    
    float embeddings[8] = {1, 2, 3, 4,
                          5, 6, 7, 8};
    
    float output[8];
    
    multihead_attention(embeddings, output);
    
    // Test: Output should be finite
    for(int i = 0; i < 8; i++) {
        assert(isfinite(output[i]));
    }
    
    printf("✅ test_multihead_single_head passed\n");
}

void test_multihead_different_inputs() {
    int total_embedding_dim = 4;
    int sequence_length = 2;
    int heads = 2;
    
    init_multihead_attention(total_embedding_dim, sequence_length, heads);
    
    // Test with different inputs should give different outputs
    float embeddings1[8] = {1, 1, 1, 1,
                           1, 1, 1, 1};
    
    float embeddings2[8] = {1, 2, 3, 4,
                           5, 6, 7, 8};
    
    float output1[8], output2[8];
    
    multihead_attention(embeddings1, output1);
    multihead_attention(embeddings2, output2);
    
    // Test: Different inputs should produce different outputs
    int outputs_different = 0;
    for(int i = 0; i < 8; i++) {
        if(fabs(output1[i] - output2[i]) > 0.001) {
            outputs_different = 1;
            break;
        }
    }
    
    assert(outputs_different);
    
    printf("✅ test_multihead_different_inputs passed\n");
}

void test_multihead_memory_safety() {
    // Test multiple initializations don't cause memory issues
    init_multihead_attention(8, 4, 2);
    init_multihead_attention(12, 3, 3);
    init_multihead_attention(16, 2, 4);
    
    float embeddings[32]; // 2 tokens * 16 dimensions
    for(int i = 0; i < 32; i++) {
        embeddings[i] = i;
    }
    
    float output[32];
    multihead_attention(embeddings, output);
    
    // Test: Should not crash and output should be finite
    for(int i = 0; i < 32; i++) {
        assert(isfinite(output[i]));
    }
    
    printf("✅ test_multihead_memory_safety passed\n");
}


void test_multihead_value_bounds() {
    int total_embedding_dim = 4;
    int sequence_length = 2;
    int heads = 2;
    
    init_multihead_attention(total_embedding_dim, sequence_length, heads);
    
    // Test with bounded input values
    float embeddings[8] = {1, 2, 3, 4,
                          5, 6, 7, 8};
    
    float output[8];
    
    multihead_attention(embeddings, output);
    
    printf("Input bounds: min=1.0, max=8.0\n");
    printf("Output values: ");
    for(int i = 0; i < 8; i++) {
        printf("%.3f ", output[i]);
    }
    printf("\n");
    
    // Find min/max of input
    float input_min = 1.0, input_max = 8.0;
    
    // Find min/max of output
    float output_min = output[0], output_max = output[0];
    for(int i = 1; i < 8; i++) {
        if(output[i] < output_min) output_min = output[i];
        if(output[i] > output_max) output_max = output[i];
    }
    
    printf("Output bounds: min=%.3f, max=%.3f\n", output_min, output_max);
    
    // Test: Output should be reasonable combinations of input values
    // Due to random weights, we can't be too strict, but outputs should be
    // in a reasonable range relative to inputs
    
    // 1. Outputs shouldn't be extremely far from input range
    float reasonable_lower = input_min - 10.0; // Allow some flexibility due to random weights
    float reasonable_upper = input_max + 10.0;
    
    for(int i = 0; i < 8; i++) {
        assert(output[i] >= reasonable_lower && output[i] <= reasonable_upper);
    }
    
    // 2. Test with zero input - should give bounded output
    float zero_embeddings[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float zero_output[8];
    
    multihead_attention(zero_embeddings, zero_output);
    
    printf("Zero input output: ");
    for(int i = 0; i < 8; i++) {
        printf("%.3f ", zero_output[i]);
        assert(isfinite(zero_output[i]));
    }
    printf("\n");
    
    // 3. Test with identical values - attention should still work
    float identical_embeddings[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    float identical_output[8];
    
    multihead_attention(identical_embeddings, identical_output);
    
    printf("Identical input output: ");
    for(int i = 0; i < 8; i++) {
        printf("%.3f ", identical_output[i]);
        assert(isfinite(identical_output[i]));
    }
    printf("\n");
    
    printf("✅ test_multihead_value_bounds passed\n");
}
int main() {
    test_matmul();
    test_transpose();
    test_softmax();
    test_compute_qkv();
    // test_attention();

    printf("Running attention tests...\n");
    test_attention_basic();
    test_attention_bounds();
    test_attention_self_attention();
    test_attention_symmetric();
    printf("All attention tests passed! ✅\n");


    printf("Running multihead attention tests...\n");
    test_multihead_init();
    test_multihead_init_invalid_dimension();
    test_multihead_basic_functionality();
    test_multihead_output_dimensions();
    test_multihead_single_head();
    test_multihead_different_inputs();
    test_multihead_memory_safety();
    test_multihead_value_bounds();
    printf("All multihead attention tests passed! ✅\n");
    return 0;
}