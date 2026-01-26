#include "rnnt_decoder.h"
#include "ggml_weights.h"

#include <cmath>
#include <cstdio>

using namespace nemo;

// Helper to check for NaN/Inf in tensor
bool check_valid(const TensorF& t, const char* name) {
    for (size_t i = 0; i < t.numel(); i++) {
        if (std::isnan(t.data[i]) || std::isinf(t.data[i])) {
            printf("FAIL: %s contains NaN or Inf at index %zu\n", name, i);
            return false;
        }
    }
    return true;
}

void print_stats(const TensorF& t, const char* name) {
    if (t.numel() == 0) {
        printf("%s: empty tensor\n", name);
        return;
    }
    float min_val = t.data[0], max_val = t.data[0], sum = 0;
    for (size_t i = 0; i < t.numel(); i++) {
        min_val = std::min(min_val, t.data[i]);
        max_val = std::max(max_val, t.data[i]);
        sum += t.data[i];
    }
    float mean = sum / t.numel();
    printf("%s: min=%.4f, max=%.4f, mean=%.4f\n", name, min_val, max_val, mean);
}

// ============================================================================
// Test single token forward
// ============================================================================
void test_single_token() {
    printf("\n=== Testing Single Token Forward ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTDecoder decoder;
    decoder.load_weights(weights);
    decoder.init_state(1);

    // Test with blank token
    TensorF output;
    decoder.forward_step(RNNTDecoder::BLANK_TOKEN, output);

    if (output.shape[0] != 1 || output.shape[1] != RNNTDecoder::HIDDEN_SIZE) {
        printf("FAIL: Expected shape [1, 640], got [%zu, %zu]\n",
               output.shape[0], output.shape[1]);
        return;
    }

    if (!check_valid(output, "output")) return;

    printf("Output shape: [%zu, %zu]\n", output.shape[0], output.shape[1]);
    print_stats(output, "After blank token");

    // Test with a regular token
    decoder.forward_step(100, output);
    if (!check_valid(output, "output")) return;
    print_stats(output, "After token 100");

    printf("OK: Single token forward\n");
}

// ============================================================================
// Test sequence of tokens
// ============================================================================
void test_token_sequence() {
    printf("\n=== Testing Token Sequence ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTDecoder decoder;
    decoder.load_weights(weights);
    decoder.init_state(1);

    // Simulate decoding a sequence: blank, then some tokens
    int tokens[] = {1024, 50, 100, 150, 200, 250};  // blank + 5 tokens
    TensorF output;

    printf("Processing sequence of %zu tokens:\n", sizeof(tokens)/sizeof(tokens[0]));

    for (int token : tokens) {
        decoder.forward_step(token, output);
        if (!check_valid(output, "output")) return;

        char buf[64];
        snprintf(buf, sizeof(buf), "  Token %4d", token);
        print_stats(output, buf);
    }

    printf("OK: Token sequence\n");
}

// ============================================================================
// Test state persistence
// ============================================================================
void test_state_persistence() {
    printf("\n=== Testing State Persistence ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTDecoder decoder;
    decoder.load_weights(weights);

    // Run same tokens twice with different initial states
    int tokens[] = {100, 200, 300};
    TensorF output1, output2;

    // First run: fresh state
    decoder.init_state(1);
    for (int token : tokens) {
        decoder.forward_step(token, output1);
    }

    // Second run: fresh state again
    decoder.init_state(1);
    for (int token : tokens) {
        decoder.forward_step(token, output2);
    }

    // Results should be identical
    bool same = true;
    for (size_t i = 0; i < output1.numel(); i++) {
        if (std::abs(output1.data[i] - output2.data[i]) > 1e-6f) {
            same = false;
            break;
        }
    }

    if (!same) {
        printf("FAIL: Same sequence with same initial state should give same output\n");
        return;
    }

    printf("OK: Same sequence + same state = same output\n");

    // Third run: different history
    decoder.init_state(1);
    decoder.forward_step(999, output2);  // Different first token
    for (int token : tokens) {
        decoder.forward_step(token, output2);
    }

    // Results should be different
    bool different = false;
    for (size_t i = 0; i < output1.numel(); i++) {
        if (std::abs(output1.data[i] - output2.data[i]) > 1e-6f) {
            different = true;
            break;
        }
    }

    if (!different) {
        printf("FAIL: Different history should give different output\n");
        return;
    }

    printf("OK: Different history = different output\n");
    printf("OK: State persistence\n");
}

// ============================================================================
// Test batch processing
// ============================================================================
void test_batch() {
    printf("\n=== Testing Batch Processing ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTDecoder decoder;
    decoder.load_weights(weights);

    // Batch size 4
    size_t batch_size = 4;
    decoder.init_state(batch_size);

    int tokens[] = {100, 200, 300, 400};
    TensorF output;
    decoder.forward(tokens, batch_size, output);

    if (output.shape[0] != batch_size || output.shape[1] != RNNTDecoder::HIDDEN_SIZE) {
        printf("FAIL: Expected shape [%zu, 640], got [%zu, %zu]\n",
               batch_size, output.shape[0], output.shape[1]);
        return;
    }

    if (!check_valid(output, "batch output")) return;

    printf("Output shape: [%zu, %zu]\n", output.shape[0], output.shape[1]);
    print_stats(output, "Batch output");

    printf("OK: Batch processing\n");
}

int main() {
    printf("=== Testing RNNT Decoder ===\n");

    test_single_token();
    test_token_sequence();
    test_state_persistence();
    test_batch();

    printf("\n=== All RNNT Decoder tests complete ===\n");
    return 0;
}
