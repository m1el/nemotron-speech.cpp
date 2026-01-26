#include "conformer_encoder.h"
#include "ggml_weights.h"

#include <cmath>
#include <cstdio>
#include <chrono>

using namespace nemo;

// Helper to check for NaN/Inf in tensor
bool check_valid(const TensorF& t, const char* name) {
    for (size_t i = 0; i < t.numel(); i++) {
        if (std::isnan(t.data[i]) || std::isinf(t.data[i])) {
            printf("FAIL: %s contains NaN or Inf at index %zu (value: %f)\n", name, i, t.data[i]);
            return false;
        }
    }
    return true;
}

// Print tensor stats
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
// Test single ConformerLayer
// ============================================================================
void test_single_layer() {
    printf("\n=== Testing Single ConformerLayer ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ConformerLayer layer;
    layer.load_weights(weights, "encoder.layers.0");

    RelPositionalEncoding pos_enc;
    pos_enc.init();

    // Test input: [1, 16, 1024]
    size_t seq_len = 16;
    TensorF input({1, seq_len, 1024});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = std::sin((float)i * 0.001f) * 0.1f;
    }

    TensorF pos_emb;
    pos_enc.get_pos_emb(seq_len, pos_emb);

    TensorF output;
    layer.forward(input, pos_emb, output);

    // Check output shape
    if (output.shape[0] != 1 || output.shape[1] != seq_len || output.shape[2] != 1024) {
        printf("FAIL: Expected shape [1, %zu, 1024], got [%zu, %zu, %zu]\n",
               seq_len, output.shape[0], output.shape[1], output.shape[2]);
        return;
    }

    if (!check_valid(output, "layer output")) return;

    print_stats(input, "Input");
    print_stats(output, "Output");

    printf("OK: Single ConformerLayer\n");
}

// ============================================================================
// Test multiple layers
// ============================================================================
void test_multiple_layers() {
    printf("\n=== Testing Multiple Layers ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RelPositionalEncoding pos_enc;
    pos_enc.init();

    size_t seq_len = 16;
    TensorF pos_emb;
    pos_enc.get_pos_emb(seq_len, pos_emb);

    TensorF x({1, seq_len, 1024});
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] = std::sin((float)i * 0.001f) * 0.1f;
    }

    printf("Processing through layers 0-5...\n");
    print_stats(x, "Input");

    for (int i = 0; i < 6; i++) {
        ConformerLayer layer;
        std::string prefix = "encoder.layers." + std::to_string(i);
        layer.load_weights(weights, prefix);

        TensorF out;
        layer.forward(x, pos_emb, out);

        if (!check_valid(out, "layer output")) {
            printf("FAIL at layer %d\n", i);
            return;
        }

        char buf[64];
        snprintf(buf, sizeof(buf), "After layer %d", i);
        print_stats(out, buf);

        x = out;
    }

    printf("OK: Multiple layers processed without explosion\n");
}

// ============================================================================
// Test full encoder
// ============================================================================
void test_full_encoder() {
    printf("\n=== Testing Full ConformerEncoder ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ConformerEncoder encoder;
    printf("Loading encoder weights (24 layers)...\n");
    encoder.load_weights(weights);
    printf("Weights loaded.\n");

    // Test input: [1, 128, 128] mel features (128 frames)
    TensorF input({1, 128, 128});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = std::sin((float)i * 0.0001f) * 0.5f;
    }

    printf("Input shape: [%zu, %zu, %zu]\n", input.shape[0], input.shape[1], input.shape[2]);
    print_stats(input, "Input");

    TensorF output;

    auto start = std::chrono::high_resolution_clock::now();
    encoder.forward(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Output shape: [%zu, %zu, %zu]\n", output.shape[0], output.shape[1], output.shape[2]);

    // Check output shape
    // Input: 128 frames -> after 8x subsampling: ~17 frames
    size_t expected_time = ConvSubsampling::get_output_length(128);
    if (output.shape[0] != 1 || output.shape[2] != 1024) {
        printf("FAIL: Expected shape [1, ~%zu, 1024], got [%zu, %zu, %zu]\n",
               expected_time, output.shape[0], output.shape[1], output.shape[2]);
        return;
    }

    if (!check_valid(output, "encoder output")) return;

    print_stats(output, "Encoder output");
    printf("Inference time: %ld ms\n", duration.count());

    printf("OK: Full ConformerEncoder\n");
}

// ============================================================================
// Test with different input lengths
// ============================================================================
void test_variable_lengths() {
    printf("\n=== Testing Variable Input Lengths ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ConformerEncoder encoder;
    encoder.load_weights(weights);

    size_t lengths[] = {64, 128, 256, 512};

    for (size_t len : lengths) {
        TensorF input({1, len, 128}, 0.1f);
        TensorF output;

        auto start = std::chrono::high_resolution_clock::now();
        encoder.forward(input, output);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (!check_valid(output, "output")) {
            printf("FAIL at length %zu\n", len);
            return;
        }

        printf("  Input: [1, %zu, 128] -> Output: [1, %zu, 1024] (%ld ms)\n",
               len, output.shape[1], duration.count());
    }

    printf("OK: Variable input lengths\n");
}

int main() {
    printf("=== Testing Conformer Encoder ===\n");

    test_single_layer();
    test_multiple_layers();
    test_full_encoder();
    test_variable_lengths();

    printf("\n=== All Conformer Encoder tests complete ===\n");
    return 0;
}
