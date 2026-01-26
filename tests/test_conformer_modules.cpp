#include "conformer_modules.h"
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

// Print first few values of tensor for debugging
void print_sample(const TensorF& t, const char* name, size_t n = 5) {
    printf("%s sample values: ", name);
    size_t count = std::min(n, t.numel());
    for (size_t i = 0; i < count; i++) {
        printf("%.6f ", t.data[i]);
    }
    printf("\n");
}

// ============================================================================
// Test RelPositionalEncoding
// ============================================================================
void test_rel_positional_encoding() {
    printf("\n=== Testing RelPositionalEncoding ===\n");

    RelPositionalEncoding pos_enc;
    pos_enc.init();

    // Test with different sequence lengths
    size_t lengths[] = {10, 32, 64, 128};

    for (size_t len : lengths) {
        TensorF pos_emb;
        pos_enc.get_pos_emb(len, pos_emb);

        // Expected shape: [2*len - 1, 1024]
        size_t expected_len = 2 * len - 1;
        if (pos_emb.shape[0] != expected_len || pos_emb.shape[1] != 1024) {
            printf("FAIL: seq_len=%zu, expected shape [%zu, 1024], got [%zu, %zu]\n",
                   len, expected_len, pos_emb.shape[0], pos_emb.shape[1]);
            return;
        }

        if (!check_valid(pos_emb, "pos_emb")) return;

        printf("  seq_len=%zu -> pos_emb shape [%zu, %zu] OK\n",
               len, pos_emb.shape[0], pos_emb.shape[1]);
    }

    // Check that values are bounded (sinusoidal should be in [-1, 1])
    TensorF pos_emb;
    pos_enc.get_pos_emb(32, pos_emb);
    float min_val = pos_emb.data[0], max_val = pos_emb.data[0];
    for (size_t i = 0; i < pos_emb.numel(); i++) {
        min_val = std::min(min_val, pos_emb.data[i]);
        max_val = std::max(max_val, pos_emb.data[i]);
    }
    printf("  Value range: [%.4f, %.4f]\n", min_val, max_val);

    if (min_val < -1.1f || max_val > 1.1f) {
        printf("FAIL: Positional encoding values out of expected range\n");
        return;
    }

    printf("OK: RelPositionalEncoding\n");
}

// ============================================================================
// Test ConformerFeedForward
// ============================================================================
void test_feedforward() {
    printf("\n=== Testing ConformerFeedForward ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ConformerFeedForward ffn;
    // Use first layer's FFN1 weights
    ffn.load_weights(weights, "encoder.layers.0.feed_forward1");

    // Test input: [1, 10, 1024]
    TensorF input({1, 10, 1024});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = std::sin((float)i * 0.01f) * 0.1f;
    }

    TensorF output;
    ffn.forward(input, output);

    // Check output shape
    if (output.shape[0] != 1 || output.shape[1] != 10 || output.shape[2] != 1024) {
        printf("FAIL: Expected shape [1, 10, 1024], got [%zu, %zu, %zu]\n",
               output.shape[0], output.shape[1], output.shape[2]);
        return;
    }

    if (!check_valid(output, "output")) return;
    print_sample(output, "FFN output");

    printf("OK: ConformerFeedForward\n");
}

// ============================================================================
// Test ConformerConvolution
// ============================================================================
void test_conv_module() {
    printf("\n=== Testing ConformerConvolution ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ConformerConvolution conv;
    // Use first layer's conv module weights
    conv.load_weights(weights, "encoder.layers.0.conv");

    // Test input: [1, 16, 1024]
    TensorF input({1, 16, 1024});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = std::sin((float)i * 0.01f) * 0.1f;
    }

    TensorF output;
    conv.forward(input, output);

    // Check output shape - should be same as input for conv module
    if (output.shape[0] != 1 || output.shape[1] != 16 || output.shape[2] != 1024) {
        printf("FAIL: Expected shape [1, 16, 1024], got [%zu, %zu, %zu]\n",
               output.shape[0], output.shape[1], output.shape[2]);
        return;
    }

    if (!check_valid(output, "output")) return;
    print_sample(output, "Conv module output");

    // Test with different lengths
    printf("  Testing different lengths:\n");
    size_t lengths[] = {8, 16, 32, 64};
    for (size_t len : lengths) {
        TensorF in({1, len, 1024}, 0.1f);
        TensorF out;
        conv.forward(in, out);
        printf("    len=%zu -> output shape [%zu, %zu, %zu]\n",
               len, out.shape[0], out.shape[1], out.shape[2]);
    }

    printf("OK: ConformerConvolution\n");
}

// ============================================================================
// Test RelPositionMultiHeadAttention
// ============================================================================
void test_attention() {
    printf("\n=== Testing RelPositionMultiHeadAttention ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RelPositionMultiHeadAttention attn;
    // Use first layer's attention weights
    attn.load_weights(weights, "encoder.layers.0.self_attn");

    RelPositionalEncoding pos_enc;
    pos_enc.init();

    // Test with small sequence: [1, 8, 1024]
    size_t seq_len = 8;
    TensorF input({1, seq_len, 1024});
    for (size_t i = 0; i < input.numel(); i++) {
        input.data[i] = std::sin((float)i * 0.01f) * 0.1f;
    }

    TensorF pos_emb;
    pos_enc.get_pos_emb(seq_len, pos_emb);

    TensorF output;
    attn.forward(input, pos_emb, output);

    // Check output shape
    if (output.shape[0] != 1 || output.shape[1] != seq_len || output.shape[2] != 1024) {
        printf("FAIL: Expected shape [1, %zu, 1024], got [%zu, %zu, %zu]\n",
               seq_len, output.shape[0], output.shape[1], output.shape[2]);
        return;
    }

    if (!check_valid(output, "output")) return;
    print_sample(output, "Attention output");

    // Test with different sequence lengths
    printf("  Testing different lengths:\n");
    size_t lengths[] = {4, 8, 16, 32};
    for (size_t len : lengths) {
        TensorF in({1, len, 1024}, 0.1f);
        TensorF pe;
        pos_enc.get_pos_emb(len, pe);
        TensorF out;
        attn.forward(in, pe, out);
        if (!check_valid(out, "out")) return;
        printf("    len=%zu -> output shape [%zu, %zu, %zu] OK\n",
               len, out.shape[0], out.shape[1], out.shape[2]);
    }

    printf("OK: RelPositionMultiHeadAttention\n");
}

// ============================================================================
// Integration test: all components together
// ============================================================================
void test_integration() {
    printf("\n=== Integration Test ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    // Load components for layer 0
    ConformerFeedForward ffn1, ffn2;
    ConformerConvolution conv;
    RelPositionMultiHeadAttention attn;
    RelPositionalEncoding pos_enc;

    ffn1.load_weights(weights, "encoder.layers.0.feed_forward1");
    ffn2.load_weights(weights, "encoder.layers.0.feed_forward2");
    conv.load_weights(weights, "encoder.layers.0.conv");
    attn.load_weights(weights, "encoder.layers.0.self_attn");
    pos_enc.init();

    // Simulate one conformer layer pass (without layer norm for simplicity)
    // Real layer: x -> LN -> FFN1 -> +x*0.5 -> LN -> Attn -> +x -> LN -> Conv -> +x -> LN -> FFN2 -> +x*0.5 -> LN

    size_t seq_len = 16;
    TensorF x({1, seq_len, 1024});
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] = std::sin((float)i * 0.005f) * 0.1f;
    }

    printf("Input stats: min=%.6f, max=%.6f\n",
           *std::min_element(x.data.begin(), x.data.end()),
           *std::max_element(x.data.begin(), x.data.end()));

    // FFN1
    TensorF ffn1_out;
    ffn1.forward(x, ffn1_out);
    // Residual with 0.5 scale (simplified - no layer norm)
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] += 0.5f * ffn1_out.data[i];
    }
    if (!check_valid(x, "after FFN1")) return;
    printf("After FFN1: min=%.6f, max=%.6f\n",
           *std::min_element(x.data.begin(), x.data.end()),
           *std::max_element(x.data.begin(), x.data.end()));

    // Attention
    TensorF pos_emb;
    pos_enc.get_pos_emb(seq_len, pos_emb);
    TensorF attn_out;
    attn.forward(x, pos_emb, attn_out);
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] += attn_out.data[i];
    }
    if (!check_valid(x, "after Attn")) return;
    printf("After Attn: min=%.6f, max=%.6f\n",
           *std::min_element(x.data.begin(), x.data.end()),
           *std::max_element(x.data.begin(), x.data.end()));

    // Conv module
    TensorF conv_out;
    conv.forward(x, conv_out);
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] += conv_out.data[i];
    }
    if (!check_valid(x, "after Conv")) return;
    printf("After Conv: min=%.6f, max=%.6f\n",
           *std::min_element(x.data.begin(), x.data.end()),
           *std::max_element(x.data.begin(), x.data.end()));

    // FFN2
    TensorF ffn2_out;
    ffn2.forward(x, ffn2_out);
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] += 0.5f * ffn2_out.data[i];
    }
    if (!check_valid(x, "after FFN2")) return;
    printf("After FFN2: min=%.6f, max=%.6f\n",
           *std::min_element(x.data.begin(), x.data.end()),
           *std::max_element(x.data.begin(), x.data.end()));

    printf("OK: Integration test passed\n");
}

int main() {
    printf("=== Testing Conformer Modules ===\n");

    test_rel_positional_encoding();
    test_feedforward();
    test_conv_module();
    test_attention();
    test_integration();

    printf("\n=== All Conformer Module tests complete ===\n");
    return 0;
}
