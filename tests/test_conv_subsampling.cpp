#include "conv_subsampling.h"
#include "ggml_weights.h"

#include <cmath>
#include <cstdio>

using namespace nemo;

void test_basic_forward() {
    printf("Testing ConvSubsampling basic forward...\n");

    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ConvSubsampling subsample;
    subsample.load_weights(weights);

    // Create test input: [1, 100, 128]
    // (1 batch, 100 time frames, 128 mel features)
    TensorF input({1, 100, 128});
    // Fill with simple pattern
    for (size_t t = 0; t < 100; t++) {
        for (size_t f = 0; f < 128; f++) {
            input(0, t, f) = std::sin((float)t * 0.1f + (float)f * 0.05f);
        }
    }

    TensorF output;
    subsample.forward(input, output);

    // Check output shape
    printf("Input shape: [%zu, %zu, %zu]\n", input.shape[0], input.shape[1], input.shape[2]);
    printf("Output shape: [%zu, %zu, %zu]\n", output.shape[0], output.shape[1], output.shape[2]);

    // Output should be [1, time/8, 1024]
    if (output.shape[0] != 1) {
        printf("FAIL: batch size mismatch\n");
        return;
    }
    if (output.shape[2] != 1024) {
        printf("FAIL: output features should be 1024, got %zu\n", output.shape[2]);
        return;
    }

    // Expected time reduction: roughly input/8
    size_t expected_time = ConvSubsampling::get_output_length(100);
    printf("Expected output time: %zu, actual: %zu\n", expected_time, output.shape[1]);

    // Check values are not NaN or Inf
    bool has_nan = false;
    for (size_t i = 0; i < output.numel(); i++) {
        if (std::isnan(output.data[i]) || std::isinf(output.data[i])) {
            has_nan = true;
            break;
        }
    }
    if (has_nan) {
        printf("FAIL: output contains NaN or Inf\n");
        return;
    }

    // Print some output values for visual inspection
    printf("Output sample values (first 5 of first frame):\n");
    for (size_t i = 0; i < 5; i++) {
        printf("  [0, 0, %zu] = %.6f\n", i, output(0, 0, i));
    }

    printf("OK: ConvSubsampling forward\n");
}

void test_different_lengths() {
    printf("\nTesting ConvSubsampling with different input lengths...\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ConvSubsampling subsample;
    subsample.load_weights(weights);

    // Test various lengths
    size_t lengths[] = {16, 32, 64, 100, 128, 256, 512};

    for (size_t len : lengths) {
        TensorF input({1, len, 128}, 0.1f);
        TensorF output;
        subsample.forward(input, output);

        printf("  Input: %zu -> Output: %zu (expected ~%zu)\n",
               len, output.shape[1], ConvSubsampling::get_output_length(len));
    }

    printf("OK: Different lengths\n");
}

int main() {
    printf("=== Testing ConvSubsampling ===\n\n");

    test_basic_forward();
    test_different_lengths();

    printf("\n=== ConvSubsampling tests complete ===\n");
    return 0;
}
