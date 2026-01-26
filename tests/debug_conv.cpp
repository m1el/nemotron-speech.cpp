#include "conv_subsampling.h"
#include "ggml_weights.h"
#include <cstdio>
#include <cmath>

using namespace nemo;

int main() {
    ModelWeights weights;
    weights.load("weights/model.bin");

    // Check conv weights for NaN/Inf
    const char* names[] = {
        "encoder.pre_encode.conv.0.weight",
        "encoder.pre_encode.conv.0.bias",
        "encoder.pre_encode.conv.2.weight",
        "encoder.pre_encode.conv.2.bias",
        "encoder.pre_encode.conv.3.weight",
        "encoder.pre_encode.conv.3.bias",
    };

    for (const char* name : names) {
        const auto& t = weights.require(name);
        float min_val = t.data[0], max_val = t.data[0];
        bool has_nan = false;
        for (size_t i = 0; i < t.numel(); i++) {
            if (std::isnan(t.data[i]) || std::isinf(t.data[i])) has_nan = true;
            min_val = std::min(min_val, t.data[i]);
            max_val = std::max(max_val, t.data[i]);
        }
        printf("%s: min=%.4f max=%.4f nan=%d\n", name, min_val, max_val, has_nan);
    }

    // Simple conv2d test
    printf("\nTesting simple conv2d...\n");
    TensorF input({1, 1, 4, 4}, 1.0f);
    float weight[9] = {1,1,1, 1,1,1, 1,1,1};  // 3x3 kernel of 1s
    TensorF output;

    // Standard conv2d
    conv2d(input, weight, 1, 1, 3, 3, 1, 1, 1, 1, 1, nullptr, output);
    printf("conv2d output shape: [%zu, %zu, %zu, %zu]\n",
           output.shape[0], output.shape[1], output.shape[2], output.shape[3]);
    printf("conv2d output[0,0,1,1] = %.2f (expected 9.0)\n", output(0, 0, 1, 1));

    // Causal conv2d
    causal_conv2d(input, weight, 1, 1, 3, 3, 1, 1, 1, nullptr, output);
    printf("causal_conv2d output shape: [%zu, %zu, %zu, %zu]\n",
           output.shape[0], output.shape[1], output.shape[2], output.shape[3]);

    // Check causal conv with stride
    printf("\nTesting causal_conv2d with stride=2...\n");
    TensorF input2({1, 1, 8, 8}, 0.5f);
    causal_conv2d(input2, weight, 1, 1, 3, 3, 2, 2, 1, nullptr, output);
    printf("causal_conv2d(s=2) output shape: [%zu, %zu, %zu, %zu]\n",
           output.shape[0], output.shape[1], output.shape[2], output.shape[3]);

    bool has_nan = false;
    for (size_t i = 0; i < output.numel(); i++) {
        if (std::isnan(output.data[i])) {
            has_nan = true;
            break;
        }
    }
    printf("Has NaN: %d\n", has_nan);

    return 0;
}
