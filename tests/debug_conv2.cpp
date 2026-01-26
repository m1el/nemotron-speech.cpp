#include "conv_subsampling.h"
#include "ggml_weights.h"
#include <cstdio>
#include <cmath>

using namespace nemo;

bool check_nan(const TensorF& t, const char* name) {
    for (size_t i = 0; i < t.numel(); i++) {
        if (std::isnan(t.data[i]) || std::isinf(t.data[i])) {
            printf("NaN/Inf found in %s at index %zu\n", name, i);
            return true;
        }
    }
    float min_val = t.data[0], max_val = t.data[0];
    for (size_t i = 0; i < t.numel(); i++) {
        min_val = std::min(min_val, t.data[i]);
        max_val = std::max(max_val, t.data[i]);
    }
    printf("%s: shape=[", name);
    for (size_t i = 0; i < t.shape.size(); i++) {
        printf("%zu%s", t.shape[i], i < t.shape.size()-1 ? ", " : "");
    }
    printf("] min=%.4f max=%.4f\n", min_val, max_val);
    return false;
}

int main() {
    ModelWeights weights;
    weights.load("weights/model.bin");

    // Get weights
    const float* conv0_weight = weights.require("encoder.pre_encode.conv.0.weight").data.data();
    const float* conv0_bias = weights.require("encoder.pre_encode.conv.0.bias").data.data();
    const float* conv2_weight = weights.require("encoder.pre_encode.conv.2.weight").data.data();
    const float* conv2_bias = weights.require("encoder.pre_encode.conv.2.bias").data.data();
    const float* conv3_weight = weights.require("encoder.pre_encode.conv.3.weight").data.data();
    const float* conv3_bias = weights.require("encoder.pre_encode.conv.3.bias").data.data();

    // Create test input
    TensorF input({1, 32, 128});
    for (size_t t = 0; t < 32; t++) {
        for (size_t f = 0; f < 128; f++) {
            input(0, t, f) = std::sin((float)t * 0.1f + (float)f * 0.05f);
        }
    }
    check_nan(input, "input");

    // Reshape to [1, 1, 32, 128]
    TensorF x({1, 1, 32, 128});
    for (size_t t = 0; t < 32; t++) {
        for (size_t f = 0; f < 128; f++) {
            x(0, 0, t, f) = input(0, t, f);
        }
    }
    check_nan(x, "x_reshaped");

    // Conv0
    TensorF buf1, buf2;
    printf("\nApplying Conv0 (CausalConv2D 1->256, k=3, s=2)...\n");
    causal_conv2d(x, conv0_weight, 256, 1, 3, 3, 2, 2, 1, conv0_bias, buf1);
    if (check_nan(buf1, "after_conv0")) return 1;

    // ReLU
    relu_inplace(buf1);
    if (check_nan(buf1, "after_relu0")) return 1;

    // Conv2: Depthwise
    printf("\nApplying Conv2 (Depthwise CausalConv2D 256->256, k=3, s=2, groups=256)...\n");
    causal_conv2d(buf1, conv2_weight, 256, 256, 3, 3, 2, 2, 256, conv2_bias, buf2);
    if (check_nan(buf2, "after_conv2")) return 1;

    // Conv3: Pointwise
    printf("\nApplying Conv3 (Pointwise Conv2d 256->256, k=1, s=1)...\n");
    conv2d(buf2, conv3_weight, 256, 256, 1, 1, 1, 1, 0, 0, 1, conv3_bias, buf1);
    if (check_nan(buf1, "after_conv3")) return 1;

    relu_inplace(buf1);
    if (check_nan(buf1, "after_relu1")) return 1;

    printf("\nNo NaN detected through conv3!\n");
    return 0;
}
