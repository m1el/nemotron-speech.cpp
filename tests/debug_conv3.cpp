#include "conv_subsampling.h"
#include "ggml_weights.h"
#include <cstdio>
#include <cmath>

using namespace nemo;

bool check_nan(const TensorF& t, const char* name) {
    for (size_t i = 0; i < t.numel(); i++) {
        if (std::isnan(t.data[i]) || std::isinf(t.data[i])) {
            printf("NaN/Inf found in %s at index %zu (value=%f)\n", name, i, t.data[i]);
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

    // Get all weights
    const float* conv0_weight = weights.require("encoder.pre_encode.conv.0.weight").data.data();
    const float* conv0_bias = weights.require("encoder.pre_encode.conv.0.bias").data.data();
    const float* conv2_weight = weights.require("encoder.pre_encode.conv.2.weight").data.data();
    const float* conv2_bias = weights.require("encoder.pre_encode.conv.2.bias").data.data();
    const float* conv3_weight = weights.require("encoder.pre_encode.conv.3.weight").data.data();
    const float* conv3_bias = weights.require("encoder.pre_encode.conv.3.bias").data.data();
    const float* conv5_weight = weights.require("encoder.pre_encode.conv.5.weight").data.data();
    const float* conv5_bias = weights.require("encoder.pre_encode.conv.5.bias").data.data();
    const float* conv6_weight = weights.require("encoder.pre_encode.conv.6.weight").data.data();
    const float* conv6_bias = weights.require("encoder.pre_encode.conv.6.bias").data.data();
    const float* out_weight = weights.require("encoder.pre_encode.out.weight").data.data();
    const float* out_bias = weights.require("encoder.pre_encode.out.bias").data.data();

    // Create test input - use smaller values
    TensorF input({1, 100, 128});
    for (size_t t = 0; t < 100; t++) {
        for (size_t f = 0; f < 128; f++) {
            input(0, t, f) = 0.1f * std::sin((float)t * 0.1f + (float)f * 0.05f);
        }
    }

    // Reshape to [1, 1, time, 128]
    TensorF x({1, 1, 100, 128});
    for (size_t t = 0; t < 100; t++) {
        for (size_t f = 0; f < 128; f++) {
            x(0, 0, t, f) = input(0, t, f);
        }
    }

    TensorF buf1, buf2;

    // Conv0 + ReLU
    causal_conv2d(x, conv0_weight, 256, 1, 3, 3, 2, 2, 1, conv0_bias, buf1);
    relu_inplace(buf1);
    check_nan(buf1, "after_conv0_relu");

    // Conv2 (depthwise)
    causal_conv2d(buf1, conv2_weight, 256, 256, 3, 3, 2, 2, 256, conv2_bias, buf2);
    check_nan(buf2, "after_conv2");

    // Conv3 (pointwise) + ReLU
    conv2d(buf2, conv3_weight, 256, 256, 1, 1, 1, 1, 0, 0, 1, conv3_bias, buf1);
    relu_inplace(buf1);
    check_nan(buf1, "after_conv3_relu");

    // Conv5 (depthwise)
    causal_conv2d(buf1, conv5_weight, 256, 256, 3, 3, 2, 2, 256, conv5_bias, buf2);
    check_nan(buf2, "after_conv5");

    // Conv6 (pointwise) + ReLU
    conv2d(buf2, conv6_weight, 256, 256, 1, 1, 1, 1, 0, 0, 1, conv6_bias, buf1);
    relu_inplace(buf1);
    check_nan(buf1, "after_conv6_relu");

    // Reshape and linear
    size_t time_out = buf1.shape[2];
    size_t width_out = buf1.shape[3];
    size_t flat_dim = 256 * width_out;
    printf("Reshaping to [1, %zu, %zu] (flat_dim=%zu)\n", time_out, flat_dim, flat_dim);

    TensorF flat({1, time_out, flat_dim});
    for (size_t t = 0; t < time_out; t++) {
        for (size_t c = 0; c < 256; c++) {
            for (size_t w = 0; w < width_out; w++) {
                flat(0, t, c * width_out + w) = buf1(0, c, t, w);
            }
        }
    }
    check_nan(flat, "flat");

    // Linear
    printf("Applying linear(%zu -> 1024)...\n", flat_dim);
    TensorF output;
    linear(flat, out_weight, 1024, flat_dim, out_bias, output);
    check_nan(output, "output");

    return 0;
}
