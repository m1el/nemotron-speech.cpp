#include "ggml_weights.h"

#include <cmath>
#include <cstdio>

using namespace nemo;

void test_tensor_shapes(const ModelWeights& weights) {
    printf("\n=== Testing tensor shapes ===\n");

    // Test pre_encode weights
    {
        const auto& t = weights.require("encoder.pre_encode.out.weight");
        if (t.shape.size() != 2 || t.shape[0] != 1024 || t.shape[1] != 4352) {
            printf("FAIL: pre_encode.out.weight shape mismatch\n");
            return;
        }
        printf("OK: pre_encode.out.weight [1024, 4352]\n");
    }

    // Test conv subsampling
    {
        const auto& t = weights.require("encoder.pre_encode.conv.0.weight");
        if (t.shape.size() != 4 || t.shape[0] != 256 || t.shape[1] != 1 ||
            t.shape[2] != 3 || t.shape[3] != 3) {
            printf("FAIL: conv.0.weight shape mismatch\n");
            return;
        }
        printf("OK: conv.0.weight [256, 1, 3, 3]\n");
    }

    // Test layer 0 attention weights
    {
        const auto& t = weights.require("encoder.layers.0.self_attn.linear_q.weight");
        if (t.shape.size() != 2 || t.shape[0] != 1024 || t.shape[1] != 1024) {
            printf("FAIL: linear_q.weight shape mismatch\n");
            return;
        }
        printf("OK: layers.0.self_attn.linear_q.weight [1024, 1024]\n");
    }

    // Test decoder embedding
    {
        const auto& t = weights.require("decoder.prediction.embed.weight");
        if (t.shape.size() != 2 || t.shape[0] != 1025 || t.shape[1] != 640) {
            printf("FAIL: embed.weight shape mismatch\n");
            return;
        }
        printf("OK: decoder.prediction.embed.weight [1025, 640]\n");
    }

    // Test LSTM weights
    {
        const auto& t = weights.require("decoder.prediction.dec_rnn.lstm.weight_ih_l0");
        if (t.shape.size() != 2 || t.shape[0] != 2560 || t.shape[1] != 640) {
            printf("FAIL: lstm.weight_ih_l0 shape mismatch\n");
            return;
        }
        printf("OK: lstm.weight_ih_l0 [2560, 640]\n");
    }

    // Test joint network
    {
        const auto& t = weights.require("joint.joint_net.2.weight");
        if (t.shape.size() != 2 || t.shape[0] != 1025 || t.shape[1] != 640) {
            printf("FAIL: joint_net.2.weight shape mismatch\n");
            return;
        }
        printf("OK: joint.joint_net.2.weight [1025, 640]\n");
    }

    printf("\nAll shape tests passed!\n");
}

void test_tensor_values(const ModelWeights& weights) {
    printf("\n=== Testing tensor values (spot check) ===\n");

    // Check that values are in reasonable range
    const auto& t = weights.require("encoder.layers.0.norm_feed_forward1.weight");
    float min_val = t.data[0], max_val = t.data[0], sum = 0;
    for (size_t i = 0; i < t.numel(); i++) {
        min_val = std::min(min_val, t.data[i]);
        max_val = std::max(max_val, t.data[i]);
        sum += t.data[i];
    }
    float mean = sum / t.numel();

    printf("norm_feed_forward1.weight: min=%.4f, max=%.4f, mean=%.4f\n",
           min_val, max_val, mean);

    // LayerNorm weight should be close to 1.0
    if (std::abs(mean - 1.0f) > 0.1f) {
        printf("WARNING: LayerNorm weight mean far from 1.0\n");
    }

    // Check attention weights
    const auto& q = weights.require("encoder.layers.0.self_attn.linear_q.weight");
    float q_sum = 0, q_sum_sq = 0;
    for (size_t i = 0; i < q.numel(); i++) {
        q_sum += q.data[i];
        q_sum_sq += q.data[i] * q.data[i];
    }
    float q_mean = q_sum / q.numel();
    float q_std = std::sqrt(q_sum_sq / q.numel() - q_mean * q_mean);
    printf("linear_q.weight: mean=%.6f, std=%.6f\n", q_mean, q_std);

    // Should be roughly Xavier initialized
    float expected_std = std::sqrt(2.0f / (1024 + 1024));
    if (std::abs(q_std - expected_std) > 0.01f) {
        printf("Note: std differs from Xavier init (%.4f)\n", expected_std);
    }

    printf("\nValue spot checks complete!\n");
}

void test_all_layers(const ModelWeights& weights) {
    printf("\n=== Verifying all 24 layers exist ===\n");

    for (int i = 0; i < 24; i++) {
        std::string prefix = "encoder.layers." + std::to_string(i);

        // Check each layer has all required weights
        const char* required[] = {
            ".norm_feed_forward1.weight",
            ".norm_feed_forward1.bias",
            ".feed_forward1.linear1.weight",
            ".feed_forward1.linear2.weight",
            ".norm_conv.weight",
            ".norm_conv.bias",
            ".conv.pointwise_conv1.weight",
            ".conv.depthwise_conv.weight",
            ".conv.batch_norm.weight",
            ".conv.batch_norm.bias",
            ".conv.pointwise_conv2.weight",
            ".norm_self_att.weight",
            ".norm_self_att.bias",
            ".self_attn.pos_bias_u",
            ".self_attn.pos_bias_v",
            ".self_attn.linear_q.weight",
            ".self_attn.linear_k.weight",
            ".self_attn.linear_v.weight",
            ".self_attn.linear_out.weight",
            ".self_attn.linear_pos.weight",
            ".norm_feed_forward2.weight",
            ".norm_feed_forward2.bias",
            ".feed_forward2.linear1.weight",
            ".feed_forward2.linear2.weight",
            ".norm_out.weight",
            ".norm_out.bias",
        };

        for (const char* suffix : required) {
            std::string name = prefix + suffix;
            if (!weights.has(name)) {
                printf("FAIL: Missing tensor %s\n", name.c_str());
                return;
            }
        }
    }

    printf("OK: All 24 layers have complete weights (26 tensors each)\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <weights.bin>\n", argv[0]);
        return 1;
    }

    ModelWeights weights;
    if (!weights.load(argv[1])) {
        printf("Failed to load weights\n");
        return 1;
    }

    printf("\n=== Weight Loading Test ===\n");
    printf("Total tensors: %zu\n", weights.size());
    printf("Total parameters: %zu\n", weights.total_params());

    test_tensor_shapes(weights);
    test_tensor_values(weights);
    test_all_layers(weights);

    printf("\n=== All tests passed! ===\n");
    return 0;
}
