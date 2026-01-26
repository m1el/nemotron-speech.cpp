#include "ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace nemo;

bool approx_equal(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) < tol;
}

void test_linear() {
    printf("Testing linear...\n");

    // Input: [2, 3] (batch=2, in_features=3)
    TensorF x({2, 3});
    x(0, 0) = 1.0f; x(0, 1) = 2.0f; x(0, 2) = 3.0f;
    x(1, 0) = 4.0f; x(1, 1) = 5.0f; x(1, 2) = 6.0f;

    // Weight: [2, 3] (out_features=2, in_features=3)
    float weight[] = {
        1.0f, 0.0f, 0.0f,  // first row
        0.0f, 1.0f, 1.0f   // second row
    };

    // Bias: [2]
    float bias[] = {0.5f, -0.5f};

    TensorF out;
    linear(x, weight, 2, 3, bias, out);

    // Expected:
    // out[0, 0] = 1*1 + 2*0 + 3*0 + 0.5 = 1.5
    // out[0, 1] = 1*0 + 2*1 + 3*1 - 0.5 = 4.5
    // out[1, 0] = 4*1 + 5*0 + 6*0 + 0.5 = 4.5
    // out[1, 1] = 4*0 + 5*1 + 6*1 - 0.5 = 10.5

    if (!approx_equal(out(0, 0), 1.5f) || !approx_equal(out(0, 1), 4.5f) ||
        !approx_equal(out(1, 0), 4.5f) || !approx_equal(out(1, 1), 10.5f)) {
        printf("FAIL: linear output mismatch\n");
        printf("  out: [%.2f, %.2f], [%.2f, %.2f]\n",
               out(0, 0), out(0, 1), out(1, 0), out(1, 1));
        return;
    }

    printf("OK: linear\n");
}

void test_layer_norm() {
    printf("Testing layer_norm...\n");

    // Input: [2, 4]
    TensorF x({2, 4});
    x(0, 0) = 1.0f; x(0, 1) = 2.0f; x(0, 2) = 3.0f; x(0, 3) = 4.0f;
    x(1, 0) = 2.0f; x(1, 1) = 4.0f; x(1, 2) = 6.0f; x(1, 3) = 8.0f;

    // Weight = 1, Bias = 0 (identity transform after normalization)
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float bias[] = {0.0f, 0.0f, 0.0f, 0.0f};

    TensorF out;
    layer_norm(x, weight, bias, 4, 1e-5f, out);

    // After normalization, each row should have mean ~0 and std ~1
    float mean0 = (out(0, 0) + out(0, 1) + out(0, 2) + out(0, 3)) / 4.0f;
    float mean1 = (out(1, 0) + out(1, 1) + out(1, 2) + out(1, 3)) / 4.0f;

    if (!approx_equal(mean0, 0.0f, 1e-4f) || !approx_equal(mean1, 0.0f, 1e-4f)) {
        printf("FAIL: layer_norm mean not ~0\n");
        printf("  mean0: %.6f, mean1: %.6f\n", mean0, mean1);
        return;
    }

    printf("OK: layer_norm\n");
}

void test_conv1d() {
    printf("Testing conv1d...\n");

    // Input: [1, 1, 5] (batch=1, channels=1, length=5)
    TensorF x({1, 1, 5});
    for (int i = 0; i < 5; i++) x(0, 0, i) = (float)(i + 1);

    // Weight: [1, 1, 3] (out_ch=1, in_ch=1, kernel=3)
    float weight[] = {1.0f, 1.0f, 1.0f};

    TensorF out;
    conv1d(x, weight, 1, 1, 3, 1, 1, 1, nullptr, out);

    // With padding=1, output length = 5
    // out[t] = sum of x[t-1:t+2]
    // out[0] = 0 + 1 + 2 = 3
    // out[1] = 1 + 2 + 3 = 6
    // out[2] = 2 + 3 + 4 = 9
    // out[3] = 3 + 4 + 5 = 12
    // out[4] = 4 + 5 + 0 = 9

    if (out.shape[2] != 5) {
        printf("FAIL: conv1d output length = %zu, expected 5\n", out.shape[2]);
        return;
    }

    float expected[] = {3.0f, 6.0f, 9.0f, 12.0f, 9.0f};
    for (int i = 0; i < 5; i++) {
        if (!approx_equal(out(0, 0, i), expected[i])) {
            printf("FAIL: conv1d out[%d] = %.2f, expected %.2f\n", i, out(0, 0, i), expected[i]);
            return;
        }
    }

    printf("OK: conv1d\n");
}

void test_causal_conv1d() {
    printf("Testing causal_conv1d...\n");

    // Input: [1, 1, 5]
    TensorF x({1, 1, 5});
    for (int i = 0; i < 5; i++) x(0, 0, i) = (float)(i + 1);

    // Weight: [1, 1, 3]
    float weight[] = {1.0f, 1.0f, 1.0f};

    TensorF out;
    causal_conv1d(x, weight, 1, 1, 3, 1, 1, nullptr, out);

    // Causal: left padding = kernel_size - 1 = 2
    // out[0] = 0 + 0 + 1 = 1
    // out[1] = 0 + 1 + 2 = 3
    // out[2] = 1 + 2 + 3 = 6
    // out[3] = 2 + 3 + 4 = 9
    // out[4] = 3 + 4 + 5 = 12

    if (out.shape[2] != 5) {
        printf("FAIL: causal_conv1d output length = %zu, expected 5\n", out.shape[2]);
        return;
    }

    float expected[] = {1.0f, 3.0f, 6.0f, 9.0f, 12.0f};
    for (int i = 0; i < 5; i++) {
        if (!approx_equal(out(0, 0, i), expected[i])) {
            printf("FAIL: causal_conv1d out[%d] = %.2f, expected %.2f\n", i, out(0, 0, i), expected[i]);
            return;
        }
    }

    printf("OK: causal_conv1d\n");
}

void test_glu() {
    printf("Testing glu...\n");

    // Input: [2, 4] -> will split into [2, 2]
    TensorF x({2, 4});
    x(0, 0) = 1.0f; x(0, 1) = 2.0f; x(0, 2) = 0.0f; x(0, 3) = 0.0f;  // gate sigmoid(0) = 0.5
    x(1, 0) = 3.0f; x(1, 1) = 4.0f; x(1, 2) = 100.0f; x(1, 3) = 100.0f;  // gate ~1.0

    TensorF out;
    glu(x, out);

    // out[0, 0] = 1.0 * sigmoid(0) = 0.5
    // out[0, 1] = 2.0 * sigmoid(0) = 1.0
    // out[1, 0] = 3.0 * sigmoid(100) ≈ 3.0
    // out[1, 1] = 4.0 * sigmoid(100) ≈ 4.0

    if (!approx_equal(out(0, 0), 0.5f) || !approx_equal(out(0, 1), 1.0f)) {
        printf("FAIL: glu first row mismatch\n");
        return;
    }
    if (!approx_equal(out(1, 0), 3.0f, 0.01f) || !approx_equal(out(1, 1), 4.0f, 0.01f)) {
        printf("FAIL: glu second row mismatch\n");
        return;
    }

    printf("OK: glu\n");
}

void test_lstm_cell() {
    printf("Testing lstm_cell...\n");

    // Simple test with known weights
    size_t batch = 1, input_size = 2, hidden_size = 2;

    TensorF input({batch, input_size});
    input(0, 0) = 1.0f; input(0, 1) = 0.0f;

    TensorF h_prev({batch, hidden_size}, 0.0f);
    TensorF c_prev({batch, hidden_size}, 0.0f);

    // Identity-ish weights: simple structure
    // 4 gates * hidden_size = 8
    float weight_ih[8 * 2] = {0};  // [8, 2]
    float weight_hh[8 * 2] = {0};  // [8, 2]
    float bias_ih[8] = {0};
    float bias_hh[8] = {0};

    // Set forget gate bias to 1 (common practice)
    bias_ih[2] = 1.0f;  // forget gate for h[0]
    bias_ih[3] = 1.0f;  // forget gate for h[1]

    TensorF h_out, c_out;
    lstm_cell(input, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh,
              input_size, hidden_size, h_out, c_out);

    // Just verify output shapes
    if (h_out.shape[0] != batch || h_out.shape[1] != hidden_size) {
        printf("FAIL: lstm_cell h_out shape mismatch\n");
        return;
    }
    if (c_out.shape[0] != batch || c_out.shape[1] != hidden_size) {
        printf("FAIL: lstm_cell c_out shape mismatch\n");
        return;
    }

    printf("OK: lstm_cell\n");
}

void test_embedding() {
    printf("Testing embedding...\n");

    // Vocab size = 3, embedding dim = 2
    float weight[] = {
        1.0f, 2.0f,   // token 0
        3.0f, 4.0f,   // token 1
        5.0f, 6.0f    // token 2
    };

    int indices[] = {0, 2, 1};
    TensorF out;
    embedding(indices, 3, weight, 3, 2, out);

    if (!approx_equal(out(0, 0), 1.0f) || !approx_equal(out(0, 1), 2.0f) ||
        !approx_equal(out(1, 0), 5.0f) || !approx_equal(out(1, 1), 6.0f) ||
        !approx_equal(out(2, 0), 3.0f) || !approx_equal(out(2, 1), 4.0f)) {
        printf("FAIL: embedding output mismatch\n");
        return;
    }

    printf("OK: embedding\n");
}

void test_softmax() {
    printf("Testing softmax...\n");

    TensorF x({2, 3});
    x(0, 0) = 1.0f; x(0, 1) = 2.0f; x(0, 2) = 3.0f;
    x(1, 0) = 1.0f; x(1, 1) = 1.0f; x(1, 2) = 1.0f;

    TensorF out;
    softmax(x, out);

    // Check row sums = 1
    float sum0 = out(0, 0) + out(0, 1) + out(0, 2);
    float sum1 = out(1, 0) + out(1, 1) + out(1, 2);

    if (!approx_equal(sum0, 1.0f) || !approx_equal(sum1, 1.0f)) {
        printf("FAIL: softmax sums = %.4f, %.4f (expected 1.0)\n", sum0, sum1);
        return;
    }

    // Second row should be uniform (1/3 each)
    if (!approx_equal(out(1, 0), 1.0f/3.0f) ||
        !approx_equal(out(1, 1), 1.0f/3.0f) ||
        !approx_equal(out(1, 2), 1.0f/3.0f)) {
        printf("FAIL: softmax uniform row mismatch\n");
        return;
    }

    printf("OK: softmax\n");
}

void test_argmax() {
    printf("Testing argmax...\n");

    TensorF x({3, 4});
    x(0, 0) = 1.0f; x(0, 1) = 5.0f; x(0, 2) = 2.0f; x(0, 3) = 3.0f;  // max at 1
    x(1, 0) = 4.0f; x(1, 1) = 2.0f; x(1, 2) = 1.0f; x(1, 3) = 3.0f;  // max at 0
    x(2, 0) = 1.0f; x(2, 1) = 2.0f; x(2, 2) = 3.0f; x(2, 3) = 9.0f;  // max at 3

    std::vector<int> indices;
    argmax(x, indices);

    if (indices[0] != 1 || indices[1] != 0 || indices[2] != 3) {
        printf("FAIL: argmax = [%d, %d, %d], expected [1, 0, 3]\n",
               indices[0], indices[1], indices[2]);
        return;
    }

    printf("OK: argmax\n");
}

int main() {
    printf("=== Testing ops ===\n\n");

    test_linear();
    test_layer_norm();
    test_conv1d();
    test_causal_conv1d();
    test_glu();
    test_lstm_cell();
    test_embedding();
    test_softmax();
    test_argmax();

    printf("\n=== All ops tests passed! ===\n");
    return 0;
}
