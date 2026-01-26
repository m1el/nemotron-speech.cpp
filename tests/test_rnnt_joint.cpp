#include "rnnt_joint.h"
#include "ggml_weights.h"

#include <cmath>
#include <cstdio>

using namespace nemo;

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
// Test single frame forward
// ============================================================================
void test_single_frame() {
    printf("\n=== Testing Single Frame Forward ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTJoint joint;
    joint.load_weights(weights);

    // Create test inputs
    TensorF enc_out({1, 1024});  // [batch=1, enc_dim=1024]
    TensorF dec_out({1, 640});   // [batch=1, dec_dim=640]

    // Fill with small random-ish values
    for (size_t i = 0; i < enc_out.numel(); i++) {
        enc_out.data[i] = std::sin((float)i * 0.01f) * 0.5f;
    }
    for (size_t i = 0; i < dec_out.numel(); i++) {
        dec_out.data[i] = std::cos((float)i * 0.01f) * 0.5f;
    }

    TensorF logits;
    joint.forward(enc_out, dec_out, logits);

    if (logits.shape[0] != 1 || logits.shape[1] != RNNTJoint::VOCAB_SIZE) {
        printf("FAIL: Expected shape [1, 1025], got [%zu, %zu]\n",
               logits.shape[0], logits.shape[1]);
        return;
    }

    if (!check_valid(logits, "logits")) return;

    printf("Logits shape: [%zu, %zu]\n", logits.shape[0], logits.shape[1]);
    print_stats(logits, "Logits");

    // Check argmax
    int best_token = 0;
    float best_score = logits(0, 0);
    for (size_t i = 1; i < RNNTJoint::VOCAB_SIZE; i++) {
        if (logits(0, i) > best_score) {
            best_score = logits(0, i);
            best_token = i;
        }
    }
    printf("Argmax token: %d (score: %.4f)\n", best_token, best_score);

    printf("OK: Single frame forward\n");
}

// ============================================================================
// Test with 3D encoder input
// ============================================================================
void test_3d_encoder() {
    printf("\n=== Testing 3D Encoder Input ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTJoint joint;
    joint.load_weights(weights);

    // Create test inputs with 3D encoder [batch, time=1, dim]
    TensorF enc_out({1, 1, 1024});
    TensorF dec_out({1, 640});

    for (size_t i = 0; i < enc_out.numel(); i++) {
        enc_out.data[i] = std::sin((float)i * 0.01f) * 0.5f;
    }
    for (size_t i = 0; i < dec_out.numel(); i++) {
        dec_out.data[i] = std::cos((float)i * 0.01f) * 0.5f;
    }

    TensorF logits;
    joint.forward(enc_out, dec_out, logits);

    if (!check_valid(logits, "logits")) return;

    printf("Logits shape: [%zu, %zu]\n", logits.shape[0], logits.shape[1]);
    print_stats(logits, "Logits");

    printf("OK: 3D encoder input\n");
}

// ============================================================================
// Test full sequence forward
// ============================================================================
void test_full_sequence() {
    printf("\n=== Testing Full Sequence Forward ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTJoint joint;
    joint.load_weights(weights);

    // Create test inputs
    size_t time = 10;
    TensorF enc_out({1, time, 1024});  // [batch=1, time=10, enc_dim=1024]
    TensorF dec_out({1, 640});          // [batch=1, dec_dim=640]

    for (size_t i = 0; i < enc_out.numel(); i++) {
        enc_out.data[i] = std::sin((float)i * 0.001f) * 0.5f;
    }
    for (size_t i = 0; i < dec_out.numel(); i++) {
        dec_out.data[i] = std::cos((float)i * 0.01f) * 0.5f;
    }

    TensorF logits;
    joint.forward_full(enc_out, dec_out, logits);

    if (logits.shape[0] != 1 || logits.shape[1] != time || logits.shape[2] != RNNTJoint::VOCAB_SIZE) {
        printf("FAIL: Expected shape [1, %zu, 1025], got [%zu, %zu, %zu]\n",
               time, logits.shape[0], logits.shape[1], logits.shape[2]);
        return;
    }

    if (!check_valid(logits, "logits")) return;

    printf("Logits shape: [%zu, %zu, %zu]\n", logits.shape[0], logits.shape[1], logits.shape[2]);
    print_stats(logits, "Logits");

    // Print argmax for each time step
    printf("Argmax tokens per frame: ");
    for (size_t t = 0; t < time; t++) {
        int best_token = 0;
        float best_score = logits(0, t, 0);
        for (size_t v = 1; v < RNNTJoint::VOCAB_SIZE; v++) {
            if (logits(0, t, v) > best_score) {
                best_score = logits(0, t, v);
                best_token = v;
            }
        }
        printf("%d ", best_token);
    }
    printf("\n");

    printf("OK: Full sequence forward\n");
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

    RNNTJoint joint;
    joint.load_weights(weights);

    size_t batch = 4;
    TensorF enc_out({batch, 1024});
    TensorF dec_out({batch, 640});

    for (size_t i = 0; i < enc_out.numel(); i++) {
        enc_out.data[i] = std::sin((float)i * 0.001f) * 0.5f;
    }
    for (size_t i = 0; i < dec_out.numel(); i++) {
        dec_out.data[i] = std::cos((float)i * 0.001f) * 0.5f;
    }

    TensorF logits;
    joint.forward(enc_out, dec_out, logits);

    if (logits.shape[0] != batch || logits.shape[1] != RNNTJoint::VOCAB_SIZE) {
        printf("FAIL: Expected shape [%zu, 1025], got [%zu, %zu]\n",
               batch, logits.shape[0], logits.shape[1]);
        return;
    }

    if (!check_valid(logits, "logits")) return;

    printf("Logits shape: [%zu, %zu]\n", logits.shape[0], logits.shape[1]);
    print_stats(logits, "Logits");

    printf("OK: Batch processing\n");
}

int main() {
    printf("=== Testing RNNT Joint ===\n");

    test_single_frame();
    test_3d_encoder();
    test_full_sequence();
    test_batch();

    printf("\n=== All RNNT Joint tests complete ===\n");
    return 0;
}
