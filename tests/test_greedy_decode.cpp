#include "greedy_decode.h"
#include "ggml_weights.h"

#include <cmath>
#include <cstdio>
#include <chrono>

using namespace nemo;

// ============================================================================
// Test greedy decoder with synthetic encoder output
// ============================================================================
void test_synthetic() {
    printf("\n=== Testing Greedy Decode (Synthetic) ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    RNNTDecoder decoder;
    RNNTJoint joint;
    GreedyDecoder greedy;

    decoder.load_weights(weights);
    joint.load_weights(weights);
    greedy.init(&decoder, &joint);

    // Create synthetic encoder output: [1, 10, 1024]
    size_t time = 10;
    TensorF enc_out({1, time, 1024});
    for (size_t i = 0; i < enc_out.numel(); i++) {
        enc_out.data[i] = std::sin((float)i * 0.001f) * 0.1f;
    }

    printf("Encoder output shape: [1, %zu, 1024]\n", time);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> tokens = greedy.decode(enc_out);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Decoded %zu tokens in %ld ms\n", tokens.size(), duration.count());
    printf("Tokens: ");
    for (int t : tokens) {
        printf("%d ", t);
    }
    printf("\n");

    printf("OK: Greedy decode (synthetic)\n");
}

// ============================================================================
// Test full ASR pipeline with synthetic mel
// ============================================================================
void test_full_pipeline() {
    printf("\n=== Testing Full ASR Pipeline ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ASRPipeline asr;
    printf("Loading ASR pipeline...\n");
    asr.load_weights(weights);
    printf("Loaded.\n");

    // Create synthetic mel features: [1, 128, 128]
    // 128 frames of 128 mel bins
    size_t mel_frames = 128;
    TensorF mel({1, mel_frames, 128});
    for (size_t i = 0; i < mel.numel(); i++) {
        mel.data[i] = std::sin((float)i * 0.0001f) * 0.5f;
    }

    printf("Mel features shape: [1, %zu, 128]\n", mel_frames);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> tokens = asr.transcribe(mel);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Transcribed %zu tokens in %ld ms\n", tokens.size(), duration.count());
    printf("Tokens: ");
    for (int t : tokens) {
        printf("%d ", t);
    }
    printf("\n");

    printf("OK: Full ASR pipeline\n");
}

// ============================================================================
// Test with longer sequence
// ============================================================================
void test_long_sequence() {
    printf("\n=== Testing Long Sequence ===\n");

    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        printf("FAIL: Could not load weights\n");
        return;
    }

    ASRPipeline asr;
    asr.load_weights(weights);

    // 5 seconds of audio at 100 fps = 500 frames
    size_t mel_frames = 500;
    TensorF mel({1, mel_frames, 128});
    for (size_t i = 0; i < mel.numel(); i++) {
        mel.data[i] = std::sin((float)i * 0.0001f) * 0.5f;
    }

    printf("Mel features shape: [1, %zu, 128]\n", mel_frames);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> tokens = asr.transcribe(mel);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Transcribed %zu tokens in %ld ms\n", tokens.size(), duration.count());

    // Just print first 20 tokens
    printf("First 20 tokens: ");
    for (size_t i = 0; i < std::min(tokens.size(), (size_t)20); i++) {
        printf("%d ", tokens[i]);
    }
    if (tokens.size() > 20) printf("...");
    printf("\n");

    printf("OK: Long sequence\n");
}

int main() {
    printf("=== Testing Greedy Decode ===\n");

    test_synthetic();
    test_full_pipeline();
    test_long_sequence();

    printf("\n=== All Greedy Decode tests complete ===\n");
    return 0;
}
