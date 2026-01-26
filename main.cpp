// NeMo ASR in C++ - Main entry point
// Usage: ./nemotron-speech <audio.wav> or <mel.bin>

#include "greedy_decode.h"
#include "ggml_weights.h"
#include "tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <vector>

using namespace nemo;

// Load mel features from binary file
// Format: [time, 128] float32, little-endian
bool load_mel_bin(const char* path, TensorF& mel) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate dimensions
    size_t num_floats = file_size / sizeof(float);
    size_t time = num_floats / 128;

    if (time * 128 != num_floats) {
        fprintf(stderr, "Error: Mel file size not divisible by 128 features\n");
        return false;
    }

    mel.resize({1, time, 128});
    file.read(reinterpret_cast<char*>(mel.data.data()), file_size);

    return true;
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <input.mel.bin> [--weights weights/model.bin] [--vocab vocab.txt]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --weights PATH   Path to model weights (default: weights/model.bin)\n");
    fprintf(stderr, "  --vocab PATH     Path to vocab file (default: vocab.txt)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Input should be mel spectrogram in binary format:\n");
    fprintf(stderr, "  - Shape: [time, 128] float32, little-endian\n");
    fprintf(stderr, "  - Can be generated with preprocessor tool\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    const char* input_path = nullptr;
    const char* weights_path = "weights/model.bin";
    const char* vocab_path = "vocab.txt";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (argv[i][0] != '-') {
            input_path = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!input_path) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load model weights
    printf("Loading model from %s...\n", weights_path);
    auto t_start = std::chrono::high_resolution_clock::now();

    ModelWeights weights;
    if (!weights.load(weights_path)) {
        fprintf(stderr, "Error: Failed to load weights from %s\n", weights_path);
        return 1;
    }

    ASRPipeline asr;
    asr.load_weights(weights);

    auto t_load = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load - t_start).count();
    printf("Model loaded in %ld ms\n", load_ms);

    // Load tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load(vocab_path)) {
        fprintf(stderr, "Error: Failed to load vocab from %s\n", vocab_path);
        return 1;
    }
    printf("Loaded vocabulary: %zu tokens\n", tokenizer.size());

    // Load mel features
    printf("Loading mel features from %s...\n", input_path);
    TensorF mel;
    if (!load_mel_bin(input_path, mel)) {
        fprintf(stderr, "Error: Failed to load mel features from %s\n", input_path);
        return 1;
    }
    printf("Mel shape: [1, %zu, 128]\n", mel.shape[1]);

    // Transcribe
    printf("Transcribing...\n");
    auto t_infer_start = std::chrono::high_resolution_clock::now();

    std::vector<int> tokens = asr.transcribe(mel);

    auto t_infer_end = std::chrono::high_resolution_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_infer_end - t_infer_start).count();

    // Decode tokens to text
    std::string text = tokenizer.decode(tokens);

    // Print results
    printf("\n");
    printf("=== Results ===\n");
    printf("Tokens (%zu): ", tokens.size());
    for (size_t i = 0; i < std::min(tokens.size(), (size_t)20); i++) {
        printf("%d ", tokens[i]);
    }
    if (tokens.size() > 20) printf("...");
    printf("\n");

    printf("Text: %s\n", text.c_str());
    printf("\n");
    printf("Inference time: %ld ms\n", infer_ms);

    // Calculate real-time factor
    // Assuming 100 fps mel features (10ms per frame)
    float audio_duration = mel.shape[1] * 0.01f;  // seconds
    float rtf = infer_ms / 1000.0f / audio_duration;
    printf("Audio duration: %.2f s\n", audio_duration);
    printf("Real-time factor: %.2fx\n", rtf);

    return 0;
}
