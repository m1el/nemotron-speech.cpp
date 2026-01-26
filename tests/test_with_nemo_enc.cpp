// Test C++ decoder/joint with NeMo encoder output
#include "greedy_decode.h"
#include "ggml_weights.h"
#include "tokenizer.h"

#include <cstdio>
#include <fstream>

using namespace nemo;

int main() {
    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Load NeMo encoder output
    std::ifstream enc_file("nemo_encoder_correct.bin", std::ios::binary);
    enc_file.seekg(0, std::ios::end);
    size_t enc_size = enc_file.tellg() / sizeof(float);
    enc_file.seekg(0, std::ios::beg);

    TensorF enc_out({1, enc_size / 1024, 1024});
    enc_file.read(reinterpret_cast<char*>(enc_out.data.data()), enc_size * sizeof(float));
    printf("Loaded NeMo encoder output: [%zu, %zu, %zu]\n", enc_out.shape[0], enc_out.shape[1], enc_out.shape[2]);

    // Load decoder and joint
    RNNTDecoder decoder;
    decoder.load_weights(weights);

    RNNTJoint joint;
    joint.load_weights(weights);

    // Greedy decode
    GreedyDecoder greedy;
    greedy.init(&decoder, &joint);

    printf("Running greedy decode...\n");
    std::vector<int> tokens = greedy.decode(enc_out);

    printf("Decoded %zu tokens\n", tokens.size());
    printf("First 20 tokens: ");
    for (size_t i = 0; i < std::min(tokens.size(), (size_t)20); i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    // Load tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load("vocab.txt")) {
        fprintf(stderr, "Failed to load vocab\n");
        return 1;
    }

    std::string text = tokenizer.decode(tokens);
    printf("Text: %s\n", text.c_str());

    return 0;
}
