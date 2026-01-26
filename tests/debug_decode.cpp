// Debug decoder and joint outputs
#include "rnnt_decoder.h"
#include "rnnt_joint.h"
#include "ggml_weights.h"
#include "ops.h"

#include <cstdio>
#include <fstream>
#include <algorithm>

using namespace nemo;

int main() {
    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Load encoder output
    std::ifstream enc_file("cpp_encoder_out.bin", std::ios::binary);
    if (!enc_file.is_open()) {
        fprintf(stderr, "Failed to load cpp_encoder_out.bin\n");
        return 1;
    }
    enc_file.seekg(0, std::ios::end);
    size_t enc_size = enc_file.tellg() / sizeof(float);
    enc_file.seekg(0, std::ios::beg);

    TensorF enc_out({1, enc_size / 1024, 1024});
    enc_file.read(reinterpret_cast<char*>(enc_out.data.data()), enc_size * sizeof(float));
    printf("Loaded encoder output: [%zu, %zu, %zu]\n", enc_out.shape[0], enc_out.shape[1], enc_out.shape[2]);
    printf("Enc frame 0 first 5: %.6f %.6f %.6f %.6f %.6f\n",
        enc_out(0,0,0), enc_out(0,0,1), enc_out(0,0,2), enc_out(0,0,3), enc_out(0,0,4));

    // Load decoder and joint
    RNNTDecoder decoder;
    decoder.load_weights(weights);

    RNNTJoint joint;
    joint.load_weights(weights);

    // Initialize decoder
    decoder.init_state(1);

    // Run decoder with blank token
    int blank_id = 1024;
    TensorF dec_out;
    decoder.forward_step(blank_id, dec_out);
    printf("\nDecoder output shape: [%zu, %zu]\n", dec_out.shape[0], dec_out.shape[1]);
    printf("Decoder output first 5: %.6f %.6f %.6f %.6f %.6f\n",
        dec_out(0,0), dec_out(0,1), dec_out(0,2), dec_out(0,3), dec_out(0,4));

    // Compare with NeMo decoder output
    std::ifstream nemo_dec_file("nemo_decoder_out_0.bin", std::ios::binary);
    if (nemo_dec_file.is_open()) {
        std::vector<float> nemo_dec(640);
        nemo_dec_file.read(reinterpret_cast<char*>(nemo_dec.data()), 640 * sizeof(float));
        printf("NeMo decoder first 5: %.6f %.6f %.6f %.6f %.6f\n",
            nemo_dec[0], nemo_dec[1], nemo_dec[2], nemo_dec[3], nemo_dec[4]);

        float max_diff = 0;
        for (size_t i = 0; i < 640; i++) {
            max_diff = std::max(max_diff, std::abs(dec_out(0, i) - nemo_dec[i]));
        }
        printf("Max decoder diff: %.6f\n", max_diff);
    }

    // Extract first encoder frame
    TensorF enc_frame({1, 1024});
    for (size_t i = 0; i < 1024; i++) {
        enc_frame(0, i) = enc_out(0, 0, i);
    }

    // Run joint
    TensorF logits;
    joint.forward(enc_frame, dec_out, logits);
    printf("\nJoint output shape: [%zu, %zu]\n", logits.shape[0], logits.shape[1]);
    printf("Joint logits first 5: %.6f %.6f %.6f %.6f %.6f\n",
        logits(0,0), logits(0,1), logits(0,2), logits(0,3), logits(0,4));
    printf("Joint logits last 5: %.6f %.6f %.6f %.6f %.6f\n",
        logits(0,1020), logits(0,1021), logits(0,1022), logits(0,1023), logits(0,1024));

    // Compare with NeMo joint output
    std::ifstream nemo_joint_file("nemo_joint_logits_0.bin", std::ios::binary);
    if (nemo_joint_file.is_open()) {
        std::vector<float> nemo_joint(1025);
        nemo_joint_file.read(reinterpret_cast<char*>(nemo_joint.data()), 1025 * sizeof(float));
        printf("\nNeMo joint logits first 5: %.6f %.6f %.6f %.6f %.6f\n",
            nemo_joint[0], nemo_joint[1], nemo_joint[2], nemo_joint[3], nemo_joint[4]);
        printf("NeMo joint logits last 5: %.6f %.6f %.6f %.6f %.6f\n",
            nemo_joint[1020], nemo_joint[1021], nemo_joint[1022], nemo_joint[1023], nemo_joint[1024]);

        float max_diff = 0;
        for (size_t i = 0; i < 1025; i++) {
            max_diff = std::max(max_diff, std::abs(logits(0, i) - nemo_joint[i]));
        }
        printf("Max joint diff: %.6f\n", max_diff);
    }

    // Argmax
    int best_token = 0;
    float best_score = logits(0, 0);
    for (size_t i = 1; i < 1025; i++) {
        if (logits(0, i) > best_score) {
            best_score = logits(0, i);
            best_token = i;
        }
    }
    printf("\nArgmax: %d (blank=%d)\n", best_token, blank_id);

    return 0;
}
