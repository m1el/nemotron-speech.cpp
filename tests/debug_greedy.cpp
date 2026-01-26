// Debug greedy decoding step by step
#include "greedy_decode.h"
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
    enc_file.seekg(0, std::ios::end);
    size_t enc_size = enc_file.tellg() / sizeof(float);
    enc_file.seekg(0, std::ios::beg);

    TensorF enc_out({1, enc_size / 1024, 1024});
    enc_file.read(reinterpret_cast<char*>(enc_out.data.data()), enc_size * sizeof(float));
    printf("Loaded encoder output: [%zu, %zu, %zu]\n", enc_out.shape[0], enc_out.shape[1], enc_out.shape[2]);

    size_t time = enc_out.shape[1];
    size_t enc_dim = enc_out.shape[2];

    // Load decoder and joint
    RNNTDecoder decoder;
    decoder.load_weights(weights);

    RNNTJoint joint;
    joint.load_weights(weights);

    const int BLANK_TOKEN = 1024;

    // Initialize decoder
    decoder.init_state(1);
    int last_token = BLANK_TOKEN;
    TensorF dec_out;
    decoder.forward_step(last_token, dec_out);
    printf("Initialized decoder with blank token\n");
    printf("Dec out first 5: %.6f %.6f %.6f %.6f %.6f\n",
        dec_out(0,0), dec_out(0,1), dec_out(0,2), dec_out(0,3), dec_out(0,4));

    std::vector<int> tokens;
    TensorF enc_frame({1, enc_dim});
    TensorF logits;

    // Process first 10 encoder frames for debugging
    for (size_t t = 0; t < std::min(time, (size_t)10); t++) {
        // Extract encoder frame
        for (size_t d = 0; d < enc_dim; d++) {
            enc_frame(0, d) = enc_out(0, t, d);
        }

        printf("\n=== Frame %zu ===\n", t);
        printf("Enc frame first 5: %.6f %.6f %.6f %.6f %.6f\n",
            enc_frame(0,0), enc_frame(0,1), enc_frame(0,2), enc_frame(0,3), enc_frame(0,4));

        // Inner loop
        for (size_t sym = 0; sym < 10; sym++) {
            // Compute joint
            joint.forward(enc_frame, dec_out, logits);

            // Argmax
            int best_token = 0;
            float best_score = logits(0, 0);
            for (size_t v = 1; v < 1025; v++) {
                if (logits(0, v) > best_score) {
                    best_score = logits(0, v);
                    best_token = v;
                }
            }

            printf("  Sym %zu: argmax=%d (score=%.4f), blank_score=%.4f\n",
                sym, best_token, best_score, logits(0, BLANK_TOKEN));

            if (best_token == BLANK_TOKEN) {
                printf("  -> blank, moving to next frame\n");
                break;
            }

            // Emit non-blank
            tokens.push_back(best_token);
            last_token = best_token;
            printf("  -> emit token %d\n", best_token);

            // Update decoder
            decoder.forward_step(last_token, dec_out);
            printf("  Dec out after update first 5: %.6f %.6f %.6f %.6f %.6f\n",
                dec_out(0,0), dec_out(0,1), dec_out(0,2), dec_out(0,3), dec_out(0,4));
        }
    }

    printf("\n=== Tokens emitted ===\n");
    printf("Count: %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    return 0;
}
