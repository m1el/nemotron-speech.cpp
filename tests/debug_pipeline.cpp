// Debug full pipeline encoder output
#include "greedy_decode.h"
#include "ggml_weights.h"
#include "ops.h"

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace nemo;

bool load_mel_bin(const char* path, TensorF& mel) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_floats = file_size / sizeof(float);
    size_t time = num_floats / 128;

    mel.resize({1, time, 128});
    file.read(reinterpret_cast<char*>(mel.data.data()), file_size);
    return true;
}

int main() {
    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Load mel
    TensorF mel;
    if (!load_mel_bin("test.mel.bin", mel)) {
        fprintf(stderr, "Failed to load mel\n");
        return 1;
    }
    printf("Mel shape: [%zu, %zu, %zu]\n", mel.shape[0], mel.shape[1], mel.shape[2]);

    // Load full pipeline
    ASRPipeline asr;
    asr.load_weights(weights);

    // Run encoder (internal)
    printf("\nRunning encoder...\n");
    TensorF enc_out;
    asr.encoder.forward(mel, enc_out);

    printf("Pipeline encoder output shape: [%zu, %zu, %zu]\n",
        enc_out.shape[0], enc_out.shape[1], enc_out.shape[2]);
    printf("Pipeline enc first 5: %.6f %.6f %.6f %.6f %.6f\n",
        enc_out(0,0,0), enc_out(0,0,1), enc_out(0,0,2), enc_out(0,0,3), enc_out(0,0,4));
    printf("Stats: min=%.4f, max=%.4f, mean=%.4f\n",
        *std::min_element(enc_out.data.begin(), enc_out.data.end()),
        *std::max_element(enc_out.data.begin(), enc_out.data.end()),
        std::accumulate(enc_out.data.begin(), enc_out.data.end(), 0.0f) / enc_out.numel());

    // Compare with saved encoder output
    std::ifstream saved_file("cpp_encoder_out.bin", std::ios::binary);
    if (saved_file.is_open()) {
        saved_file.seekg(0, std::ios::end);
        size_t saved_size = saved_file.tellg() / sizeof(float);
        saved_file.seekg(0, std::ios::beg);

        TensorF saved_enc({1, saved_size / 1024, 1024});
        saved_file.read(reinterpret_cast<char*>(saved_enc.data.data()), saved_size * sizeof(float));

        printf("\nSaved encoder output shape: [%zu, %zu, %zu]\n",
            saved_enc.shape[0], saved_enc.shape[1], saved_enc.shape[2]);
        printf("Saved enc first 5: %.6f %.6f %.6f %.6f %.6f\n",
            saved_enc(0,0,0), saved_enc(0,0,1), saved_enc(0,0,2), saved_enc(0,0,3), saved_enc(0,0,4));

        if (enc_out.numel() == saved_enc.numel()) {
            float max_diff = 0;
            for (size_t i = 0; i < enc_out.numel(); i++) {
                max_diff = std::max(max_diff, std::abs(enc_out.data[i] - saved_enc.data[i]));
            }
            printf("Max diff between pipeline and saved: %.6f\n", max_diff);
        }
    }

    // Now run greedy decode on pipeline encoder output
    printf("\n=== Running greedy decode on pipeline encoder output ===\n");

    asr.decoder.init_state(1);
    int last_token = 1024;  // blank
    TensorF dec_out;
    asr.decoder.forward_step(last_token, dec_out);

    std::vector<int> tokens;
    TensorF enc_frame({1, 1024});
    TensorF logits;

    size_t time = enc_out.shape[1];

    for (size_t t = 0; t < std::min(time, (size_t)10); t++) {
        for (size_t d = 0; d < 1024; d++) {
            enc_frame(0, d) = enc_out(0, t, d);
        }

        printf("\n=== Frame %zu ===\n", t);
        printf("Enc frame first 5: %.6f %.6f %.6f %.6f %.6f\n",
            enc_frame(0,0), enc_frame(0,1), enc_frame(0,2), enc_frame(0,3), enc_frame(0,4));

        for (size_t sym = 0; sym < 10; sym++) {
            asr.joint.forward(enc_frame, dec_out, logits);

            int best_token = 0;
            float best_score = logits(0, 0);
            for (size_t v = 1; v < 1025; v++) {
                if (logits(0, v) > best_score) {
                    best_score = logits(0, v);
                    best_token = v;
                }
            }

            printf("  Sym %zu: argmax=%d (score=%.4f), blank_score=%.4f\n",
                sym, best_token, best_score, logits(0, 1024));

            if (best_token == 1024) {
                break;
            }

            tokens.push_back(best_token);
            last_token = best_token;
            asr.decoder.forward_step(last_token, dec_out);
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
