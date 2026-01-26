// Debug decoder state handling
#include "rnnt_decoder.h"
#include "rnnt_joint.h"
#include "ggml_weights.h"

#include <cstdio>
#include <fstream>

using namespace nemo;

int main() {
    ModelWeights weights;
    weights.load("weights/model.bin");

    // Load NeMo encoder output
    std::ifstream enc_file("nemo_encoder_correct.bin", std::ios::binary);
    enc_file.seekg(0, std::ios::end);
    size_t enc_size = enc_file.tellg() / sizeof(float);
    enc_file.seekg(0, std::ios::beg);
    TensorF enc_out({1, enc_size / 1024, 1024});
    enc_file.read(reinterpret_cast<char*>(enc_out.data.data()), enc_size * sizeof(float));

    RNNTDecoder decoder;
    decoder.load_weights(weights);

    RNNTJoint joint;
    joint.load_weights(weights);

    int blank_id = 1024;
    TensorF dec_out, logits;
    TensorF enc_frame({1, 1024});

    // Initialize
    decoder.init_state(1);
    decoder.forward_step(blank_id, dec_out);
    
    printf("Initial dec_out first 5: %.6f %.6f %.6f %.6f %.6f\n",
        dec_out(0,0), dec_out(0,1), dec_out(0,2), dec_out(0,3), dec_out(0,4));

    // Frame 0
    for (size_t d = 0; d < 1024; d++) enc_frame(0, d) = enc_out(0, 0, d);
    joint.forward(enc_frame, dec_out, logits);
    
    int best = 0;
    for (size_t v = 1; v < 1025; v++) {
        if (logits(0, v) > logits(0, best)) best = v;
    }
    printf("Frame 0: best=%d, score=%.4f, blank=%.4f\n", best, logits(0, best), logits(0, blank_id));

    // Frame 1 - WITHOUT re-running decoder (state maintained)
    for (size_t d = 0; d < 1024; d++) enc_frame(0, d) = enc_out(0, 1, d);
    joint.forward(enc_frame, dec_out, logits);
    
    best = 0;
    for (size_t v = 1; v < 1025; v++) {
        if (logits(0, v) > logits(0, best)) best = v;
    }
    printf("Frame 1 (same dec_out): best=%d, score=%.4f, blank=%.4f\n", best, logits(0, best), logits(0, blank_id));

    // If frame 0 was blank, re-run decoder and check
    if (true) {
        decoder.init_state(1);  // Reset
        decoder.forward_step(blank_id, dec_out);  // Fresh state
        
        // Frame 0
        for (size_t d = 0; d < 1024; d++) enc_frame(0, d) = enc_out(0, 0, d);
        joint.forward(enc_frame, dec_out, logits);
        best = 0;
        for (size_t v = 1; v < 1025; v++) {
            if (logits(0, v) > logits(0, best)) best = v;
        }
        printf("\nFresh run - Frame 0: best=%d\n", best);
        
        // Now Frame 0 was blank, so we DON'T update decoder, just move to frame 1
        // This is what the greedy decode does
        for (size_t d = 0; d < 1024; d++) enc_frame(0, d) = enc_out(0, 1, d);
        joint.forward(enc_frame, dec_out, logits);  // Same dec_out
        best = 0;
        for (size_t v = 1; v < 1025; v++) {
            if (logits(0, v) > logits(0, best)) best = v;
        }
        printf("Fresh run - Frame 1 (same dec_out after blank): best=%d, score=%.4f, blank=%.4f\n", 
            best, logits(0, best), logits(0, blank_id));
    }

    return 0;
}
