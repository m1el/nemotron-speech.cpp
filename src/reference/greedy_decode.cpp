#include "include/greedy_decode.h"

namespace nemo {

std::vector<int> GreedyDecoder::decode(const TensorF& enc_out) {
    std::vector<int> tokens;

    if (enc_out.shape[0] != 1) {
        // Only batch size 1 supported for now
        return tokens;
    }

    size_t time = enc_out.shape[1];
    size_t enc_dim = enc_out.shape[2];

    // Initialize decoder with blank token
    decoder->init_state(1);
    int last_token = BLANK_TOKEN;
    decoder->forward_step(last_token, dec_out_);

    // Process each encoder frame
    for (size_t t = 0; t < time; t++) {
        // Extract single encoder frame: [1, enc_dim]
        enc_frame_.resize({1, enc_dim});
        for (size_t d = 0; d < enc_dim; d++) {
            enc_frame_(0, d) = enc_out(0, t, d);
        }

        // Inner loop: emit symbols until blank
        for (size_t sym = 0; sym < MAX_SYMBOLS_PER_STEP; sym++) {
            // Compute joint logits
            joint->forward(enc_frame_, dec_out_, logits_);

            // Argmax
            int best_token = 0;
            float best_score = logits_(0, 0);
            for (size_t v = 1; v < RNNTJoint::VOCAB_SIZE; v++) {
                if (logits_(0, v) > best_score) {
                    best_score = logits_(0, v);
                    best_token = v;
                }
            }

            if (best_token == BLANK_TOKEN) {
                // Move to next encoder frame
                break;
            }

            // Emit non-blank token
            tokens.push_back(best_token);
            last_token = best_token;

            // Update decoder state
            decoder->forward_step(last_token, dec_out_);
        }
    }

    return tokens;
}

void ASRPipeline::load_weights(const ModelWeights& weights) {
    encoder.load_weights(weights);
    decoder.load_weights(weights);
    joint.load_weights(weights);
    greedy.init(&decoder, &joint);
}

std::vector<int> ASRPipeline::transcribe(const TensorF& mel) {
    // Encode
    encoder.forward(mel, enc_out_);

    // Greedy decode
    return greedy.decode(enc_out_);
}

}  // namespace nemo
