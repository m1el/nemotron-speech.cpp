#include "rnnt_decoder.h"

namespace nemo {

void RNNTDecoder::load_weights(const ModelWeights& weights) {
    // Load embedding
    embed_weight = weights.require("decoder.prediction.embed.weight").data.data();

    // Load LSTM weights for each layer
    for (size_t i = 0; i < NUM_LAYERS; i++) {
        std::string prefix = "decoder.prediction.dec_rnn.lstm.";
        std::string layer = "l" + std::to_string(i);

        lstm_weight_ih[i] = weights.require(prefix + "weight_ih_" + layer).data.data();
        lstm_weight_hh[i] = weights.require(prefix + "weight_hh_" + layer).data.data();
        lstm_bias_ih[i] = weights.require(prefix + "bias_ih_" + layer).data.data();
        lstm_bias_hh[i] = weights.require(prefix + "bias_hh_" + layer).data.data();
    }
}

void RNNTDecoder::init_state(size_t batch_size) {
    // Initialize h and c to zeros for all layers
    state.h.resize({NUM_LAYERS, batch_size, HIDDEN_SIZE});
    state.c.resize({NUM_LAYERS, batch_size, HIDDEN_SIZE});
    state.h.zero();
    state.c.zero();
}

void RNNTDecoder::forward_step(int token, TensorF& output) {
    // Single token forward
    int tokens[1] = {token};
    forward(tokens, 1, output);
}

void RNNTDecoder::forward(const int* tokens, size_t batch_size, TensorF& output) {
    // Embedding lookup: [batch] -> [batch, 640]
    embedding(tokens, batch_size, embed_weight, VOCAB_SIZE, EMBED_DIM, embed_out_);

    // Process through LSTM layers
    lstm_forward(
        embed_out_,
        state,
        lstm_weight_ih,
        lstm_weight_hh,
        lstm_bias_ih,
        lstm_bias_hh,
        EMBED_DIM,  // input_size (same as hidden for layer 1)
        HIDDEN_SIZE,
        NUM_LAYERS,
        output
    );
}

}  // namespace nemo
