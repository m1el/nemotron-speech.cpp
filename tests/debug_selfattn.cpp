// Debug self-attention step by step
#include "conformer_encoder.h"
#include "ggml_weights.h"

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

    const size_t D_MODEL = 1024;
    const size_t time = 251;

    // Create dummy input after FFN1 residual
    // Using values that match what we'd get after FFN1
    TensorF ffn1_residual({1, time, D_MODEL});
    
    // Load subsample and apply FFN1 to get the residual
    std::ifstream sub_file("cpp_subsampling_debug.bin", std::ios::binary);
    TensorF input({1, time, D_MODEL});
    sub_file.read(reinterpret_cast<char*>(input.data.data()), time * D_MODEL * sizeof(float));
    sub_file.close();

    // Get FFN1 weights and apply
    auto& norm_ff1_weight = weights.require("encoder.layers.0.norm_feed_forward1.weight");
    auto& norm_ff1_bias = weights.require("encoder.layers.0.norm_feed_forward1.bias");
    auto& ffn1_linear1_weight = weights.require("encoder.layers.0.feed_forward1.linear1.weight");
    auto& ffn1_linear2_weight = weights.require("encoder.layers.0.feed_forward1.linear2.weight");

    TensorF buf1, buf2;
    layer_norm(input, norm_ff1_weight.data.data(), norm_ff1_bias.data.data(), D_MODEL, 1e-5f, buf1);
    linear_no_bias(buf1, ffn1_linear1_weight.data.data(), 4096, D_MODEL, buf2);
    swish_inplace(buf2);
    linear_no_bias(buf2, ffn1_linear2_weight.data.data(), D_MODEL, 4096, buf1);
    
    // FFN1 residual
    ffn1_residual = input;
    for (size_t i = 0; i < input.numel(); i++) {
        ffn1_residual.data[i] += 0.5f * buf1.data[i];
    }
    printf("FFN1 residual [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ffn1_residual(0, 0, 0), ffn1_residual(0, 0, 1), ffn1_residual(0, 0, 2),
           ffn1_residual(0, 0, 3), ffn1_residual(0, 0, 4));

    // Get self-attention norm weights
    auto& norm_attn_weight = weights.require("encoder.layers.0.norm_self_att.weight");
    auto& norm_attn_bias = weights.require("encoder.layers.0.norm_self_att.bias");

    // Apply norm
    TensorF attn_input;
    layer_norm(ffn1_residual, norm_attn_weight.data.data(), norm_attn_bias.data.data(), 
               D_MODEL, 1e-5f, attn_input);
    printf("\nAfter norm_self_att [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           attn_input(0, 0, 0), attn_input(0, 0, 1), attn_input(0, 0, 2),
           attn_input(0, 0, 3), attn_input(0, 0, 4));
    printf("NeMo reference: -0.010288, 0.002577, -0.019723, -0.035143, 0.019097\n");

    // Get positional encoding
    RelPositionalEncoding pos_enc;
    pos_enc.init();
    TensorF pos_emb;
    pos_enc.get_pos_emb(time, pos_emb);

    // Run self-attention
    RelPositionMultiHeadAttention self_attn;
    self_attn.load_weights(weights, "encoder.layers.0.self_attn");

    TensorF attn_output;
    self_attn.forward(attn_input, pos_emb, attn_output);

    printf("\nAfter self_attn [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           attn_output(0, 0, 0), attn_output(0, 0, 1), attn_output(0, 0, 2),
           attn_output(0, 0, 3), attn_output(0, 0, 4));
    printf("NeMo reference: -9.492669, 41.225033, -26.114357, 36.293732, 1.978500\n");

    return 0;
}
