// Debug first Conformer layer step by step
#include "conformer_encoder.h"
#include "ggml_weights.h"
#include "ops.h"

#include <cstdio>
#include <fstream>
#include <algorithm>

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

int main(int argc, char** argv) {
    const char* mel_path = argc > 1 ? argv[1] : "test.mel.bin";

    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Load mel
    TensorF mel;
    if (!load_mel_bin(mel_path, mel)) {
        fprintf(stderr, "Failed to load mel\n");
        return 1;
    }
    printf("Mel shape: [%zu, %zu, %zu]\n", mel.shape[0], mel.shape[1], mel.shape[2]);

    // Load subsampling
    ConvSubsampling subsample;
    subsample.load_weights(weights);

    // Run subsampling
    TensorF sub_out;
    subsample.forward(mel, sub_out);
    printf("Subsampling output: [%zu, %zu, %zu]\n", sub_out.shape[0], sub_out.shape[1], sub_out.shape[2]);
    printf("Subsampling first 5: %.6f %.6f %.6f %.6f %.6f\n",
        sub_out(0,0,0), sub_out(0,0,1), sub_out(0,0,2), sub_out(0,0,3), sub_out(0,0,4));

    // Load positional encoding
    RelPositionalEncoding pos_enc;
    pos_enc.load_weights(weights);

    // Apply positional encoding
    TensorF pos_out, pos_emb;
    pos_enc.forward(sub_out, pos_out, pos_emb);
    printf("\nAfter pos encoding: [%zu, %zu, %zu]\n", pos_out.shape[0], pos_out.shape[1], pos_out.shape[2]);
    printf("First 5: %.6f %.6f %.6f %.6f %.6f\n",
        pos_out(0,0,0), pos_out(0,0,1), pos_out(0,0,2), pos_out(0,0,3), pos_out(0,0,4));

    // Load first layer
    ConformerLayer layer0;
    layer0.load_weights(weights, 0);

    // Process through layer 0 step by step
    size_t batch = pos_out.shape[0];
    size_t time = pos_out.shape[1];
    size_t dim = pos_out.shape[2];

    TensorF x = pos_out;
    TensorF x_norm, ff_out, attn_out, conv_out;

    // FF1
    layer_norm(x, layer0.norm_ff1_weight, layer0.norm_ff1_bias, dim, x_norm);
    layer0.ff1.forward(x_norm, ff_out);
    // x = x + 0.5 * ff_out
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] = x.data[i] + 0.5f * ff_out.data[i];
    }
    printf("\nAfter FF1: first 5 = %.6f %.6f %.6f %.6f %.6f\n",
        x(0,0,0), x(0,0,1), x(0,0,2), x(0,0,3), x(0,0,4));

    // Self-attention
    layer_norm(x, layer0.norm_attn_weight, layer0.norm_attn_bias, dim, x_norm);
    layer0.self_attn.forward(x_norm, x_norm, x_norm, pos_emb, attn_out);
    // x = x + attn_out
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] = x.data[i] + attn_out.data[i];
    }
    printf("After Attn: first 5 = %.6f %.6f %.6f %.6f %.6f\n",
        x(0,0,0), x(0,0,1), x(0,0,2), x(0,0,3), x(0,0,4));

    // Conv
    layer_norm(x, layer0.norm_conv_weight, layer0.norm_conv_bias, dim, x_norm);
    layer0.conv.forward(x_norm, conv_out);
    // x = x + conv_out
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] = x.data[i] + conv_out.data[i];
    }
    printf("After Conv: first 5 = %.6f %.6f %.6f %.6f %.6f\n",
        x(0,0,0), x(0,0,1), x(0,0,2), x(0,0,3), x(0,0,4));

    // FF2
    layer_norm(x, layer0.norm_ff2_weight, layer0.norm_ff2_bias, dim, x_norm);
    layer0.ff2.forward(x_norm, ff_out);
    // x = x + 0.5 * ff_out
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] = x.data[i] + 0.5f * ff_out.data[i];
    }
    printf("After FF2: first 5 = %.6f %.6f %.6f %.6f %.6f\n",
        x(0,0,0), x(0,0,1), x(0,0,2), x(0,0,3), x(0,0,4));

    // Output norm
    layer_norm(x, layer0.norm_out_weight, layer0.norm_out_bias, dim, x_norm);
    printf("After norm_out: first 5 = %.6f %.6f %.6f %.6f %.6f\n",
        x_norm(0,0,0), x_norm(0,0,1), x_norm(0,0,2), x_norm(0,0,3), x_norm(0,0,4));

    // Save
    std::ofstream out("cpp_layer0_out.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(x_norm.data.data()), x_norm.numel() * sizeof(float));
    printf("Saved cpp_layer0_out.bin: [%zu, %zu, %zu]\n", x_norm.shape[0], x_norm.shape[1], x_norm.shape[2]);

    return 0;
}
