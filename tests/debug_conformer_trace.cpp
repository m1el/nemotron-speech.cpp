// Debug C++ conformer layer step by step
#include "conformer_encoder.h"
#include "ggml_weights.h"

#include <cstdio>
#include <fstream>

using namespace nemo;

void print_first5(const TensorF& t, const char* name) {
    printf("%s [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           name, t(0, 0, 0), t(0, 0, 1), t(0, 0, 2), t(0, 0, 3), t(0, 0, 4));
}

int main() {
    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Load C++ subsample output
    std::ifstream sub_file("cpp_subsampling_debug.bin", std::ios::binary);
    sub_file.seekg(0, std::ios::end);
    size_t sub_size = sub_file.tellg() / sizeof(float);
    sub_file.seekg(0, std::ios::beg);

    size_t time = sub_size / 1024;
    TensorF input({1, time, 1024});
    sub_file.read(reinterpret_cast<char*>(input.data.data()), sub_size * sizeof(float));
    sub_file.close();

    printf("Loaded subsample: [1, %zu, 1024]\n", time);
    print_first5(input, "Input");

    // Get positional encoding
    RelPositionalEncoding pos_enc;
    pos_enc.init();
    TensorF pos_emb;
    pos_enc.get_pos_emb(time, pos_emb);
    printf("\nPos emb shape: [%zu, %zu]\n", pos_emb.shape[0], pos_emb.shape[1]);

    // Load layer 0 weights
    const size_t D_MODEL = 1024;
    
    // Get layer norm weights
    auto& norm_ff1_weight = weights.require("encoder.layers.0.norm_feed_forward1.weight");
    auto& norm_ff1_bias = weights.require("encoder.layers.0.norm_feed_forward1.bias");
    
    // Get FFN1 weights
    auto& ffn1_linear1_weight = weights.require("encoder.layers.0.feed_forward1.linear1.weight");
    auto& ffn1_linear2_weight = weights.require("encoder.layers.0.feed_forward1.linear2.weight");

    printf("\n=== Tracing Layer 0 ===\n");

    // buf1 = input (residual)
    TensorF buf1 = input;
    TensorF buf2, buf3;
    
    // 1. LayerNorm -> FFN1
    layer_norm(buf1, norm_ff1_weight.data.data(), norm_ff1_bias.data.data(), 
               D_MODEL, 1e-5f, buf2);
    print_first5(buf2, "After norm_feed_forward1");
    
    // FFN1: Linear1 -> Swish -> Linear2
    TensorF ffn_buf;
    linear_no_bias(buf2, ffn1_linear1_weight.data.data(), 4096, D_MODEL, ffn_buf);
    printf("After FFN1 linear1 [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ffn_buf(0, 0, 0), ffn_buf(0, 0, 1), ffn_buf(0, 0, 2), 
           ffn_buf(0, 0, 3), ffn_buf(0, 0, 4));
    
    swish_inplace(ffn_buf);
    printf("After FFN1 swish [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ffn_buf(0, 0, 0), ffn_buf(0, 0, 1), ffn_buf(0, 0, 2), 
           ffn_buf(0, 0, 3), ffn_buf(0, 0, 4));
    
    linear_no_bias(ffn_buf, ffn1_linear2_weight.data.data(), D_MODEL, 4096, buf3);
    print_first5(buf3, "After feed_forward1");

    // Residual: buf1 = buf1 + 0.5 * buf3
    for (size_t i = 0; i < buf1.numel(); i++) {
        buf1.data[i] += 0.5f * buf3.data[i];
    }
    print_first5(buf1, "After FFN1 residual");

    printf("\n=== NeMo Reference Values ===\n");
    printf("After norm_feed_forward1: -0.040058, 0.086292, -0.145103, -0.489813, 0.188412\n");
    printf("After feed_forward1: -1.700375, -48.296082, -18.395834, 22.160685, 53.329433\n");
    printf("After FFN1 residual: -32.944389, 3.455408, -59.504982, -141.510345, 98.848396\n");

    return 0;
}
