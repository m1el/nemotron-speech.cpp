// Debug attention Q/K/V
#include "conformer_encoder.h"
#include "ggml_weights.h"
#include <cstdio>
#include <fstream>

using namespace nemo;

int main() {
    ModelWeights weights;
    weights.load("weights/model.bin");
    
    const size_t D_MODEL = 1024;
    const size_t time = 251;
    
    // Load subsample and compute attn_input (same as before)
    std::ifstream sub_file("cpp_subsampling_debug.bin", std::ios::binary);
    TensorF input({1, time, D_MODEL});
    sub_file.read(reinterpret_cast<char*>(input.data.data()), time * D_MODEL * sizeof(float));
    sub_file.close();
    
    // Compute FFN1 residual
    auto& norm_ff1_weight = weights.require("encoder.layers.0.norm_feed_forward1.weight");
    auto& norm_ff1_bias = weights.require("encoder.layers.0.norm_feed_forward1.bias");
    auto& ffn1_linear1_weight = weights.require("encoder.layers.0.feed_forward1.linear1.weight");
    auto& ffn1_linear2_weight = weights.require("encoder.layers.0.feed_forward1.linear2.weight");
    
    TensorF buf1, buf2;
    layer_norm(input, norm_ff1_weight.data.data(), norm_ff1_bias.data.data(), D_MODEL, 1e-5f, buf1);
    linear_no_bias(buf1, ffn1_linear1_weight.data.data(), 4096, D_MODEL, buf2);
    swish_inplace(buf2);
    linear_no_bias(buf2, ffn1_linear2_weight.data.data(), D_MODEL, 4096, buf1);
    
    TensorF ffn1_residual = input;
    for (size_t i = 0; i < input.numel(); i++) {
        ffn1_residual.data[i] += 0.5f * buf1.data[i];
    }
    
    // Compute attn_input
    auto& norm_attn_weight = weights.require("encoder.layers.0.norm_self_att.weight");
    auto& norm_attn_bias = weights.require("encoder.layers.0.norm_self_att.bias");
    TensorF attn_input;
    layer_norm(ffn1_residual, norm_attn_weight.data.data(), norm_attn_bias.data.data(), 
               D_MODEL, 1e-5f, attn_input);
    printf("attn_input [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           attn_input(0,0,0), attn_input(0,0,1), attn_input(0,0,2), 
           attn_input(0,0,3), attn_input(0,0,4));
    
    // Get Q/K/V weights
    auto& q_weight = weights.require("encoder.layers.0.self_attn.linear_q.weight");
    auto& k_weight = weights.require("encoder.layers.0.self_attn.linear_k.weight");
    auto& v_weight = weights.require("encoder.layers.0.self_attn.linear_v.weight");
    auto& pos_weight = weights.require("encoder.layers.0.self_attn.linear_pos.weight");
    auto& pos_bias_u = weights.require("encoder.layers.0.self_attn.pos_bias_u");
    auto& pos_bias_v = weights.require("encoder.layers.0.self_attn.pos_bias_v");
    
    printf("\nWeights shapes:\n");
    printf("  q_weight: [%zu, %zu]\n", q_weight.shape[0], q_weight.shape[1]);
    printf("  pos_bias_u: [%zu, %zu]\n", pos_bias_u.shape[0], pos_bias_u.shape[1]);
    
    // Compute Q, K, V
    TensorF q, k, v;
    linear_no_bias(attn_input, q_weight.data.data(), D_MODEL, D_MODEL, q);
    linear_no_bias(attn_input, k_weight.data.data(), D_MODEL, D_MODEL, k);
    linear_no_bias(attn_input, v_weight.data.data(), D_MODEL, D_MODEL, v);
    
    printf("\nQ [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           q(0,0,0), q(0,0,1), q(0,0,2), q(0,0,3), q(0,0,4));
    printf("K [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           k(0,0,0), k(0,0,1), k(0,0,2), k(0,0,3), k(0,0,4));
    printf("V [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           v(0,0,0), v(0,0,1), v(0,0,2), v(0,0,3), v(0,0,4));
    
    printf("\nNeMo reference:\n");
    printf("Q [0,0,:5]: 0.581807, -0.631431, 0.337379, 0.448331, -0.399870\n");
    printf("K [0,0,:5]: -0.987436, -0.077863, 0.641622, 4.158535, 0.480071\n");
    printf("V [0,0,:5]: 0.642659, 0.265970, -1.245766, -0.900855, -0.211937\n");
    
    // Position projection
    RelPositionalEncoding pos_enc;
    pos_enc.init();
    TensorF pos_emb;
    pos_enc.get_pos_emb(time, pos_emb);
    
    TensorF pos_emb_3d({1, pos_emb.shape[0], pos_emb.shape[1]});
    for (size_t i = 0; i < pos_emb.numel(); i++) {
        pos_emb_3d.data[i] = pos_emb.data[i];
    }
    TensorF p;
    linear_no_bias(pos_emb_3d, pos_weight.data.data(), D_MODEL, D_MODEL, p);
    
    printf("\nP (pos proj) [0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           p(0,0,0), p(0,0,1), p(0,0,2), p(0,0,3), p(0,0,4));
    printf("NeMo reference: 0.127029, -0.535859, -3.569016, -5.386309, -3.918348\n");
    
    // pos_bias values
    printf("\npos_bias_u [0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           pos_bias_u.data[0], pos_bias_u.data[1], pos_bias_u.data[2],
           pos_bias_u.data[3], pos_bias_u.data[4]);
    printf("pos_bias_v [0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           pos_bias_v.data[0], pos_bias_v.data[1], pos_bias_v.data[2],
           pos_bias_v.data[3], pos_bias_v.data[4]);
    printf("NeMo reference u: -0.424462, 0.044243, 0.053922, -0.087996, 0.033922\n");
    printf("NeMo reference v: -0.421362, 0.001310, 0.053401, 0.016990, -0.064958\n");
    
    return 0;
}
