// Debug C++ conformer layer - focus on self-attention
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

    // Get positional encoding
    RelPositionalEncoding pos_enc;
    pos_enc.init();
    TensorF pos_emb;
    pos_enc.get_pos_emb(time, pos_emb);

    // Create layer 0 using the class
    ConformerLayer layer0;
    layer0.load_weights(weights, "encoder.layers.0");

    // Run full forward
    TensorF output;
    layer0.forward(input, pos_emb, output);

    printf("=== Full Layer 0 Output ===\n");
    print_first5(output, "Layer 0 output");
    printf("Layer 0 output [0,1,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           output(0, 1, 0), output(0, 1, 1), output(0, 1, 2), 
           output(0, 1, 3), output(0, 1, 4));

    printf("\n=== NeMo Reference Values ===\n");
    printf("Layer 0 output [0,0,:5]: -0.043618, -0.430568, -1.367670, -1.879998, 1.547850\n");
    printf("Layer 0 output [0,1,:5]: 0.086229, 2.121558, -5.951703, 0.945651, 4.832818\n");

    return 0;
}
