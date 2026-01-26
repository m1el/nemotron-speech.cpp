// Debug C++ conformer encoder layer by layer
#include "conformer_encoder.h"
#include "ggml_weights.h"

#include <cstdio>
#include <fstream>
#include <algorithm>

using namespace nemo;

int main() {
    printf("Starting...\n");
    
    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Load C++ subsample output
    std::ifstream sub_file("cpp_subsampling_debug.bin", std::ios::binary);
    if (!sub_file.is_open()) {
        fprintf(stderr, "Failed to open cpp_subsampling_debug.bin\n");
        return 1;
    }

    sub_file.seekg(0, std::ios::end);
    size_t sub_size = sub_file.tellg() / sizeof(float);
    sub_file.seekg(0, std::ios::beg);

    size_t time = sub_size / 1024;
    TensorF subsample({1, time, 1024});
    sub_file.read(reinterpret_cast<char*>(subsample.data.data()), sub_size * sizeof(float));
    sub_file.close();

    printf("Loaded subsample: [1, %zu, 1024]\n", time);
    printf("Subsample first 5 of frame 0: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           subsample(0, 0, 0), subsample(0, 0, 1), subsample(0, 0, 2), 
           subsample(0, 0, 3), subsample(0, 0, 4));

    // Create positional encoding
    RelPositionalEncoding pos_enc;
    pos_enc.init();
    
    TensorF pos_emb;
    pos_enc.get_pos_emb(time, pos_emb);
    printf("\nPos emb first 5: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           pos_emb(0, 0, 0), pos_emb(0, 0, 1), pos_emb(0, 0, 2),
           pos_emb(0, 0, 3), pos_emb(0, 0, 4));

    // Create and run through first conformer layer
    ConformerLayer layer0;
    layer0.load_weights(weights, "encoder.layers.0");

    TensorF layer0_out;
    layer0.forward(subsample, pos_emb, layer0_out);

    printf("\nLayer 0 output first 5 of frame 0: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           layer0_out(0, 0, 0), layer0_out(0, 0, 1), layer0_out(0, 0, 2),
           layer0_out(0, 0, 3), layer0_out(0, 0, 4));
    printf("Layer 0 output first 5 of frame 1: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           layer0_out(0, 1, 0), layer0_out(0, 1, 1), layer0_out(0, 1, 2),
           layer0_out(0, 1, 3), layer0_out(0, 1, 4));

    // Save layer 0 output
    std::ofstream out("cpp_layer0_out.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(layer0_out.data.data()), 
              layer0_out.numel() * sizeof(float));
    out.close();
    printf("\nSaved cpp_layer0_out.bin\n");

    return 0;
}
