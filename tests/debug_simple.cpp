#include <cstdio>
#include "conformer_encoder.h"
#include "ggml_weights.h"

using namespace nemo;

int main() {
    printf("Starting\n");
    fflush(stdout);
    
    ModelWeights weights;
    printf("Loading weights...\n");
    fflush(stdout);
    if (!weights.load("weights/model.bin")) {
        printf("Failed to load\n");
        return 1;
    }
    printf("Loaded\n");
    fflush(stdout);

    printf("Creating layer\n");
    fflush(stdout);
    ConformerLayer layer;
    printf("Loading layer weights\n");
    fflush(stdout);
    layer.load_weights(weights, "encoder.layers.0");  // Fixed: pass string prefix
    printf("Done\n");
    fflush(stdout);
    
    return 0;
}
