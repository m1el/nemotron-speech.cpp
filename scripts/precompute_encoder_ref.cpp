// Precompute encoder reference output and save to file
// This avoids running the slow reference implementation during tests
//
// Usage: ./precompute_encoder_ref
// Output: weights/encoder_ref.bin

#include "../include/ggml_weights.h"
#include "../include/conformer_encoder.h"

#include <cstdio>
#include <vector>

int main() {
    printf("Precomputing encoder reference output...\n");

    // Load original weights
    nemo::ModelWeights ref_weights;
    if (!ref_weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load reference weights\n");
        return 1;
    }

    // Load mel input from file - raw float32 data, shape [time, 128]
    FILE * f = fopen("test.mel.bin", "rb");
    if (!f) {
        fprintf(stderr, "Failed to open test.mel.bin\n");
        return 1;
    }

    // Get file size to determine time dimension
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t batch = 1;
    size_t features = 128;
    size_t time_in = file_size / (sizeof(float) * features);

    printf("Mel input shape: [%zu, %zu, %zu]\n", batch, time_in, features);

    // Read raw float data - file is [time, features]
    std::vector<float> raw_mel(time_in * features);
    size_t read = fread(raw_mel.data(), sizeof(float), raw_mel.size(), f);
    fclose(f);

    if (read != raw_mel.size()) {
        fprintf(stderr, "Failed to read mel data\n");
        return 1;
    }

    // Reshape to [batch, time, features]
    nemo::TensorF mel_input({batch, time_in, features});
    for (size_t t = 0; t < time_in; t++) {
        for (size_t ff = 0; ff < features; ff++) {
            mel_input(0, t, ff) = raw_mel[t * features + ff];
        }
    }

    // Run original encoder
    printf("Running encoder (this may take a while)...\n");
    nemo::ConformerEncoder encoder_ref;
    encoder_ref.load_weights(ref_weights);

    nemo::TensorF ref_output;
    encoder_ref.forward(mel_input, ref_output);

    printf("Encoder output shape: [%zu, %zu, %zu]\n",
           ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]);
    printf("Output[0,0,:5]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           ref_output(0,0,0), ref_output(0,0,1), ref_output(0,0,2),
           ref_output(0,0,3), ref_output(0,0,4));

    // Save to file
    // Format: [batch, time_out, d_model] as 3 uint64_t, then float data
    FILE * out = fopen("weights/encoder_ref.bin", "wb");
    if (!out) {
        fprintf(stderr, "Failed to open output file\n");
        return 1;
    }

    uint64_t shape[3] = {ref_output.shape[0], ref_output.shape[1], ref_output.shape[2]};
    fwrite(shape, sizeof(uint64_t), 3, out);
    fwrite(ref_output.data.data(), sizeof(float), ref_output.numel(), out);
    fclose(out);

    printf("Saved encoder reference output to weights/encoder_ref.bin\n");
    printf("File size: %zu bytes\n", 3 * sizeof(uint64_t) + ref_output.numel() * sizeof(float));

    return 0;
}
