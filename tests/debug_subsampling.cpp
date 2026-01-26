// Debug ConvSubsampling layer by layer
#include "conv_subsampling.h"
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

void save_tensor(const TensorF& t, const char* path) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(t.data.data()), t.numel() * sizeof(float));
    printf("Saved %s: [%zu", path, t.shape[0]);
    for (size_t i = 1; i < t.shape.size(); i++) printf(", %zu", t.shape[i]);
    printf("]\n");
}

int main(int argc, char** argv) {
    const char* mel_path = argc > 1 ? argv[1] : "test.mel.bin";

    // Load weights
    ModelWeights weights;
    if (!weights.load("weights/model.bin")) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Get conv weights
    auto& conv0_weight = weights.require("encoder.pre_encode.conv.0.weight");
    auto& conv0_bias = weights.require("encoder.pre_encode.conv.0.bias");
    auto& conv2_weight = weights.require("encoder.pre_encode.conv.2.weight");
    auto& conv2_bias = weights.require("encoder.pre_encode.conv.2.bias");
    auto& conv3_weight = weights.require("encoder.pre_encode.conv.3.weight");
    auto& conv3_bias = weights.require("encoder.pre_encode.conv.3.bias");
    auto& conv5_weight = weights.require("encoder.pre_encode.conv.5.weight");
    auto& conv5_bias = weights.require("encoder.pre_encode.conv.5.bias");
    auto& conv6_weight = weights.require("encoder.pre_encode.conv.6.weight");
    auto& conv6_bias = weights.require("encoder.pre_encode.conv.6.bias");
    auto& out_weight = weights.require("encoder.pre_encode.out.weight");
    auto& out_bias = weights.require("encoder.pre_encode.out.bias");

    printf("Conv0 weight shape: [%zu", conv0_weight.shape[0]);
    for (size_t i = 1; i < conv0_weight.shape.size(); i++) printf(", %zu", conv0_weight.shape[i]);
    printf("]\n");

    // Load mel
    TensorF mel;
    if (!load_mel_bin(mel_path, mel)) {
        fprintf(stderr, "Failed to load mel\n");
        return 1;
    }
    printf("Mel shape: [%zu, %zu, %zu]\n", mel.shape[0], mel.shape[1], mel.shape[2]);

    size_t batch = mel.shape[0];
    size_t time = mel.shape[1];
    size_t features = mel.shape[2];

    // Reshape to [batch, 1, time, 128] for Conv2D
    TensorF x({batch, 1, time, features});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t f = 0; f < features; f++) {
                x(b, 0, t, f) = mel(b, t, f);
            }
        }
    }
    printf("Input to conv0: [%zu, %zu, %zu, %zu]\n", x.shape[0], x.shape[1], x.shape[2], x.shape[3]);

    // Conv0: CausalConv2D(1, 256, k=3, s=2)
    TensorF buf1, buf2;
    causal_conv2d(x, conv0_weight.data.data(), 256, 1, 3, 3, 2, 2, 1, conv0_bias.data.data(), buf1);
    printf("\nConv0 output: [%zu, %zu, %zu, %zu]\n", buf1.shape[0], buf1.shape[1], buf1.shape[2], buf1.shape[3]);
    printf("Conv0 stats: min=%.4f, max=%.4f\n",
        *std::min_element(buf1.data.begin(), buf1.data.end()),
        *std::max_element(buf1.data.begin(), buf1.data.end()));
    printf("Conv0 [0,0,0,:5]: %.6f %.6f %.6f %.6f %.6f\n",
        buf1(0,0,0,0), buf1(0,0,0,1), buf1(0,0,0,2), buf1(0,0,0,3), buf1(0,0,0,4));
    save_tensor(buf1, "cpp_conv0_out.bin");

    // ReLU
    relu_inplace(buf1);
    printf("\nAfter ReLU1: min=%.4f, max=%.4f\n",
        *std::min_element(buf1.data.begin(), buf1.data.end()),
        *std::max_element(buf1.data.begin(), buf1.data.end()));

    // Conv2: Depthwise CausalConv2D(256, 256, k=3, s=2, groups=256)
    causal_conv2d(buf1, conv2_weight.data.data(), 256, 256, 3, 3, 2, 2, 256, conv2_bias.data.data(), buf2);
    printf("\nConv2 output: [%zu, %zu, %zu, %zu]\n", buf2.shape[0], buf2.shape[1], buf2.shape[2], buf2.shape[3]);
    printf("Conv2 stats: min=%.4f, max=%.4f\n",
        *std::min_element(buf2.data.begin(), buf2.data.end()),
        *std::max_element(buf2.data.begin(), buf2.data.end()));

    // Conv3: Pointwise Conv2d(256, 256, k=1, s=1)
    conv2d(buf2, conv3_weight.data.data(), 256, 256, 1, 1, 1, 1, 0, 0, 1, conv3_bias.data.data(), buf1);
    printf("\nConv3 output: [%zu, %zu, %zu, %zu]\n", buf1.shape[0], buf1.shape[1], buf1.shape[2], buf1.shape[3]);

    // ReLU
    relu_inplace(buf1);
    printf("After ReLU4: min=%.4f, max=%.4f\n",
        *std::min_element(buf1.data.begin(), buf1.data.end()),
        *std::max_element(buf1.data.begin(), buf1.data.end()));

    // Conv5: Depthwise CausalConv2D(256, 256, k=3, s=2, groups=256)
    causal_conv2d(buf1, conv5_weight.data.data(), 256, 256, 3, 3, 2, 2, 256, conv5_bias.data.data(), buf2);
    printf("\nConv5 output: [%zu, %zu, %zu, %zu]\n", buf2.shape[0], buf2.shape[1], buf2.shape[2], buf2.shape[3]);

    // Conv6: Pointwise Conv2d(256, 256, k=1, s=1)
    conv2d(buf2, conv6_weight.data.data(), 256, 256, 1, 1, 1, 1, 0, 0, 1, conv6_bias.data.data(), buf1);
    printf("\nConv6 output: [%zu, %zu, %zu, %zu]\n", buf1.shape[0], buf1.shape[1], buf1.shape[2], buf1.shape[3]);

    // ReLU
    relu_inplace(buf1);
    printf("After ReLU7: min=%.4f, max=%.4f\n",
        *std::min_element(buf1.data.begin(), buf1.data.end()),
        *std::max_element(buf1.data.begin(), buf1.data.end()));

    // Reshape
    size_t time_out = buf1.shape[2];
    size_t width_out = buf1.shape[3];
    size_t flat_dim = 256 * width_out;

    printf("\nBefore reshape: [%zu, 256, %zu, %zu]\n", batch, time_out, width_out);
    printf("Reshape to: [%zu, %zu, %zu]\n", batch, time_out, flat_dim);

    TensorF flat({batch, time_out, flat_dim});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time_out; t++) {
            for (size_t c = 0; c < 256; c++) {
                for (size_t w = 0; w < width_out; w++) {
                    flat(b, t, c * width_out + w) = buf1(b, c, t, w);
                }
            }
        }
    }
    printf("Before linear [0,0,:10]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
        flat(0,0,0), flat(0,0,1), flat(0,0,2), flat(0,0,3), flat(0,0,4),
        flat(0,0,5), flat(0,0,6), flat(0,0,7), flat(0,0,8), flat(0,0,9));
    printf("Before linear [0,-1,:10]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
        flat(0,time_out-1,0), flat(0,time_out-1,1), flat(0,time_out-1,2), flat(0,time_out-1,3), flat(0,time_out-1,4),
        flat(0,time_out-1,5), flat(0,time_out-1,6), flat(0,time_out-1,7), flat(0,time_out-1,8), flat(0,time_out-1,9));

    // Linear
    TensorF output;
    linear(flat, out_weight.data.data(), 1024, flat_dim, out_bias.data.data(), output);
    printf("\nFinal output: [%zu, %zu, %zu]\n", output.shape[0], output.shape[1], output.shape[2]);
    printf("Final output [0,0,:5]: %.6f %.6f %.6f %.6f %.6f\n",
        output(0,0,0), output(0,0,1), output(0,0,2), output(0,0,3), output(0,0,4));
    printf("Final output [0,-1,:5]: %.6f %.6f %.6f %.6f %.6f\n",
        output(0,time_out-1,0), output(0,time_out-1,1), output(0,time_out-1,2), output(0,time_out-1,3), output(0,time_out-1,4));

    save_tensor(output, "cpp_subsampling_debug.bin");

    return 0;
}
