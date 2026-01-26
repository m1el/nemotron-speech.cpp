#include "conv_subsampling.h"

namespace nemo {

void ConvSubsampling::load_weights(const ModelWeights& weights) {
    // Load conv weights
    conv0_weight = weights.require("encoder.pre_encode.conv.0.weight").data.data();
    conv0_bias = weights.require("encoder.pre_encode.conv.0.bias").data.data();

    conv2_weight = weights.require("encoder.pre_encode.conv.2.weight").data.data();
    conv2_bias = weights.require("encoder.pre_encode.conv.2.bias").data.data();

    conv3_weight = weights.require("encoder.pre_encode.conv.3.weight").data.data();
    conv3_bias = weights.require("encoder.pre_encode.conv.3.bias").data.data();

    conv5_weight = weights.require("encoder.pre_encode.conv.5.weight").data.data();
    conv5_bias = weights.require("encoder.pre_encode.conv.5.bias").data.data();

    conv6_weight = weights.require("encoder.pre_encode.conv.6.weight").data.data();
    conv6_bias = weights.require("encoder.pre_encode.conv.6.bias").data.data();

    // Load output linear layer
    out_weight = weights.require("encoder.pre_encode.out.weight").data.data();
    out_bias = weights.require("encoder.pre_encode.out.bias").data.data();
}

void ConvSubsampling::forward(const TensorF& input, TensorF& output) {
    // Input shape: [batch, time, 128]
    size_t batch = input.shape[0];
    size_t time = input.shape[1];
    size_t features = input.shape[2];  // Should be 128

    // Reshape to [batch, 1, time, 128] for Conv2D
    TensorF x({batch, 1, time, features});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t f = 0; f < features; f++) {
                x(b, 0, t, f) = input(b, t, f);
            }
        }
    }

    // Conv0: CausalConv2D(1, 256, k=3, s=2) + ReLU
    causal_conv2d(x, conv0_weight, CONV_CHANNELS, 1, 3, 3, 2, 2, 1, conv0_bias, buf1_);
    relu_inplace(buf1_);

    // Conv2: Depthwise CausalConv2D(256, 256, k=3, s=2, groups=256)
    causal_conv2d(buf1_, conv2_weight, CONV_CHANNELS, CONV_CHANNELS, 3, 3, 2, 2, CONV_CHANNELS, conv2_bias, buf2_);

    // Conv3: Pointwise Conv2d(256, 256, k=1, s=1)
    conv2d(buf2_, conv3_weight, CONV_CHANNELS, CONV_CHANNELS, 1, 1, 1, 1, 0, 0, 1, conv3_bias, buf1_);
    relu_inplace(buf1_);

    // Conv5: Depthwise CausalConv2D(256, 256, k=3, s=2, groups=256)
    causal_conv2d(buf1_, conv5_weight, CONV_CHANNELS, CONV_CHANNELS, 3, 3, 2, 2, CONV_CHANNELS, conv5_bias, buf2_);

    // Conv6: Pointwise Conv2d(256, 256, k=1, s=1)
    conv2d(buf2_, conv6_weight, CONV_CHANNELS, CONV_CHANNELS, 1, 1, 1, 1, 0, 0, 1, conv6_bias, buf1_);
    relu_inplace(buf1_);

    // buf1_ shape: [batch, 256, time_out, width_out]
    size_t time_out = buf1_.shape[2];
    size_t width_out = buf1_.shape[3];
    size_t flat_dim = CONV_CHANNELS * width_out;

    // Reshape to [batch, time_out, 256*width]
    // Layout: for each (b, t), flatten channels and width
    TensorF flat({batch, time_out, flat_dim});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time_out; t++) {
            for (size_t c = 0; c < CONV_CHANNELS; c++) {
                for (size_t w = 0; w < width_out; w++) {
                    flat(b, t, c * width_out + w) = buf1_(b, c, t, w);
                }
            }
        }
    }

    // Linear projection: [batch, time_out, 4352] -> [batch, time_out, 1024]
    linear(flat, out_weight, OUT_FEATURES, flat_dim, out_bias, output);
}

}  // namespace nemo
