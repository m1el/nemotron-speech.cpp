#include "include/ops.h"

#include <algorithm>
#include <cstring>

namespace nemo {

// ============================================================================
// Activation functions
// ============================================================================

void swish_inplace(TensorF& x) {
    for (size_t i = 0; i < x.data.size(); i++) {
        x.data[i] = swish(x.data[i]);
    }
}

void relu_inplace(TensorF& x) {
    for (size_t i = 0; i < x.data.size(); i++) {
        x.data[i] = std::max(0.0f, x.data[i]);
    }
}

// ============================================================================
// Linear layers
// ============================================================================

void linear(
    const TensorF& x,
    const float* weight, size_t out_features, size_t in_features,
    const float* bias,
    TensorF& out
) {
    // x shape: [batch..., in_features]
    // Compute batch size (all dims except last)
    size_t batch_size = x.numel() / in_features;

    // Output shape
    std::vector<size_t> out_shape = x.shape;
    out_shape.back() = out_features;
    out.resize(out_shape);

    // Matrix multiply: out[b, o] = sum_i(x[b, i] * weight[o, i]) + bias[o]
    for (size_t b = 0; b < batch_size; b++) {
        const float* x_row = x.data.data() + b * in_features;
        float* out_row = out.data.data() + b * out_features;

        for (size_t o = 0; o < out_features; o++) {
            float sum = bias ? bias[o] : 0.0f;
            const float* w_row = weight + o * in_features;
            for (size_t i = 0; i < in_features; i++) {
                sum += x_row[i] * w_row[i];
            }
            out_row[o] = sum;
        }
    }
}

void linear_no_bias(
    const TensorF& x,
    const float* weight, size_t out_features, size_t in_features,
    TensorF& out
) {
    linear(x, weight, out_features, in_features, nullptr, out);
}

// ============================================================================
// Normalization
// ============================================================================

void layer_norm(
    const TensorF& x,
    const float* weight, const float* bias,
    size_t normalized_shape,
    float eps,
    TensorF& out
) {
    out.resize(x.shape);

    size_t batch_size = x.numel() / normalized_shape;

    for (size_t b = 0; b < batch_size; b++) {
        const float* x_ptr = x.data.data() + b * normalized_shape;
        float* out_ptr = out.data.data() + b * normalized_shape;

        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < normalized_shape; i++) {
            mean += x_ptr[i];
        }
        mean /= normalized_shape;

        // Compute variance
        float var = 0.0f;
        for (size_t i = 0; i < normalized_shape; i++) {
            float diff = x_ptr[i] - mean;
            var += diff * diff;
        }
        var /= normalized_shape;

        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < normalized_shape; i++) {
            out_ptr[i] = (x_ptr[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

void layer_norm_inplace(
    TensorF& x,
    const float* weight, const float* bias,
    size_t normalized_shape,
    float eps
) {
    size_t batch_size = x.numel() / normalized_shape;

    for (size_t b = 0; b < batch_size; b++) {
        float* x_ptr = x.data.data() + b * normalized_shape;

        // Compute mean
        float mean = 0.0f;
        for (size_t i = 0; i < normalized_shape; i++) {
            mean += x_ptr[i];
        }
        mean /= normalized_shape;

        // Compute variance
        float var = 0.0f;
        for (size_t i = 0; i < normalized_shape; i++) {
            float diff = x_ptr[i] - mean;
            var += diff * diff;
        }
        var /= normalized_shape;

        // Normalize in-place
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < normalized_shape; i++) {
            x_ptr[i] = (x_ptr[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

// ============================================================================
// Convolution operations
// ============================================================================

void conv1d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kernel_size, size_t stride, size_t padding, size_t groups,
    const float* bias,
    TensorF& output
) {
    size_t batch = input.shape[0];
    size_t in_len = input.shape[2];
    size_t out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    output.resize({batch, out_channels, out_len});
    output.zero();

    size_t in_channels_per_group = in_channels / groups;
    size_t out_channels_per_group = out_channels / groups;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            for (size_t oc = 0; oc < out_channels_per_group; oc++) {
                size_t oc_abs = g * out_channels_per_group + oc;

                for (size_t ot = 0; ot < out_len; ot++) {
                    float sum = bias ? bias[oc_abs] : 0.0f;

                    for (size_t ic = 0; ic < in_channels_per_group; ic++) {
                        size_t ic_abs = g * in_channels_per_group + ic;

                        for (size_t k = 0; k < kernel_size; k++) {
                            int it = (int)(ot * stride + k) - (int)padding;
                            if (it >= 0 && it < (int)in_len) {
                                float w = weight[(oc_abs * in_channels_per_group + ic) * kernel_size + k];
                                float x = input(b, ic_abs, it);
                                sum += w * x;
                            }
                        }
                    }
                    output(b, oc_abs, ot) = sum;
                }
            }
        }
    }
}

void causal_conv1d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kernel_size, size_t stride, size_t groups,
    const float* bias,
    TensorF& output
) {
    // Causal padding: (kernel_size - 1) on left, 0 on right
    size_t padding = kernel_size - 1;
    conv1d(input, weight, out_channels, in_channels, kernel_size, stride, padding, groups, bias, output);

    // Trim to match input length (remove right padding effect)
    // For stride=1, output length should equal input length
    if (stride == 1 && output.shape[2] > input.shape[2]) {
        size_t target_len = input.shape[2];
        TensorF trimmed({output.shape[0], output.shape[1], target_len});
        for (size_t b = 0; b < output.shape[0]; b++) {
            for (size_t c = 0; c < output.shape[1]; c++) {
                for (size_t t = 0; t < target_len; t++) {
                    trimmed(b, c, t) = output(b, c, t);
                }
            }
        }
        output = std::move(trimmed);
    }
}

void conv2d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kH, size_t kW,
    size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w,
    size_t groups,
    const float* bias,
    TensorF& output
) {
    size_t batch = input.shape[0];
    size_t in_h = input.shape[2];
    size_t in_w = input.shape[3];
    size_t out_h = (in_h + 2 * pad_h - kH) / stride_h + 1;
    size_t out_w = (in_w + 2 * pad_w - kW) / stride_w + 1;

    output.resize({batch, out_channels, out_h, out_w});
    output.zero();

    size_t in_channels_per_group = in_channels / groups;
    size_t out_channels_per_group = out_channels / groups;

    for (size_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            for (size_t oc = 0; oc < out_channels_per_group; oc++) {
                size_t oc_abs = g * out_channels_per_group + oc;

                for (size_t oh = 0; oh < out_h; oh++) {
                    for (size_t ow = 0; ow < out_w; ow++) {
                        float sum = bias ? bias[oc_abs] : 0.0f;

                        for (size_t ic = 0; ic < in_channels_per_group; ic++) {
                            size_t ic_abs = g * in_channels_per_group + ic;

                            for (size_t kh = 0; kh < kH; kh++) {
                                for (size_t kw = 0; kw < kW; kw++) {
                                    int ih = (int)(oh * stride_h + kh) - (int)pad_h;
                                    int iw = (int)(ow * stride_w + kw) - (int)pad_w;

                                    if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                        size_t w_idx = ((oc_abs * in_channels_per_group + ic) * kH + kh) * kW + kw;
                                        float w = weight[w_idx];
                                        float x = input(b, ic_abs, ih, iw);
                                        sum += w * x;
                                    }
                                }
                            }
                        }
                        output(b, oc_abs, oh, ow) = sum;
                    }
                }
            }
        }
    }
}

void causal_conv2d(
    const TensorF& input,
    const float* weight, size_t out_channels, size_t in_channels,
    size_t kH, size_t kW,
    size_t stride_h, size_t stride_w,
    size_t groups,
    const float* bias,
    TensorF& output
) {
    // NeMo CausalConv2D padding: left/top = k-1, right/bottom = stride-1
    // F.pad(x, pad=(left, right, top, bottom))
    size_t pad_left = kW - 1;
    size_t pad_right = stride_w - 1;
    size_t pad_top = kH - 1;
    size_t pad_bottom = stride_h - 1;

    size_t batch = input.shape[0];
    size_t in_ch = input.shape[1];
    size_t in_h = input.shape[2];
    size_t in_w = input.shape[3];

    // Create padded input
    size_t padded_h = in_h + pad_top + pad_bottom;
    size_t padded_w = in_w + pad_left + pad_right;
    TensorF padded({batch, in_ch, padded_h, padded_w}, 0.0f);

    // Copy input to padded tensor (offset by pad_top, pad_left)
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < in_ch; c++) {
            for (size_t h = 0; h < in_h; h++) {
                for (size_t w = 0; w < in_w; w++) {
                    padded(b, c, h + pad_top, w + pad_left) = input(b, c, h, w);
                }
            }
        }
    }

    // Now do standard conv2d with no padding
    conv2d(padded, weight, out_channels, in_channels, kH, kW,
           stride_h, stride_w, 0, 0, groups, bias, output);
}

// ============================================================================
// Gated Linear Unit
// ============================================================================

void glu(const TensorF& input, TensorF& output) {
    // Input shape: [*, 2*features]
    // Split along last dimension
    size_t features = input.shape.back() / 2;
    size_t batch_size = input.numel() / (2 * features);

    std::vector<size_t> out_shape = input.shape;
    out_shape.back() = features;
    output.resize(out_shape);

    for (size_t b = 0; b < batch_size; b++) {
        const float* in_ptr = input.data.data() + b * 2 * features;
        float* out_ptr = output.data.data() + b * features;

        for (size_t i = 0; i < features; i++) {
            float a = in_ptr[i];
            float gate = sigmoid(in_ptr[features + i]);
            out_ptr[i] = a * gate;
        }
    }
}

// ============================================================================
// LSTM operations
// ============================================================================

void lstm_cell(
    const TensorF& input,
    const TensorF& h_prev, const TensorF& c_prev,
    const float* weight_ih, const float* weight_hh,
    const float* bias_ih, const float* bias_hh,
    size_t input_size, size_t hidden_size,
    TensorF& h_out, TensorF& c_out
) {
    size_t batch = input.shape[0];
    h_out.resize({batch, hidden_size});
    c_out.resize({batch, hidden_size});

    // Gates: [i, f, g, o] each of size hidden_size
    // Total: 4 * hidden_size

    for (size_t b = 0; b < batch; b++) {
        const float* x = input.data.data() + b * input_size;
        const float* h = h_prev.data.data() + b * hidden_size;
        const float* c = c_prev.data.data() + b * hidden_size;
        float* h_new = h_out.data.data() + b * hidden_size;
        float* c_new = c_out.data.data() + b * hidden_size;

        // Compute gates: gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
        std::vector<float> gates(4 * hidden_size);

        for (size_t g = 0; g < 4 * hidden_size; g++) {
            float sum = bias_ih[g] + bias_hh[g];

            // Input contribution: W_ih[g, :] @ x
            const float* wih_row = weight_ih + g * input_size;
            for (size_t i = 0; i < input_size; i++) {
                sum += wih_row[i] * x[i];
            }

            // Hidden contribution: W_hh[g, :] @ h
            const float* whh_row = weight_hh + g * hidden_size;
            for (size_t i = 0; i < hidden_size; i++) {
                sum += whh_row[i] * h[i];
            }

            gates[g] = sum;
        }

        // Apply activations and compute new states
        for (size_t i = 0; i < hidden_size; i++) {
            float i_gate = sigmoid(gates[i]);                          // input gate
            float f_gate = sigmoid(gates[hidden_size + i]);            // forget gate
            float g_gate = std::tanh(gates[2 * hidden_size + i]);      // cell gate
            float o_gate = sigmoid(gates[3 * hidden_size + i]);        // output gate

            c_new[i] = f_gate * c[i] + i_gate * g_gate;
            h_new[i] = o_gate * std::tanh(c_new[i]);
        }
    }
}

void lstm_forward(
    const TensorF& input,
    LSTMState& state,
    const float* const* weight_ih,
    const float* const* weight_hh,
    const float* const* bias_ih,
    const float* const* bias_hh,
    size_t input_size, size_t hidden_size,
    size_t num_layers,
    TensorF& output
) {
    size_t batch = input.shape[0];

    TensorF layer_input = input;
    TensorF h_out, c_out;

    for (size_t layer = 0; layer < num_layers; layer++) {
        // Extract h and c for this layer
        TensorF h_prev({batch, hidden_size});
        TensorF c_prev({batch, hidden_size});

        for (size_t b = 0; b < batch; b++) {
            for (size_t i = 0; i < hidden_size; i++) {
                h_prev(b, i) = state.h(layer, b, i);
                c_prev(b, i) = state.c(layer, b, i);
            }
        }

        // Determine input size for this layer
        size_t layer_input_size = (layer == 0) ? input_size : hidden_size;

        // Run LSTM cell
        lstm_cell(layer_input, h_prev, c_prev,
                  weight_ih[layer], weight_hh[layer],
                  bias_ih[layer], bias_hh[layer],
                  layer_input_size, hidden_size,
                  h_out, c_out);

        // Store updated state
        for (size_t b = 0; b < batch; b++) {
            for (size_t i = 0; i < hidden_size; i++) {
                state.h(layer, b, i) = h_out(b, i);
                state.c(layer, b, i) = c_out(b, i);
            }
        }

        // Output of this layer is input to next layer
        layer_input = h_out;
    }

    // Output is the hidden state of the last layer
    output = std::move(h_out);
}

// ============================================================================
// Embedding
// ============================================================================

void embedding(
    const int* indices, size_t batch_size,
    const float* weight, size_t vocab_size, size_t embedding_dim,
    TensorF& output
) {
    (void)vocab_size;  // Used for bounds checking in debug
    output.resize({batch_size, embedding_dim});

    for (size_t b = 0; b < batch_size; b++) {
        int idx = indices[b];
        assert(idx >= 0 && (size_t)idx < vocab_size);

        const float* emb = weight + idx * embedding_dim;
        float* out = output.data.data() + b * embedding_dim;
        std::memcpy(out, emb, embedding_dim * sizeof(float));
    }
}

// ============================================================================
// Utility operations
// ============================================================================

void add(const TensorF& a, const TensorF& b, TensorF& out) {
    assert(a.numel() == b.numel());
    out.resize(a.shape);

    for (size_t i = 0; i < a.numel(); i++) {
        out.data[i] = a.data[i] + b.data[i];
    }
}

void add_inplace(TensorF& a, const TensorF& b) {
    assert(a.numel() == b.numel());
    for (size_t i = 0; i < a.numel(); i++) {
        a.data[i] += b.data[i];
    }
}

void scale_inplace(TensorF& x, float scale) {
    for (size_t i = 0; i < x.numel(); i++) {
        x.data[i] *= scale;
    }
}

void softmax(const TensorF& input, TensorF& output) {
    output.resize(input.shape);

    size_t last_dim = input.shape.back();
    size_t batch_size = input.numel() / last_dim;

    for (size_t b = 0; b < batch_size; b++) {
        const float* in_ptr = input.data.data() + b * last_dim;
        float* out_ptr = output.data.data() + b * last_dim;

        // Find max for numerical stability
        float max_val = in_ptr[0];
        for (size_t i = 1; i < last_dim; i++) {
            max_val = std::max(max_val, in_ptr[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; i++) {
            out_ptr[i] = std::exp(in_ptr[i] - max_val);
            sum += out_ptr[i];
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < last_dim; i++) {
            out_ptr[i] *= inv_sum;
        }
    }
}

void argmax(const TensorF& input, std::vector<int>& indices) {
    size_t last_dim = input.shape.back();
    size_t batch_size = input.numel() / last_dim;

    indices.resize(batch_size);

    for (size_t b = 0; b < batch_size; b++) {
        const float* ptr = input.data.data() + b * last_dim;
        int max_idx = 0;
        float max_val = ptr[0];

        for (size_t i = 1; i < last_dim; i++) {
            if (ptr[i] > max_val) {
                max_val = ptr[i];
                max_idx = (int)i;
            }
        }

        indices[b] = max_idx;
    }
}

}  // namespace nemo
