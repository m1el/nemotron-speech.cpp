#include "include/conformer_modules.h"

#include <cmath>

namespace nemo {

// ============================================================================
// ConformerFeedForward
// ============================================================================

void ConformerFeedForward::load_weights(const ModelWeights& weights, const std::string& prefix) {
    linear1_weight = weights.require(prefix + ".linear1.weight").data.data();
    linear2_weight = weights.require(prefix + ".linear2.weight").data.data();
}

void ConformerFeedForward::forward(const TensorF& input, TensorF& output) {
    // Linear1: [batch, time, 1024] -> [batch, time, 4096]
    linear_no_bias(input, linear1_weight, D_FF, D_MODEL, buf_);

    // Swish activation
    swish_inplace(buf_);

    // Linear2: [batch, time, 4096] -> [batch, time, 1024]
    linear_no_bias(buf_, linear2_weight, D_MODEL, D_FF, output);
}

// ============================================================================
// ConformerConvolution
// ============================================================================

void ConformerConvolution::load_weights(const ModelWeights& weights, const std::string& prefix) {
    pointwise_conv1_weight = weights.require(prefix + ".pointwise_conv1.weight").data.data();
    depthwise_conv_weight = weights.require(prefix + ".depthwise_conv.weight").data.data();
    batch_norm_weight = weights.require(prefix + ".batch_norm.weight").data.data();
    batch_norm_bias = weights.require(prefix + ".batch_norm.bias").data.data();
    pointwise_conv2_weight = weights.require(prefix + ".pointwise_conv2.weight").data.data();
}

void ConformerConvolution::forward(const TensorF& input, TensorF& output) {
    // Input: [batch, time, 1024]
    size_t batch = input.shape[0];
    size_t time = input.shape[1];

    // Transpose to [batch, 1024, time] for conv1d
    buf1_.resize({batch, D_MODEL, time});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t d = 0; d < D_MODEL; d++) {
                buf1_(b, d, t) = input(b, t, d);
            }
        }
    }

    // Pointwise Conv1: [batch, 1024, time] -> [batch, 2048, time]
    // Weight shape is [2048, 1024, 1] - treating as 1x1 conv
    conv1d(buf1_, pointwise_conv1_weight, 2048, D_MODEL, 1, 1, 0, 1, nullptr, buf2_);

    // Transpose back to [batch, time, 2048] for GLU
    buf3_.resize({batch, time, 2048});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t d = 0; d < 2048; d++) {
                buf3_(b, t, d) = buf2_(b, d, t);
            }
        }
    }

    // GLU: [batch, time, 2048] -> [batch, time, 1024]
    glu(buf3_, buf1_);

    // Transpose to [batch, 1024, time] for depthwise conv
    buf2_.resize({batch, D_MODEL, time});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t d = 0; d < D_MODEL; d++) {
                buf2_(b, d, t) = buf1_(b, t, d);
            }
        }
    }

    // Depthwise Causal Conv1d: [batch, 1024, time] -> [batch, 1024, time]
    causal_conv1d(buf2_, depthwise_conv_weight, D_MODEL, D_MODEL, KERNEL_SIZE, 1, D_MODEL, nullptr, buf1_);

    // Transpose to [batch, time, 1024] for layer norm
    buf2_.resize({batch, time, D_MODEL});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t d = 0; d < D_MODEL; d++) {
                buf2_(b, t, d) = buf1_(b, d, t);
            }
        }
    }

    // LayerNorm
    layer_norm_inplace(buf2_, batch_norm_weight, batch_norm_bias, D_MODEL, 1e-5f);

    // Swish
    swish_inplace(buf2_);

    // Transpose to [batch, 1024, time] for pointwise conv
    buf1_.resize({batch, D_MODEL, time});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t d = 0; d < D_MODEL; d++) {
                buf1_(b, d, t) = buf2_(b, t, d);
            }
        }
    }

    // Pointwise Conv2: [batch, 1024, time] -> [batch, 1024, time]
    conv1d(buf1_, pointwise_conv2_weight, D_MODEL, D_MODEL, 1, 1, 0, 1, nullptr, buf2_);

    // Transpose to [batch, time, 1024]
    output.resize({batch, time, D_MODEL});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t d = 0; d < D_MODEL; d++) {
                output(b, t, d) = buf2_(b, d, t);
            }
        }
    }
}

// ============================================================================
// RelPositionalEncoding
// ============================================================================

void RelPositionalEncoding::init() {
    if (initialized_) return;

    // Precompute sinusoidal position embeddings
    // Shape: [2*MAX_LEN - 1, D_MODEL]
    size_t len = 2 * MAX_LEN - 1;
    pos_emb_cache_.resize({len, D_MODEL});

    for (size_t pos = 0; pos < len; pos++) {
        // Position relative to center (MAX_LEN - 1)
        float p = (float)pos - (float)(MAX_LEN - 1);

        for (size_t i = 0; i < D_MODEL; i += 2) {
            float div_term = std::exp(-(float)i * std::log(10000.0f) / (float)D_MODEL);
            pos_emb_cache_(pos, i) = std::sin(p * div_term);
            if (i + 1 < D_MODEL) {
                pos_emb_cache_(pos, i + 1) = std::cos(p * div_term);
            }
        }
    }

    initialized_ = true;
}

void RelPositionalEncoding::get_pos_emb(size_t seq_len, TensorF& pos_emb) {
    init();

    // For sequence length T, we need positions from (T-1) down to -(T-1)
    // That's 2*T - 1 positions
    // NeMo order: positions go from (seq_len-1) to -(seq_len-1)
    // This means output[0] = pos (seq_len-1), output[seq_len-1] = pos 0, output[2*seq_len-2] = pos -(seq_len-1)
    size_t out_len = 2 * seq_len - 1;
    pos_emb.resize({out_len, D_MODEL});

    // Extract from cache in reverse order
    // Cache is indexed as: cache[MAX_LEN-1 + p] = embedding for position p
    // We want output[i] = embedding for position (seq_len - 1 - i)
    for (size_t i = 0; i < out_len; i++) {
        int p = (int)seq_len - 1 - (int)i;  // position value
        size_t cache_idx = MAX_LEN - 1 + p;
        for (size_t d = 0; d < D_MODEL; d++) {
            pos_emb(i, d) = pos_emb_cache_(cache_idx, d);
        }
    }
}

// ============================================================================
// RelPositionMultiHeadAttention
// ============================================================================

void RelPositionMultiHeadAttention::load_weights(const ModelWeights& weights, const std::string& prefix) {
    pos_bias_u = weights.require(prefix + ".pos_bias_u").data.data();
    pos_bias_v = weights.require(prefix + ".pos_bias_v").data.data();
    linear_q_weight = weights.require(prefix + ".linear_q.weight").data.data();
    linear_k_weight = weights.require(prefix + ".linear_k.weight").data.data();
    linear_v_weight = weights.require(prefix + ".linear_v.weight").data.data();
    linear_out_weight = weights.require(prefix + ".linear_out.weight").data.data();
    linear_pos_weight = weights.require(prefix + ".linear_pos.weight").data.data();
}

void RelPositionMultiHeadAttention::rel_shift(const TensorF& x, TensorF& out) {
    // x: [batch, heads, qlen, pos_len] where pos_len = 2*qlen - 1
    // Implements NeMo's rel_shift via pad-reshape trick:
    // 1. Pad left with zeros: [b, h, qlen, pos_len+1]
    // 2. Reshape to [b, h, pos_len+1, qlen]
    // 3. Drop first row: [b, h, pos_len, qlen]
    // 4. Reshape to [b, h, qlen, pos_len]
    // Result: position encoding is shifted so that for query i, key j,
    // the output[i,j] contains the relative position embedding for (j-i)

    size_t batch = x.shape[0];
    size_t heads = x.shape[1];
    size_t qlen = x.shape[2];
    size_t pos_len = x.shape[3];  // 2*qlen - 1

    // Step 1: Pad with zero on the left
    // padded: [batch, heads, qlen, pos_len + 1]
    TensorF padded({batch, heads, qlen, pos_len + 1});
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t i = 0; i < qlen; i++) {
                padded(b, h, i, 0) = 0.0f;  // left padding
                for (size_t j = 0; j < pos_len; j++) {
                    padded(b, h, i, j + 1) = x(b, h, i, j);
                }
            }
        }
    }

    // Steps 2-4: Reshape and drop first row
    // NeMo's pad-reshape-drop algorithm:
    // 1. Pad left: [b, h, qlen, pos_len] -> [b, h, qlen, pos_len+1]
    // 2. View as [b, h, pos_len+1, qlen]
    // 3. Drop first row: [b, h, pos_len, qlen]
    // 4. View as [b, h, qlen, pos_len]
    // 5. Slice [:,:,:,:qlen] to get final [b, h, qlen, qlen]
    //
    // Tracing through the index math: out[i, j] = x[i, j + qlen - 1 - i]
    // This aligns position embeddings so that for query i and key j,
    // we get the embedding for relative position (i - j).
    out.resize({batch, heads, qlen, qlen});
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t i = 0; i < qlen; i++) {
                for (size_t j = 0; j < qlen; j++) {
                    // k = j + qlen - 1 - i = (qlen - 1) + (j - i)
                    size_t k = j + qlen - 1 - i;
                    out(b, h, i, j) = x(b, h, i, k);
                }
            }
        }
    }
}

void RelPositionMultiHeadAttention::forward(const TensorF& input, const TensorF& pos_emb, TensorF& output) {
    // input: [batch, time, d_model]
    // pos_emb: [2*time-1, d_model]

    size_t batch = input.shape[0];
    size_t time = input.shape[1];

    // Q, K, V projections
    linear_no_bias(input, linear_q_weight, D_MODEL, D_MODEL, q_);
    linear_no_bias(input, linear_k_weight, D_MODEL, D_MODEL, k_);
    linear_no_bias(input, linear_v_weight, D_MODEL, D_MODEL, v_);

    // Position projection: [2*time-1, d_model] -> [2*time-1, d_model]
    TensorF pos_emb_3d({1, pos_emb.shape[0], pos_emb.shape[1]});
    for (size_t i = 0; i < pos_emb.numel(); i++) {
        pos_emb_3d.data[i] = pos_emb.data[i];
    }
    linear_no_bias(pos_emb_3d, linear_pos_weight, D_MODEL, D_MODEL, pos_);

    // Reshape Q, K, V to [batch, heads, time, d_head]
    TensorF q_heads({batch, N_HEADS, time, D_HEAD});
    TensorF k_heads({batch, N_HEADS, time, D_HEAD});
    TensorF v_heads({batch, N_HEADS, time, D_HEAD});

    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t h = 0; h < N_HEADS; h++) {
                for (size_t d = 0; d < D_HEAD; d++) {
                    size_t idx = h * D_HEAD + d;
                    q_heads(b, h, t, d) = q_(b, t, idx);
                    k_heads(b, h, t, d) = k_(b, t, idx);
                    v_heads(b, h, t, d) = v_(b, t, idx);
                }
            }
        }
    }

    // Add positional bias to queries
    // q_with_bias_u = q + pos_bias_u: for content attention
    // q_with_bias_v = q + pos_bias_v: for position attention
    TensorF q_u({batch, N_HEADS, time, D_HEAD});
    TensorF q_v({batch, N_HEADS, time, D_HEAD});

    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < N_HEADS; h++) {
            for (size_t t = 0; t < time; t++) {
                for (size_t d = 0; d < D_HEAD; d++) {
                    float q_val = q_heads(b, h, t, d);
                    q_u(b, h, t, d) = q_val + pos_bias_u[h * D_HEAD + d];
                    q_v(b, h, t, d) = q_val + pos_bias_v[h * D_HEAD + d];
                }
            }
        }
    }

    // Reshape pos to [1, heads, 2*time-1, d_head]
    size_t pos_len = 2 * time - 1;
    TensorF pos_heads({1, N_HEADS, pos_len, D_HEAD});
    for (size_t p = 0; p < pos_len; p++) {
        for (size_t h = 0; h < N_HEADS; h++) {
            for (size_t d = 0; d < D_HEAD; d++) {
                pos_heads(0, h, p, d) = pos_(0, p, h * D_HEAD + d);
            }
        }
    }

    // Content attention: q_u @ k^T
    // [batch, heads, time, d_head] @ [batch, heads, d_head, time] -> [batch, heads, time, time]
    TensorF content_attn({batch, N_HEADS, time, time});
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < N_HEADS; h++) {
            for (size_t i = 0; i < time; i++) {
                for (size_t j = 0; j < time; j++) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < D_HEAD; d++) {
                        sum += q_u(b, h, i, d) * k_heads(b, h, j, d);
                    }
                    content_attn(b, h, i, j) = sum;
                }
            }
        }
    }

    // Position attention: q_v @ pos^T
    // [batch, heads, time, d_head] @ [1, heads, d_head, 2*time-1] -> [batch, heads, time, 2*time-1]
    TensorF pos_attn_raw({batch, N_HEADS, time, pos_len});
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < N_HEADS; h++) {
            for (size_t i = 0; i < time; i++) {
                for (size_t p = 0; p < pos_len; p++) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < D_HEAD; d++) {
                        sum += q_v(b, h, i, d) * pos_heads(0, h, p, d);
                    }
                    pos_attn_raw(b, h, i, p) = sum;
                }
            }
        }
    }

    // Rel shift to align position attention
    TensorF pos_attn;
    rel_shift(pos_attn_raw, pos_attn);

    // Combine attention scores
    float scale = 1.0f / std::sqrt((float)D_HEAD);
    attn_scores_.resize({batch, N_HEADS, time, time});
    for (size_t i = 0; i < attn_scores_.numel(); i++) {
        attn_scores_.data[i] = (content_attn.data[i] + pos_attn.data[i]) * scale;
    }

    // Softmax over last dimension
    attn_weights_.resize({batch, N_HEADS, time, time});
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < N_HEADS; h++) {
            for (size_t i = 0; i < time; i++) {
                // Find max for stability
                float max_val = attn_scores_(b, h, i, 0);
                for (size_t j = 1; j < time; j++) {
                    max_val = std::max(max_val, attn_scores_(b, h, i, j));
                }
                // Exp and sum
                float sum = 0.0f;
                for (size_t j = 0; j < time; j++) {
                    float exp_val = std::exp(attn_scores_(b, h, i, j) - max_val);
                    attn_weights_(b, h, i, j) = exp_val;
                    sum += exp_val;
                }
                // Normalize
                for (size_t j = 0; j < time; j++) {
                    attn_weights_(b, h, i, j) /= sum;
                }
            }
        }
    }

    // Apply attention to values
    // [batch, heads, time, time] @ [batch, heads, time, d_head] -> [batch, heads, time, d_head]
    context_.resize({batch, N_HEADS, time, D_HEAD});
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < N_HEADS; h++) {
            for (size_t i = 0; i < time; i++) {
                for (size_t d = 0; d < D_HEAD; d++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < time; j++) {
                        sum += attn_weights_(b, h, i, j) * v_heads(b, h, j, d);
                    }
                    context_(b, h, i, d) = sum;
                }
            }
        }
    }

    // Reshape context to [batch, time, d_model]
    TensorF context_flat({batch, time, D_MODEL});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t h = 0; h < N_HEADS; h++) {
                for (size_t d = 0; d < D_HEAD; d++) {
                    context_flat(b, t, h * D_HEAD + d) = context_(b, h, t, d);
                }
            }
        }
    }

    // Output projection
    linear_no_bias(context_flat, linear_out_weight, D_MODEL, D_MODEL, output);
}

}  // namespace nemo
