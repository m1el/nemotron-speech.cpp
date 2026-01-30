# GGML to NeMo Implementation Mapping

This document provides detailed mapping between the ggml C++ implementation and NeMo Python implementation.

## Encoder Flow

### 1. ConvSubsampling (mel → encoder input)

| ggml (nemo-ggml.cpp) | NeMo Python | Weights |
|---------------------|-------------|---------|
| `build_conv_subsampling()` | `ConformerEncoder.pre_encode()` → `ConvSubsampling.forward()` | |
| `build_causal_conv2d(conv0)` | Conv2D(1→256, k=3, s=2) + ReLU | `encoder.pre_encode.conv.0.{weight,bias}` |
| `build_causal_dw_conv2d(conv2)` | DepthwiseConv2D(256, k=3, s=2) | `encoder.pre_encode.conv.2.{weight,bias}` |
| `ggml_conv_2d(conv3)` | PointwiseConv2D(256→256, k=1) + ReLU | `encoder.pre_encode.conv.3.{weight,bias}` |
| `build_causal_dw_conv2d(conv5)` | DepthwiseConv2D(256, k=3, s=2) | `encoder.pre_encode.conv.5.{weight,bias}` |
| `ggml_conv_2d(conv6)` | PointwiseConv2D(256→256, k=1) + ReLU | `encoder.pre_encode.conv.6.{weight,bias}` |
| `ggml_mul_mat(out)` | Linear(W*256 → d_model) | `encoder.pre_encode.out.{weight,bias}` |

**Input**: `[n_mels=128, time, batch=1]`
**Output**: `[d_model=1024, time/8, batch=1]`

### 2. Positional Encoding

| ggml | NeMo Python |
|------|-------------|
| `compute_pos_emb()` (precomputed sinusoidal) | `ConformerEncoder.pos_enc` → `RelPositionalEncoding` |
| `model.pos_emb` tensor `[d_model, 2*max_len-1]` | Dynamic computation based on sequence length |

### 3. Conformer Layers (x24)

Each layer follows: **FFN1 → Attention → Conv → FFN2 → Final LN**

#### 3.1 FFN1 Module

| ggml | NeMo | Weights |
|------|------|---------|
| `build_layer_norm()` | `norm_feed_forward1` | `encoder.layers.{i}.norm_feed_forward1.{weight,bias}` |
| `build_ffn()` | `feed_forward1` | `encoder.layers.{i}.feed_forward1.linear1.weight` |
| Linear → SiLU → Linear | Linear → Swish → Linear | `encoder.layers.{i}.feed_forward1.linear2.weight` |
| `ggml_scale(0.5)` + residual | Same | |

#### 3.2 Self-Attention Module

| ggml | NeMo | Weights |
|------|------|---------|
| `build_layer_norm()` | `norm_self_att` | `encoder.layers.{i}.norm_self_att.{weight,bias}` |
| `build_rel_pos_mha()` / `build_cached_rel_pos_mha()` | `self_attn` (RelPositionMultiHeadAttention) | |
| Q projection | `linear_q` | `encoder.layers.{i}.self_attn.linear_q.weight` |
| K projection | `linear_k` | `encoder.layers.{i}.self_attn.linear_k.weight` |
| V projection | `linear_v` | `encoder.layers.{i}.self_attn.linear_v.weight` |
| Pos projection | `linear_pos` | `encoder.layers.{i}.self_attn.linear_pos.weight` |
| `build_rel_shift()` | `rel_shift()` | |
| Output projection | `linear_out` | `encoder.layers.{i}.self_attn.linear_out.weight` |
| `bias_u`, `bias_v` | `pos_bias_u`, `pos_bias_v` | `encoder.layers.{i}.self_attn.pos_bias_{u,v}` |

#### 3.3 Convolution Module

| ggml | NeMo | Weights |
|------|------|---------|
| `build_layer_norm()` | `norm_conv` | `encoder.layers.{i}.norm_conv.{weight,bias}` |
| `build_conformer_conv()` / `build_cached_causal_conv1d()` | `conv` (ConformerConvolution) | |
| Pointwise Conv1 (pw1) | `pointwise_conv1` | `encoder.layers.{i}.conv.pointwise_conv1.weight` |
| GLU activation | GLU | |
| Depthwise Conv (dw) | `depthwise_conv` | `encoder.layers.{i}.conv.depthwise_conv.weight` |
| LayerNorm | `batch_norm` | `encoder.layers.{i}.conv.batch_norm.{weight,bias}` |
| SiLU | Swish | |
| Pointwise Conv2 (pw2) | `pointwise_conv2` | `encoder.layers.{i}.conv.pointwise_conv2.weight` |

#### 3.4 FFN2 Module

Same structure as FFN1, uses `norm_feed_forward2` and `feed_forward2`.

#### 3.5 Final Layer Norm

| ggml | NeMo | Weights |
|------|------|---------|
| `build_layer_norm()` | `norm_out` | `encoder.layers.{i}.norm_out.{weight,bias}` |

## Decoder Flow (RNNT Prediction Network)

### 1. LSTM Decoder

| ggml | NeMo | Weights |
|------|------|---------|
| `build_decoder_step()` | `RNNTDecoder.forward()` | |
| Token embedding lookup | `embed` | `decoder.prediction.embed.weight` |
| `build_lstm_cell()` layer 0 | LSTM layer 0 | `decoder.prediction.dec_rnn.lstm.{weight,bias}_{ih,hh}_l0` |
| `build_lstm_cell()` layer 1 | LSTM layer 1 | `decoder.prediction.dec_rnn.lstm.{weight,bias}_{ih,hh}_l1` |

**Hidden size**: 640, **Num layers**: 2

### 2. Joint Network

| ggml | NeMo | Weights |
|------|------|---------|
| `build_joint()` | `RNNTJoint.forward()` | |
| Encoder projection | `joint.enc` | `joint.enc.{weight,bias}` |
| Decoder projection | `joint.pred` | `joint.pred.{weight,bias}` |
| Add + ReLU | Same | |
| Output projection | `joint.joint_net.2` | `joint.joint_net.2.{weight,bias}` |

## Streaming State Tensors

### Attention Cache (cache_last_channel)

| ggml | NeMo | Shape |
|------|------|-------|
| `k_cache_ins[l]`, `k_cache_outs[l]` | `cache_last_channel` K component | `[d_model, att_left_context]` per layer |
| `v_cache_ins[l]`, `v_cache_outs[l]` | `cache_last_channel` V component | `[d_model, att_left_context]` per layer |

NeMo shape: `[n_layers=24, batch=1, att_context_size[0]=70, d_model=1024]`

### Convolution Cache (cache_last_time)

| ggml | NeMo | Shape |
|------|------|-------|
| `conv_cache_ins[l]`, `conv_cache_outs[l]` | `cache_last_time` | `[d_model, conv_kernel_size-1=8]` per layer |

NeMo shape: `[n_layers=24, batch=1, d_model=1024, conv_context_size[0]=8]`

### Decoder State

| ggml | NeMo |
|------|------|
| `nemo_decoder_state.h` | LSTM hidden states |
| `nemo_decoder_state.c` | LSTM cell states |
| `nemo_decoder_state.prev_token` | Previous hypothesis token |

## Streaming Chunk Sizes

Based on `att_context_size = [left_chunks, right_context]`:

| att_context_size | mel_chunk_frames | latency |
|-----------------|------------------|---------|
| `[70, 0]` | 17 | ~80ms (pure causal) |
| `[70, 1]` | 25 | ~160ms |
| `[70, 6]` | 65 | ~560ms |
| `[70, 13]` | 121 | ~1.12s |

Formula: `mel_chunk_frames = (right_context + 2) * 8 + 1`

## Key Functions Mapping

| ggml Function | NeMo Python Location |
|--------------|---------------------|
| `nemo_stream_process_incremental()` | `ASRModuleMixin.conformer_stream_step()` |
| `build_streaming_encoder()` | `StreamingEncoder.cache_aware_stream_step()` |
| `build_cached_conformer_layer()` | `ConformerLayer.forward()` with caching |
| `process_mel_chunk_streaming()` | `ConformerEncoder.forward()` |
| `decode_one_step()` | `GreedyBatchedRNNTInfer.forward()` |
| `greedy_decode_with_state()` | `RNNTBPEDecoding.rnnt_decoder_predictions_tensor()` |
