# Streaming Verification Status

**Current issue**: GGML streaming produces empty transcription. MEL inputs verified correct, so bug is in encoder→decoder pipeline.

## Verification Checklist

| Layer | Status | Notes |
|-------|--------|-------|
| MEL Input | ✅ PASS | 94% exact match, max diff 1.9e-06 |
| ConvSubsampling | ✅ PASS | Matches NeMo, corr ≥ 0.999 |
| Conformer Layers | ✅ PASS | All 24 layers match, corr ≥ 0.997 |
| Encoder Output | ✅ PASS | 250 chunks match NeMo |
| Decoder (LSTM) | ✅ PASS | Multiple token bug fixed |
| Joint Network | ✅ PASS | Same as batch mode |

## ✅ RESOLVED: Streaming Now Working (2026-01-30)

The streaming transcriber now produces correct output matching batch transcription.

**Root cause found and fixed**: The `decode_one_step` function could emit multiple tokens per encoder frame in RNN-T decoding, but only the last token was returned. See [STREAMING_CACHE_INVESTIGATION.md](../STREAMING_CACHE_INVESTIGATION.md) for details.

**Performance**: Real-time factor 0.248x (4x faster than real-time)

---

# Reference

Example streaming implemented in `scripts/my_streaming.py`.
Run using `uv run scripts/my_streaming.py`.

# Model config audio:
sample_rate: 16000
window_size: 0.025sec / 400samples
window_stride: 0.01sec / 160samples

Audio chunking is defined by `att_context_size = (left_chunks_num, chunk_size)`

`cache_last_channel, cache_last_time, cache_last_channel_len`
is the streaming state preserved between the runs.

mel chunking: 121

The sizes of these tensors are defined by
n_layers: 24
d_model: 1024
conv_kernel_size: 9
conv_context_size: 'causal' -> [(conv_kernel_size-1), 0] -> [8, 0]
0  => chunk_size=[1, 8], valid_out_len=1,
1  => chunk_size=[9, 16], valid_out_len=2,
6  => chunk_size=[49, 56], valid_out_len=7,
13 => chunk_size=[105, 112], valid_out_len=14,

cache_last_channel = create_tensor(
    (
        model_config.n_layers, # 24
        batch_size,            # 1
        att_context_size[0],   # 70 self.streaming_cfg.last_channel_cache_size
        model_config.d_model,  # 1024
    )
)
cache_last_time = create_tensor(
    (
        model_config.n_layers,             # 24
        batch_size,                        # 1
        d_model,                           # 1024
        # last_time_cache_size
        model_config.conv_context_size[0], # 8
    ),
    device=device,
    dtype=dtype,
)
cache_last_channel_len = torch.zeros(batch_size) # 1

# starting point for NeMo audio streaming:
nemotron-speech.cpp/scripts/my_streaming.py
run with:
    cd /var/data/nvidia-speech/nemotron-speech.cpp
    uv run scripts/my_streaming.py

# Location of live installed python code, modify to look into the data:
nemotron-speech.cpp/.venv/lib/python3.10/site-packages/nemo

# Data flow for mel chunks -> tokens
nemo/collections/asr/parts/mixins/mixins.py:
ASRModuleMixin.conformer_stream_step

## Encoder
    nemo/collections/asr/parts/mixins/streaming.py
    StreamingEncoder.cache_aware_stream_step (inherited)
    nemo/collections/asr/modules/conformer_encoder.py:
    ConformerEncoder.cache_aware_stream_step
    ConformerEncoder.forward
    ConformerEncoder.pre_encode
        nemo/collections/asr/parts/submodules/subsampling.py
        ConvSubsampling.forward
    ConformerEncoder.pos_enc
        nemo/collections/asr/parts/submodules/multi_head_attention.py
        RelPositionalEncoding      [!!!] positional depends on the length, do we do that?
    ConformerEncoder._create_masks [!!!] not implemented, do we need this?
    ConformerEncoder.layers
        nemo/collections/asr/parts/submodules/conformer_modules.py
        ConformerLayer.forward
            ??? if this works completely, no need to go deeper

## Decoder:
    nemo/collections/asr/parts/submodules/rnnt_decoding.py
    RNNTBPEDecoding.rnnt_decoder_predictions_tensor
    nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py
    GreedyBatchedRNNTInfer.forward

audio_signal [1, 17, 128] ->
pre_encode [1, 3, 1024] ->
# drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor = 2
drop_extra_pre_encoded [1, 1, 1024] ->
pos_enc(cache_len) ([1, 1, 1024], [1, 141, 1024])

## THE PLAN

You need to verify and fix if necessary the ggml implementation of nemotron-speech, residing in nemotron-speech.cpp/src.
You can build the code using `make` in the nemotron-speech.cpp directory

0) Look at the ggml implementation in `src/nemo-ggml.cpp` and `src/nemo-stream.cpp`

1) Map the ggml implementation to the NeMo implementation, and write the documentation in this file

2) create a python script to compare binary dumps of tensors
    the script accepts two file names and tensor shape

3) Istrument and verify these encoder and decoder steps:
3.1) ConformerEncoder.pre_encode
3.2) ConvSubsampling.forward
3.3) ConformerEncoder.pos_enc
3.4) ConformerEncoder.layers
3.5) RNNTBPEDecoding.rnnt_decoder_predictions_tensor
3.6) GreedyBatchedRNNTInfer.forward

# for every step of the encoder and decoder:
1) instrument the python implementation (see below for the pattern)
2) instrument the ggml implementation (see below for the pattern)
3) produce outputs
```bash
# running NeMo
uv run scripts/my_streaming.py
# running ggml
make transcribe_stream && ./transcribe_stream weights/model.gguf data/audio_test.pcm 70 0
```
4) Compare the output using `scripts/compare_tensors.py` tool (see below)
5) Clear the instrumentation from python and cpp
6) document the status to this document
---

## SUPER FUCKING IMPORTANT

The joint decoder doesn't work like you expect.
my_stream.py:352: call asr_model.conformer_stream_step
nemo/collections/asr/parts/mixins/mixins.py:592
ASRModuleMixin.conformer_stream_step (call rnnt_decoder_predictions_tensor on line 707)
AbstractRNNTDecoding.rnnt_decoder_predictions_tensor -> (call on line 719)
nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py:529
nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py:736
GreedyBatchedRNNTInfer.forward -> (call on line 767)
GreedyBatchedRNNTInfer._greedy_decode_blank_as_pad_loop_labels:779 -> (call self.decoding_computer on line 817)
nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:179
nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:666
GreedyBatchedRNNTLabelLoopingComputer.cuda_graphs_impl

# GGML to NeMo Implementation Mapping

See [GGML_NEMO_MAPPING.md](GGML_NEMO_MAPPING.md) for detailed tables mapping:
- ConvSubsampling layers and weights
- Conformer layer structure (FFN1 → Attention → Conv → FFN2)
- Decoder LSTM and Joint network
- Streaming state tensors (K/V cache, conv cache)
- Chunk size formulas

**Quick reference:**
- Encoder: `build_conv_subsampling()` → `build_conformer_layer()` x24
- Decoder: `build_decoder_step()` (2-layer LSTM) → `build_joint()`
- Chunk formula: `mel_frames = (right_context + 2) * 8 + 1`

---

# Verification Status

## Tensor Comparison Tool

Use `scripts/compare_tensors.py` to compare binary tensor dumps:

```bash
python scripts/compare_tensors.py <nemo_dump> <ggml_dump>
# Example:
python scripts/compare_tensors.py my_bin/nemo_mel_input.bin my_bin/ggml_mel_input.bin
```

The tool provides:
- Difference statistics (max, mean, median absolute/relative diff)
- Log-scale histogram showing distribution of differences
- Percentile summary (50%, 90%, 95%, 99%, 99.9%)

## Verified Layers

### ✅ MEL Input (2025-01-30)

**Result**: PASS

| Metric | Value |
|--------|-------|
| Exact match | 94.12% |
| Max absolute diff | 1.9e-06 |
| Mean absolute diff | 8.2e-08 |
| 99% of diffs | ≤ 1.9e-06 |

All differences are within floating-point precision. The mel spectrogram preprocessing produces identical results.

### ⬜ ConvSubsampling (pre_encode)

Not yet verified.

### ⬜ Conformer Layers

Not yet verified.

### ⬜ Decoder (LSTM)

Not yet verified.

### ⬜ Joint Network

Not yet verified.

---

# Notes for Future Work

## Critical: GGML vs PyTorch Tensor Layout

**GGML stores dimensions in REVERSE order compared to PyTorch.**

- GGML: `ne[0]` is the **innermost/contiguous** dimension
- PyTorch: last dimension is innermost/contiguous

Example for mel input:
- PyTorch shape `[batch=1, n_mels=128, time=17]` → time is contiguous
- GGML shape `ne = [128, 17, 1]` → n_mels (ne[0]) is contiguous

To dump PyTorch tensor for comparison with GGML:
```python
# PyTorch [batch, mels, time] → need mels innermost for GGML [mels, time, batch]
mel_for_dump = chunk_audio.permute(0, 2, 1).contiguous()  # → [batch, time, mels]
```

## Running Tests

```bash
# NeMo Python streaming (uses ../test/HFTKzy5xRM-cut.wav)
cd nemotron-speech.cpp
uv run scripts/my_streaming.py

# GGML streaming (convert wav to pcm first if needed)
ffmpeg -i ../test/HFTKzy5xRM-cut.wav -f s16le -ar 16000 -ac 1 data/audio.pcm
./transcribe_stream weights/model.gguf data/audio.pcm 70 0
```

## Instrumentation Pattern

### Python (NeMo)
Modify `scripts/my_streaming.py:20`
```python
def mk_hook(module_name):
    def hook(module, args, kwargs, output):
        # example instrumentation
        if module_name == 'ASRModel.encoder.pre_encode':
            dump_append_data(kwargs["x"], 'my_bin/nemo_subsampling_input.bin')
            dump_append_data(output[0], 'my_bin/nemo_subsampling_output.bin')
```

### C++ (GGML)
```cpp
// Use existing helper (clears file on first call automatically):
int64_t shape[4] = {x,y,z,w};
append_dump_array(data_ptr, n_elements, &shape[0], "my_bin/ggml_<tensor_name>.bin");

// Or for named tensors in graph:
append_dump_tensor(ctx, "tensor_name", "my_bin/ggml_<tensor_name>.bin");
```

## Key File Locations

| Purpose | Location |
|---------|----------|
| NeMo streaming script | `scripts/my_streaming.py` |
| Tensor comparison tool | `scripts/compare_tensors.py` |
| GGML streaming impl | `src/nemo-stream.cpp` |
| GGML batch impl | `src/nemo-ggml.cpp` |
| NeMo installed code | `.venv/lib/python3.10/site-packages/nemo/` |
| Binary dumps | `my_bin/` |

## Known Issues

All major issues have been resolved:

1. ~~**Empty transcription from GGML streaming**~~: ✅ FIXED - Multiple token emission bug in decoder
2. ~~**Chunk count mismatch**~~: ✅ FIXED - Both NeMo and GGML now produce 250 chunks
3. ~~**Encoder divergence after chunk 0**~~: ✅ FIXED - Conv cache and attention cache issues resolved

### Minor Remaining Differences
- Streaming output may have small differences from batch (e.g., missing filler words like "you know")
- This is expected behavior for streaming vs batch processing in RNN-T models