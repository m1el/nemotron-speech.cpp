# Implementation Status

**Last Updated**: 2026-01-26

## Current State: WORKING

The C++ port of NVIDIA's NeMo ASR model (nemotron-speech-streaming-en-0.6b) is fully functional and produces correct transcriptions.

### Test Output
```
./nemotron-speech test.mel.bin

Mel shape: [1, 2000, 128]
Tokens (121): 130 41 23 115 65 45 77 210 ...
Text: So you might have heard that there's quite a bit of hype around
artificial intelligence and math right now. And, you know, I will admit
I've been guilty of hyping it a little bit because in the grand scheme
of things, we are getting to the point where advanced math is going to
be commoditized. We are very clearly on that track

Inference time: ~140s for 20s audio (7x real-time, unoptimized)
```

### Component Status
| Component | Status |
|-----------|--------|
| Mel spectrogram | PASS |
| ConvSubsampling | PASS |
| Positional Encoding | PASS |
| Conformer Encoder (24 layers) | PASS |
| RNN-T Decoder (LSTM) | PASS |
| RNN-T Joint | PASS |
| Greedy Decoding | PASS |
| SentencePiece Tokenizer | PASS |

### Bugs Fixed

#### 1. rel_shift Implementation (Jan 26)
The index formula for relative position shift was inverted:
```cpp
// Wrong:  k = qlen - 1 - j + i
// Correct: k = qlen - 1 + j - i
```

#### 2. Tokenizer (Jan 26)
Changed from WordPiece (`##`) to SentencePiece (`▁`) format.

## Build

```bash
make clean && make all
./nemotron-speech test.mel.bin
```

## Architecture

- **Encoder**: ConvSubsampling (8x) + 24 Conformer layers
- **Decoder**: LSTM with embedding
- **Joint**: Encoder + Decoder projection with ReLU
- **Decoding**: Greedy RNN-T

See `arch.md` for detailed architecture.
