#!/usr/bin/env python3
"""Trace joint network computation step by step and save intermediate values."""

import sys
sys.path.insert(0, '/var/data/nvidia-speech/test')

import torch
import numpy as np
from pathlib import Path

# Load model
from nemo.collections.asr.models import ASRModel

print("Loading model...")
model = ASRModel.restore_from("/var/data/nvidia-speech/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo")
model.eval()

device = next(model.parameters()).device
print(f"Model device: {device}")

# Load encoder output
enc_path = Path("/var/data/nvidia-speech/nemotron-speech.cpp/nemo_encoder_correct.bin")
enc_data = np.fromfile(enc_path, dtype=np.float32)
enc_out = torch.tensor(enc_data.reshape(1, -1, 1024)).to(device)
print(f"Encoder shape: {enc_out.shape}")

# Get decoder initial state
decoder = model.decoder
joint = model.joint

# Get LSTM from dec_rnn module
lstm = decoder.prediction['dec_rnn'].lstm

# Initialize LSTM state
batch = 1
hidden = torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=device)
cell = torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=device)

# Run decoder with blank token to get initial dec_out
blank_id = 1024
y = torch.tensor([[blank_id]], device=device)
y_embed = decoder.prediction["embed"](y)  # [1, 1, 640]
y_embed = y_embed.transpose(0, 1)  # [1, 1, 640] -> transpose for LSTM input
lstm_out, (hidden, cell) = lstm(y_embed, (hidden, cell))
lstm_out = lstm_out.transpose(0, 1)  # [1, 1, 640]
dec_out = lstm_out.squeeze(1)  # [1, 640] - no additional layer needed

print(f"\nDecoder output shape: {dec_out.shape}")
print(f"Dec out first 5: {dec_out[0, :5].tolist()}")

# Save decoder output
dec_out.detach().cpu().numpy().astype(np.float32).tofile("nemo_dec_out_frame0.bin")

# Now trace joint for frame 0 and frame 1
for frame_idx in [0, 1]:
    print(f"\n=== Frame {frame_idx} ===")

    # Extract encoder frame
    enc_frame = enc_out[:, frame_idx:frame_idx+1, :]  # [1, 1, 1024]
    print(f"Enc frame shape: {enc_frame.shape}")
    print(f"Enc frame first 5: {enc_frame[0, 0, :5].tolist()}")

    # Save encoder frame
    enc_frame.detach().cpu().numpy().astype(np.float32).tofile(f"nemo_enc_frame{frame_idx}.bin")

    # Joint network computation - step by step
    # 1. Encoder projection
    enc_proj = joint.enc(enc_frame)  # [1, 1, 640]
    print(f"Enc proj shape: {enc_proj.shape}")
    print(f"Enc proj first 5: {enc_proj[0, 0, :5].tolist()}")
    enc_proj.detach().cpu().numpy().astype(np.float32).tofile(f"nemo_enc_proj_frame{frame_idx}.bin")

    # 2. Decoder projection
    dec_proj = joint.pred(dec_out.unsqueeze(1))  # [1, 1, 640]
    print(f"Dec proj shape: {dec_proj.shape}")
    print(f"Dec proj first 5: {dec_proj[0, 0, :5].tolist()}")
    dec_proj.detach().cpu().numpy().astype(np.float32).tofile(f"nemo_dec_proj_frame{frame_idx}.bin")

    # 3. Sum
    sum_out = enc_proj + dec_proj  # [1, 1, 640]
    print(f"Sum first 5: {sum_out[0, 0, :5].tolist()}")
    sum_out.detach().cpu().numpy().astype(np.float32).tofile(f"nemo_sum_frame{frame_idx}.bin")

    # 4. ReLU (joint_net[0] is ReLU)
    relu_out = joint.joint_net[0](sum_out)  # ReLU
    print(f"ReLU first 5: {relu_out[0, 0, :5].tolist()}")
    relu_out.detach().cpu().numpy().astype(np.float32).tofile(f"nemo_relu_frame{frame_idx}.bin")

    # 5. Dropout (joint_net[1] - skipped in eval mode)
    # 6. Output projection (joint_net[2] is Linear)
    logits = joint.joint_net[2](relu_out)  # [1, 1, 1025]
    print(f"Logits shape: {logits.shape}")
    print(f"Logits first 5: {logits[0, 0, :5].tolist()}")
    print(f"Logits last 5: {logits[0, 0, -5:].tolist()}")
    logits.detach().cpu().numpy().astype(np.float32).tofile(f"nemo_logits_frame{frame_idx}.bin")

    # Argmax
    best = logits.argmax(dim=-1).item()
    print(f"Argmax: {best}")
    print(f"Blank score (1024): {logits[0, 0, 1024].item():.4f}")
    if best != 1024:
        print(f"Best token {best} score: {logits[0, 0, best].item():.4f}")

print("\nSaved all intermediate tensors!")
