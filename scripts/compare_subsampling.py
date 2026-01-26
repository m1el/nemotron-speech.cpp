#!/usr/bin/env python3
"""Compare ConvSubsampling output between C++ and NeMo."""

import numpy as np
import torch

def load_mel_bin(path):
    data = np.fromfile(path, dtype=np.float32)
    time = len(data) // 128
    return data.reshape(time, 128)

def main():
    import sys
    import os

    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr

    print("Loading NeMo model...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model.eval()
    device = next(model.parameters()).device

    # Load mel - find it relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mel_path = os.path.join(script_dir, "..", "test.mel.bin")
    mel = load_mel_bin(mel_path)
    print(f"Mel shape: {mel.shape}")

    # Convert to NeMo format: [batch, features, time]
    mel_t = torch.tensor(mel, dtype=torch.float32).to(device)
    mel_t = mel_t.T.unsqueeze(0)  # [1, 128, 2000]
    mel_len = torch.tensor([mel.shape[0]]).to(device)

    print(f"Mel tensor shape: {mel_t.shape}")

    # Get subsampling module
    pre_encode = model.encoder.pre_encode
    print(f"pre_encode type: {type(pre_encode)}")

    # Run subsampling
    with torch.no_grad():
        # pre_encode expects [batch, features, time]
        sub_out, sub_len = pre_encode(mel_t, mel_len)

    print(f"Subsampling output shape: {sub_out.shape}")
    sub_np = sub_out.squeeze(0).cpu().numpy()
    # NeMo: [features, time] or [time, features]?
    print(f"sub_np shape: {sub_np.shape}")

    if sub_np.shape[0] == 1024:
        # [features, time] -> [time, features]
        sub_np = sub_np.T

    print(f"Subsampling output shape (transposed): {sub_np.shape}")
    print(f"Stats: min={sub_np.min():.4f}, max={sub_np.max():.4f}, mean={sub_np.mean():.4f}")
    print(f"First 5 values of first frame: {sub_np[0, :5]}")

    # Save for comparison
    out_path = os.path.join(script_dir, "..", "nemo_subsampling_out.bin")
    sub_np.astype(np.float32).tofile(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
