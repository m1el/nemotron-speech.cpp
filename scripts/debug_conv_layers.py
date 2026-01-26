#!/usr/bin/env python3
"""
Debug ConvSubsampling by tracing each conv layer output.
"""

import numpy as np
import torch
import os

def load_bin(path):
    return np.fromfile(path, dtype=np.float32)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..")

    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr

    print("Loading NeMo model...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model.eval()
    device = next(model.parameters()).device

    # Load mel spectrogram
    mel_path = os.path.join(base_dir, "test.mel.bin")
    mel = load_bin(mel_path)
    time_steps = len(mel) // 128
    mel = mel.reshape(time_steps, 128)
    print(f"Mel shape: {mel.shape}")

    # Convert to torch [batch, time, features]
    mel_t = torch.tensor(mel, dtype=torch.float32).to(device)
    mel_t = mel_t.unsqueeze(0)  # [1, time, 128]
    mel_len = torch.tensor([time_steps]).to(device)

    print(f"\nInput to pre_encode: {mel_t.shape}")

    # Get the pre_encode module
    pre_encode = model.encoder.pre_encode

    # Trace through each layer
    with torch.no_grad():
        x = mel_t  # [1, time, 128]

        # Layer 0: CausalConv2D
        conv0 = pre_encode.conv[0]
        print(f"\nConv0 (CausalConv2D):")
        print(f"  in_channels={conv0.in_channels}, out_channels={conv0.out_channels}")
        print(f"  kernel_size={conv0.kernel_size}, stride={conv0.stride}")
        print(f"  left_padding={conv0._left_padding}, right_padding={conv0._right_padding}")

        # Reshape for 2D conv: [batch, time, features] -> [batch, 1, time, features]
        x = x.unsqueeze(1)  # [1, 1, time, 128]
        print(f"  Input shape: {x.shape}")

        x = conv0(x)
        print(f"  Output shape: {x.shape}")
        print(f"  Output stats: min={x.min():.4f}, max={x.max():.4f}")
        print(f"  Output [0,0,0,:5]: {x[0,0,0,:5].cpu().numpy()}")
        conv0_out = x.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [time, width, channels]
        conv0_out.astype(np.float32).tofile(os.path.join(base_dir, "nemo_conv0_out.bin"))

        # Layer 1: ReLU
        x = pre_encode.conv[1](x)
        print(f"\nReLU1 output shape: {x.shape}")

        # Layer 2: CausalConv2D (depthwise)
        conv2 = pre_encode.conv[2]
        print(f"\nConv2 (CausalConv2D depthwise):")
        print(f"  groups={conv2.groups}")
        x = conv2(x)
        print(f"  Output shape: {x.shape}")
        print(f"  Output stats: min={x.min():.4f}, max={x.max():.4f}")

        # Layer 3: Conv2d (pointwise)
        x = pre_encode.conv[3](x)
        print(f"\nConv3 (pointwise) output shape: {x.shape}")

        # Layer 4: ReLU
        x = pre_encode.conv[4](x)
        print(f"ReLU4 output shape: {x.shape}")

        # Layer 5: CausalConv2D (depthwise)
        conv5 = pre_encode.conv[5]
        x = conv5(x)
        print(f"\nConv5 (CausalConv2D depthwise) output shape: {x.shape}")

        # Layer 6: Conv2d (pointwise)
        x = pre_encode.conv[6](x)
        print(f"Conv6 (pointwise) output shape: {x.shape}")

        # Layer 7: ReLU
        x = pre_encode.conv[7](x)
        print(f"ReLU7 output shape: {x.shape}")

        # Final: reshape and linear
        b, c, t, f = x.shape
        print(f"\nBefore reshape: [batch={b}, channels={c}, time={t}, features={f}]")
        print(f"Reshape to: [batch={b}, time={t}, channels*features={c*f}]")

        # NeMo's reshape: x.transpose(1, 2).reshape(b, t, -1)
        # This is [b, c, t, f] -> [b, t, c, f] -> [b, t, c*f]
        x_reshaped = x.transpose(1, 2).reshape(b, t, -1)
        print(f"After reshape: {x_reshaped.shape}")
        print(f"Before linear [0,0,:10]: {x_reshaped[0,0,:10].cpu().numpy()}")
        print(f"Before linear [0,-1,:10]: {x_reshaped[0,-1,:10].cpu().numpy()}")

        # Linear layer
        x_out = pre_encode.out(x_reshaped)
        print(f"\nFinal output shape: {x_out.shape}")
        print(f"Final output [0,0,:5]: {x_out[0,0,:5].cpu().numpy()}")
        print(f"Final output [0,-1,:5]: {x_out[0,-1,:5].cpu().numpy()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
