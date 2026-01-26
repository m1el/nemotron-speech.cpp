#!/usr/bin/env python3
"""
Layer-by-layer comparison between C++ and NeMo implementations.
"""

import numpy as np
import torch
import os
import soundfile as sf
from scipy import signal

def load_bin(path, shape=None):
    """Load binary float32 file."""
    data = np.fromfile(path, dtype=np.float32)
    if shape:
        data = data.reshape(shape)
    return data

def compare(name, cpp_data, nemo_data, rtol=1e-4, atol=1e-5):
    """Compare two arrays and report differences."""
    if cpp_data.shape != nemo_data.shape:
        print(f"  {name}: SHAPE MISMATCH - cpp={cpp_data.shape}, nemo={nemo_data.shape}")
        return False

    diff = np.abs(cpp_data - nemo_data)
    max_diff = diff.max()
    mean_diff = diff.mean()

    close = np.allclose(cpp_data, nemo_data, rtol=rtol, atol=atol)
    status = "PASS" if close else "FAIL"

    print(f"  {name}: {status}")
    print(f"    Shape: {cpp_data.shape}")
    print(f"    Max abs diff: {max_diff:.6e}, Mean abs diff: {mean_diff:.6e}")
    print(f"    CPP:  first 5 = {cpp_data.flat[:5]}")
    print(f"    NeMo: first 5 = {nemo_data.flat[:5]}")

    if not close:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"    Max diff at index {idx}: cpp={cpp_data[idx]:.6f}, nemo={nemo_data[idx]:.6f}")

    return close


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

    # Load audio file
    audio_path = os.path.join(base_dir, "..", "test", "HFTKzy5xRM-cut.wav")
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        for p in ["/var/data/nvidia-speech/test/HFTKzy5xRM-cut.wav",
                  "/var/data/nvidia-speech/test/test.wav"]:
            if os.path.exists(p):
                audio_path = p
                break

    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)
        sr = 16000

    print(f"Audio: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f}s)")

    audio_t = torch.tensor(audio, dtype=torch.float32).to(device).unsqueeze(0)
    audio_len = torch.tensor([len(audio)]).to(device)

    print("\n" + "="*60)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*60)

    # ============================================================
    # 1. Preprocessor (mel spectrogram)
    # ============================================================
    print("\n[1] Preprocessor (Mel Spectrogram)")
    print("-"*40)

    with torch.no_grad():
        mel, mel_len = model.preprocessor(
            input_signal=audio_t,
            length=audio_len
        )

    # mel: [batch, features, time] from preprocessor
    nemo_mel = mel.squeeze(0).cpu().numpy()  # [128, time]
    nemo_mel = nemo_mel.T  # [time, 128] for easier comparison
    print(f"  NeMo mel shape: {nemo_mel.shape}")
    print(f"  NeMo mel stats: min={nemo_mel.min():.4f}, max={nemo_mel.max():.4f}")
    print(f"  NeMo mel first 5: {nemo_mel[0, :5]}")
    nemo_mel.astype(np.float32).tofile(os.path.join(base_dir, "nemo_mel.bin"))

    # Load C++ mel
    cpp_mel_path = os.path.join(base_dir, "test.mel.bin")
    if os.path.exists(cpp_mel_path):
        cpp_mel = load_bin(cpp_mel_path)
        cpp_mel = cpp_mel.reshape(-1, 128)
        compare("Mel Spectrogram", cpp_mel, nemo_mel)

    # ============================================================
    # 2. ConvSubsampling
    # ============================================================
    print("\n[2] ConvSubsampling")
    print("-"*40)

    with torch.no_grad():
        # Encoder transposes to [batch, time, features] before calling pre_encode
        mel_transposed = mel.transpose(1, 2)  # [1, time, 128]
        sub_out, sub_len = model.encoder.pre_encode(x=mel_transposed, lengths=mel_len)
        sub_len = sub_len.to(torch.int64)

    # sub_out: [batch, time, dim]
    nemo_sub = sub_out.squeeze(0).cpu().numpy()  # [time, 1024]
    print(f"  NeMo subsampling shape: {nemo_sub.shape}")
    print(f"  NeMo subsampling stats: min={nemo_sub.min():.4f}, max={nemo_sub.max():.4f}")
    print(f"  NeMo subsampling first 5: {nemo_sub[0, :5]}")
    nemo_sub.astype(np.float32).tofile(os.path.join(base_dir, "nemo_subsampling_out.bin"))

    # Load C++ subsampling
    cpp_sub_path = os.path.join(base_dir, "cpp_subsampling_out.bin")
    if os.path.exists(cpp_sub_path):
        cpp_sub = load_bin(cpp_sub_path)
        cpp_sub = cpp_sub.reshape(-1, 1024)
        compare("ConvSubsampling", cpp_sub, nemo_sub)

    # ============================================================
    # 3. Positional Encoding
    # ============================================================
    print("\n[3] Positional Encoding")
    print("-"*40)

    with torch.no_grad():
        pos_out, pos_emb = model.encoder.pos_enc(sub_out)

    nemo_pos_out = pos_out.squeeze(0).cpu().numpy()  # [time, 1024]
    nemo_pos_emb = pos_emb.squeeze(0).cpu().numpy()  # [2*time-1, 1024]
    print(f"  NeMo pos_out shape: {nemo_pos_out.shape}")
    print(f"  NeMo pos_emb shape: {nemo_pos_emb.shape}")
    print(f"  NeMo pos_out first 5: {nemo_pos_out[0, :5]}")
    print(f"  NeMo pos_emb first 5: {nemo_pos_emb[0, :5]}")
    nemo_pos_out.astype(np.float32).tofile(os.path.join(base_dir, "nemo_pos_out.bin"))
    nemo_pos_emb.astype(np.float32).tofile(os.path.join(base_dir, "nemo_pos_emb.bin"))

    # ============================================================
    # 4. First Conformer Layer (Layer 0)
    # ============================================================
    print("\n[4] Conformer Layer 0")
    print("-"*40)

    layer0 = model.encoder.layers[0]

    with torch.no_grad():
        x = pos_out  # [batch, time, dim]

        # 4a. Feed Forward 1
        x_norm = layer0.norm_feed_forward1(x)
        ff1_out = layer0.feed_forward1(x_norm)
        x_ff1 = x + 0.5 * ff1_out

        nemo_ff1 = x_ff1.squeeze(0).cpu().numpy()
        print(f"  After FF1 shape: {nemo_ff1.shape}")
        print(f"  After FF1 first 5: {nemo_ff1[0, :5]}")
        nemo_ff1.astype(np.float32).tofile(os.path.join(base_dir, "nemo_layer0_ff1.bin"))

        # 4b. Self-Attention
        x_norm = layer0.norm_self_att(x_ff1)
        attn_out = layer0.self_attn(x_norm, x_norm, x_norm, mask=None, pos_emb=pos_emb)
        x_attn = x_ff1 + attn_out

        nemo_attn = x_attn.squeeze(0).cpu().numpy()
        print(f"  After Attn shape: {nemo_attn.shape}")
        print(f"  After Attn first 5: {nemo_attn[0, :5]}")
        nemo_attn.astype(np.float32).tofile(os.path.join(base_dir, "nemo_layer0_attn.bin"))

        # 4c. Convolution
        x_norm = layer0.norm_conv(x_attn)
        conv_out = layer0.conv(x_norm)
        x_conv = x_attn + conv_out

        nemo_conv = x_conv.squeeze(0).cpu().numpy()
        print(f"  After Conv shape: {nemo_conv.shape}")
        print(f"  After Conv first 5: {nemo_conv[0, :5]}")
        nemo_conv.astype(np.float32).tofile(os.path.join(base_dir, "nemo_layer0_conv.bin"))

        # 4d. Feed Forward 2
        x_norm = layer0.norm_feed_forward2(x_conv)
        ff2_out = layer0.feed_forward2(x_norm)
        x_ff2 = x_conv + 0.5 * ff2_out

        nemo_ff2 = x_ff2.squeeze(0).cpu().numpy()
        print(f"  After FF2 shape: {nemo_ff2.shape}")
        print(f"  After FF2 first 5: {nemo_ff2[0, :5]}")
        nemo_ff2.astype(np.float32).tofile(os.path.join(base_dir, "nemo_layer0_ff2.bin"))

        # 4e. Output norm
        x_out = layer0.norm_out(x_ff2)

        nemo_layer0 = x_out.squeeze(0).cpu().numpy()
        print(f"  After norm_out shape: {nemo_layer0.shape}")
        print(f"  After norm_out first 5: {nemo_layer0[0, :5]}")
        nemo_layer0.astype(np.float32).tofile(os.path.join(base_dir, "nemo_layer0_out.bin"))

    # ============================================================
    # 5. Full Encoder Output
    # ============================================================
    print("\n[5] Full Encoder")
    print("-"*40)

    with torch.no_grad():
        enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)

    # enc_out: [batch, dim, time] -> transpose to [time, dim]
    nemo_enc = enc_out.squeeze(0).cpu().numpy()
    if nemo_enc.shape[0] == 1024:
        nemo_enc = nemo_enc.T  # [time, 1024]
    print(f"  NeMo encoder shape: {nemo_enc.shape}")
    print(f"  NeMo encoder stats: min={nemo_enc.min():.4f}, max={nemo_enc.max():.4f}")
    print(f"  NeMo encoder first 5: {nemo_enc[0, :5]}")
    nemo_enc.astype(np.float32).tofile(os.path.join(base_dir, "nemo_encoder_out.bin"))

    # Load C++ encoder
    cpp_enc_path = os.path.join(base_dir, "cpp_encoder_out.bin")
    if os.path.exists(cpp_enc_path):
        cpp_enc = load_bin(cpp_enc_path)
        cpp_enc = cpp_enc.reshape(-1, 1024)
        compare("Full Encoder", cpp_enc, nemo_enc)

    # ============================================================
    # 6. Full Transcription
    # ============================================================
    print("\n[6] Full Transcription")
    print("-"*40)

    with torch.no_grad():
        transcription = model.transcribe([audio_path])
        print(f"  NeMo transcription: {transcription}")

    print("\n" + "="*60)
    print("Generated NeMo reference files in", base_dir)
    print("="*60)


if __name__ == "__main__":
    main()
