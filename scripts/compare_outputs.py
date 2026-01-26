#!/usr/bin/env python3
"""
Compare C++ and NeMo outputs at each stage of the pipeline.
"""

import numpy as np
import torch
import argparse


def load_mel_bin(path):
    """Load mel features from binary file."""
    data = np.fromfile(path, dtype=np.float32)
    time = len(data) // 128
    return data.reshape(time, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--mel", help="Mel binary file to verify")
    args = parser.parse_args()

    # Load NeMo model
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    from scipy import signal

    print("Loading NeMo model...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model.eval()
    device = next(model.parameters()).device

    # Load and preprocess audio
    print(f"Loading audio: {args.audio}")
    audio, sr = sf.read(args.audio)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)

    audio_t = torch.tensor(audio, dtype=torch.float32).to(device)
    audio_len = torch.tensor([len(audio)]).to(device)

    # Get mel features from NeMo
    with torch.no_grad():
        mel, mel_len = model.preprocessor(
            input_signal=audio_t.unsqueeze(0),
            length=audio_len
        )

    mel_np = mel.squeeze(0).transpose(0, 1).cpu().numpy()
    print(f"NeMo mel shape: {mel_np.shape}")
    print(f"NeMo mel stats: min={mel_np.min():.4f}, max={mel_np.max():.4f}, mean={mel_np.mean():.4f}")
    print(f"NeMo mel first 5 values: {mel_np[0, :5]}")

    # Compare with loaded mel if provided
    if args.mel:
        loaded_mel = load_mel_bin(args.mel)
        print(f"\nLoaded mel shape: {loaded_mel.shape}")
        print(f"Loaded mel stats: min={loaded_mel.min():.4f}, max={loaded_mel.max():.4f}, mean={loaded_mel.mean():.4f}")
        print(f"Loaded mel first 5 values: {loaded_mel[0, :5]}")

        diff = np.abs(mel_np - loaded_mel).max()
        print(f"Max diff: {diff:.6f}")

    # Run encoder
    print("\nRunning encoder...")
    with torch.no_grad():
        # mel is [batch, features, time], need to transpose for encoder
        enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)

    enc_np = enc_out.squeeze(0).cpu().numpy()
    print(f"Encoder output shape: {enc_np.shape}")
    print(f"Encoder output stats: min={enc_np.min():.4f}, max={enc_np.max():.4f}, mean={enc_np.mean():.4f}")
    print(f"Encoder output first 5 values of first frame: {enc_np[0, :5]}")

    # Save encoder output for comparison
    enc_np.astype(np.float32).tofile("nemo_encoder_out.bin")
    print(f"Saved encoder output to nemo_encoder_out.bin")

    # Run full transcription
    print("\nRunning full transcription...")
    # Use greedy decoding
    with torch.no_grad():
        # Get decoder initial state
        decoder = model.decoder
        joint = model.joint

        # Initialize with blank token
        blank_id = model.decoder.blank_idx
        print(f"Blank token ID: {blank_id}")

        # Simple greedy decode
        tokens = []
        dec_state = None
        last_token = blank_id

        for t in range(enc_out.shape[1]):
            enc_frame = enc_out[:, t:t+1, :]  # [1, 1, dim]

            for _ in range(10):  # max symbols per step
                # Get decoder output
                dec_input = torch.tensor([[last_token]], device=device)
                dec_out, dec_state = decoder.predict(
                    dec_input, state=dec_state, add_sos=False, batch_size=1
                )

                # Joint network
                joint_out = joint.joint(enc_frame, dec_out)  # [1, 1, 1, vocab]
                logits = joint_out.squeeze()

                best_token = logits.argmax().item()

                if best_token == blank_id:
                    break

                tokens.append(best_token)
                last_token = best_token

        print(f"Decoded {len(tokens)} tokens")
        print(f"First 20 tokens: {tokens[:20]}")

    # Decode to text using model's tokenizer
    if hasattr(model, 'tokenizer'):
        text = model.tokenizer.ids_to_text(tokens)
        print(f"Decoded text: {text}")


if __name__ == "__main__":
    main()
