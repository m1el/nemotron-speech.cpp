#!/usr/bin/env python3
"""
Generate mel features from audio file for C++ inference using NeMo preprocessor.
Also runs NeMo model for reference transcription.

Usage:
    python scripts/gen_mel.py audio.wav output.mel.bin
"""

import argparse
import numpy as np
import torch

def save_mel_bin(mel, path):
    """Save mel features as binary file."""
    # Shape: [time, 128]
    mel = mel.astype(np.float32)
    with open(path, 'wb') as f:
        f.write(mel.tobytes())
    print(f"Saved mel features: {mel.shape} to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate mel features from audio using NeMo preprocessor")
    parser.add_argument("audio", help="Input audio file (WAV)")
    parser.add_argument("output", help="Output mel features (binary)")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Skip NeMo reference transcription")

    args = parser.parse_args()

    # Load NeMo model
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    import nemo.collections.asr as nemo_asr

    print("Loading NeMo model...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model.eval()

    # Use model's preprocessor to compute mel features
    print(f"Processing audio: {args.audio}")

    # Load audio using model's internal method
    import soundfile as sf
    audio, sr = sf.read(args.audio)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy import signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)
        sr = 16000

    audio = torch.tensor(audio, dtype=torch.float32)
    audio_len = torch.tensor([len(audio)])

    print(f"Audio shape: {audio.shape}, sample rate: {sr}")

    # Move to same device as model
    device = next(model.parameters()).device
    audio = audio.to(device)
    audio_len = audio_len.to(device)

    # Run preprocessor
    with torch.no_grad():
        mel, mel_len = model.preprocessor(
            input_signal=audio.unsqueeze(0),
            length=audio_len
        )

    # mel shape: [batch, features, time] -> [time, features]
    mel = mel.squeeze(0).transpose(0, 1).cpu().numpy()
    print(f"Mel features shape: {mel.shape}")

    save_mel_bin(mel, args.output)

    # Optional: get reference transcription
    if not args.no_transcribe:
        print("\nTranscribing with NeMo...")
        try:
            transcriptions = model.transcribe([args.audio])
            print(f"NeMo reference transcription:")
            print(f"  \"{transcriptions[0]}\"")
        except Exception as e:
            print(f"Transcription failed: {e}")


if __name__ == "__main__":
    main()
