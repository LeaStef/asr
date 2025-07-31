#!/usr/bin/env python3

import os
import sys
import torch
import librosa
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.config import create_config_from_dict
from models.asr_model import create_model
from data.preprocessing import MelSpectrogramTransform
from training.utils import load_checkpoint


def load_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file"""
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio


def preprocess_audio(audio, mel_transform):
    """Convert audio to mel spectrogram"""
    # Convert to tensor and add batch dimension
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    
    # Apply mel transform
    mel_spec = mel_transform(audio_tensor)
    
    # Add batch dimension: (1, n_mels, time)
    return mel_spec


def inference_single_file(model, audio_path, vocab, device, config):
    """Perform inference on a single audio file"""
    
    # Load audio
    audio = load_audio(audio_path)
    
    # Create mel transform
    mel_transform = MelSpectrogramTransform(
        sample_rate=config.data.sample_rate,
        n_mels=config.model.input_dim
    )
    
    # Preprocess
    mel_spec = preprocess_audio(audio, mel_transform).to(device)
    input_lengths = torch.LongTensor([mel_spec.size(2)])
    
    # Inference
    model.eval()
    with torch.no_grad():
        log_probs, _ = model(mel_spec, input_lengths)
        predictions = model.decode(log_probs, input_lengths)
    
    # Decode to text
    char_list = vocab['idx_to_char']
    predicted_text = ''.join([char_list[idx] for idx in predictions[0] if idx != 0])  # Remove blank tokens
    
    return predicted_text.strip()


def main():
    if len(sys.argv) != 3:
        print("Usage: python inference_single.py <checkpoint_path> <audio_file>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    audio_file = sys.argv[2]
    
    # Check files exist
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Load checkpoint to get config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create basic config (you might need to adjust this based on your checkpoint)
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        'model': {
            'input_dim': 80,
            'vocab_size': 29,  # Will be updated from checkpoint
            'encoder_hidden_dim': 512,
            'encoder_layers': 4,
            'encoder_memory_size': 256,
            'decoder_hidden_dim': 512
        },
        'data': {
            'sample_rate': 16000,
        }
    })
    
    config = create_config_from_dict(config)
    
    # Create model
    model = create_model(config.model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create vocab from checkpoint
    vocab = checkpoint.get('vocab', {
        'idx_to_char': {i: chr(ord('a') + i) for i in range(26)},  # Basic mapping
        'vocab_size': 29
    })
    
    # Perform inference
    print(f"Processing: {audio_file}")
    prediction = inference_single_file(model, audio_file, vocab, device, config)
    
    print(f"Transcription: {prediction}")


if __name__ == "__main__":
    main()