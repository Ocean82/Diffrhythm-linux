#!/usr/bin/env python3
import os
import sys
import torch
import torchaudio

# Test minimal audio generation and saving
def test_audio_output():
    print("Testing audio output capability...")
    
    # Create test audio (sine wave)
    sample_rate = 44100
    duration = 2  # seconds
    frequency = 440  # A4 note
    
    t = torch.linspace(0, duration, sample_rate * duration)
    audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)  # Add channel dimension
    audio = audio.repeat(2, 1)  # Make stereo
    
    # Normalize properly
    max_val = torch.max(torch.abs(audio))
    if max_val > 1e-8:
        audio = audio / max_val * 0.8  # Normalize to 80% amplitude
    else:
        print("ERROR: Generated audio is silent!")
        return False
    
    # Convert to int16
    audio_int16 = (audio * 32767).clamp(-32767, 32767).to(torch.int16)
    
    # Save
    output_path = "./output/test_sine_wave.wav"
    os.makedirs("./output", exist_ok=True)
    
    try:
        torchaudio.save(output_path, audio_int16.float(), sample_rate=sample_rate)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Test audio saved: {output_path} ({file_size} bytes)")
            return True
        else:
            print("✗ Test audio file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Failed to save test audio: {e}")
        return False

if __name__ == "__main__":
    success = test_audio_output()
    sys.exit(0 if success else 1)
