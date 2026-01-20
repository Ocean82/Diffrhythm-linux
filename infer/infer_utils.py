# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import librosa
import torchaudio
import random
import json
from muq import MuQMuLan
from mutagen.mp3 import MP3
import os
import numpy as np
from huggingface_hub import hf_hub_download

from sys import path
path.append(os.getcwd())

from model import DiT, CFM

# Determine the default cache directory
# Priority: HUGGINGFACE_HUB_CACHE -> HF_HOME -> /mnt/d/_hugging-face (WSL) -> D:\_hugging-face (Windows) -> ./pretrained
DEFAULT_CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
if not DEFAULT_CACHE_DIR:
    if os.name == 'nt': # Windows
        DEFAULT_CACHE_DIR = "D:\\_hugging-face"
    else: # Linux/WSL
        if os.path.exists("/mnt/d"):
            DEFAULT_CACHE_DIR = "/mnt/d/_hugging-face"
        else:
            DEFAULT_CACHE_DIR = "./pretrained"

print(f"DEBUG: Using cache directory: {DEFAULT_CACHE_DIR}", flush=True)

# Audio format validation constants
SUPPORTED_INPUT_FORMATS = {'.wav', '.flac', '.mp3', '.ogg', '.aac', '.m4a'}
EXPECTED_SAMPLE_RATE = 44100
MIN_AUDIO_DURATION = 10  # seconds, for style reference
EXPECTED_CHANNELS = 2
EXPECTED_BIT_DEPTH = 16


def validate_audio_file(file_path):
    """
    Validate audio file format and properties before processing
    
    Args:
        file_path: Path to audio file
        
    Returns:
        dict: Audio properties (sample_rate, duration, channels, bit_depth)
        
    Raises:
        ValueError: If file format not supported or cannot be read
    """
    print(f"   Validating audio file: {file_path}", flush=True)
    
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Check format
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {ext}\n"
            f"Supported formats: {', '.join(SUPPORTED_INPUT_FORMATS)}"
        )
    
    print(f"     ✓ Format supported: {ext}", flush=True)
    
    # Get audio info
    try:
        if ext == ".mp3":
            # Use mutagen for MP3 metadata
            try:
                from mutagen.mp3 import MP3
                audio_file = MP3(file_path)
                duration = audio_file.info.length
                channels = audio_file.info.channels if hasattr(audio_file.info, 'channels') else 2
                sample_rate = audio_file.info.sample_rate if hasattr(audio_file.info, 'sample_rate') else EXPECTED_SAMPLE_RATE
                bit_depth = 16
                
                print(f"     Duration: {duration:.1f}s, Sample Rate: {sample_rate} Hz, Channels: {channels}", flush=True)
                
                if duration < MIN_AUDIO_DURATION:
                    raise ValueError(
                        f"Audio file too short: {duration:.1f}s\n"
                        f"Minimum required: {MIN_AUDIO_DURATION}s"
                    )
                
                return {
                    'path': file_path,
                    'format': ext,
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'bit_depth': bit_depth
                }
            except Exception as e:
                print(f"     ⚠ mutagen MP3 read failed, trying librosa: {e}", flush=True)
                raise
        
        else:
            # Use librosa for WAV, FLAC, etc.
            duration = librosa.get_duration(path=file_path)
            
            # Try to get more detailed info
            try:
                y, sr = librosa.load(file_path, sr=None, mono=False)
                channels = 1 if y.ndim == 1 else y.shape[0]
            except Exception as e:
                print(f"     ⚠ Could not determine channels: {e}", flush=True)
                channels = EXPECTED_CHANNELS
                sr = EXPECTED_SAMPLE_RATE
            
            print(f"     Duration: {duration:.1f}s, Sample Rate: {sr} Hz, Channels: {channels}", flush=True)
            
            if duration < MIN_AUDIO_DURATION:
                raise ValueError(
                    f"Audio file too short: {duration:.1f}s\n"
                    f"Minimum required: {MIN_AUDIO_DURATION}s"
                )
            
            return {
                'path': file_path,
                'format': ext,
                'duration': duration,
                'sample_rate': sr,
                'channels': channels,
                'bit_depth': 16
            }
    
    except Exception as e:
        raise ValueError(f"Error reading audio file: {str(e)}")


def validate_audio_tensor_properties(audio_tensor, expected_sr=EXPECTED_SAMPLE_RATE):
    """
    Validate loaded audio tensor properties
    
    Args:
        audio_tensor: Audio tensor from librosa/torchaudio
        expected_sr: Expected sample rate
        
    Returns:
        dict: Tensor properties
        
    Raises:
        ValueError: If tensor properties unexpected
    """
    print(f"   Validating audio tensor properties", flush=True)
    
    # Check dtype
    if audio_tensor.dtype not in [np.float32, np.float64]:
        print(f"     ⚠ Unexpected dtype: {audio_tensor.dtype}", flush=True)
    
    # Check shape
    if audio_tensor.ndim not in [1, 2]:
        raise ValueError(f"Unexpected tensor dimensions: {audio_tensor.ndim}")
    
    # Check for NaN/Inf
    if np.isnan(audio_tensor).any():
        raise ValueError("Audio contains NaN values")
    
    if np.isinf(audio_tensor).any():
        raise ValueError("Audio contains Inf values")
    
    # Check value range
    max_val = np.max(np.abs(audio_tensor))
    if max_val == 0:
        raise ValueError("Audio is completely silent (all zeros)")
    
    if max_val > 1.0:
        print(f"     ⚠ Audio values exceed [-1, 1] range: max={max_val:.2f}", flush=True)
    
    properties = {
        'shape': audio_tensor.shape,
        'dtype': audio_tensor.dtype,
        'max_value': float(max_val),
        'num_samples': audio_tensor.shape[-1],
        'is_stereo': audio_tensor.ndim == 2,
        'channels': audio_tensor.shape[0] if audio_tensor.ndim == 2 else 1
    }
    
    print(f"     ✓ Tensor valid: shape={properties['shape']}, max={properties['max_value']:.3f}", flush=True)
    
    return properties

def vae_sample(mean, scale):
    stdev = torch.nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl

def normalize_audio(y, target_dbfs=0):
    max_amplitude = torch.max(torch.abs(y))

    target_amplitude = 10.0**(target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude

    normalized_audio = y * scale_factor

    return normalized_audio

def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio

class PadCrop(torch.nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = torchaudio.functional.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)
    if target_length is None:
        target_length = audio.shape[-1]
    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio

def decode_audio(latents, vae_model, chunked=False, overlap=32, chunk_size=128):
    downsampling_ratio = 2048
    io_channels = 2
    if not chunked:
        return vae_model.decode_export(latents)
    else:
        # chunked decoding
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = latents[:, :, i : i + chunk_size]
            chunks.append(chunk)
        if i + chunk_size != total_size:
            # Final chunk
            chunk = latents[:, :, -chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # samples_per_latent is just the downsampling ratio
        samples_per_latent = downsampling_ratio
        # Create an empty waveform, we will populate it with chunks as decode them
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size, io_channels, y_size)).to(latents.device)
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            # decode the chunk
            y_chunk = vae_model.decode_export(x_chunk)
            # figure out where to put the audio along the time domain
            if i == num_chunks - 1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            #  remove the edges of the overlaps
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final

def encode_audio(audio, vae_model, chunked=False, overlap=32, chunk_size=128):
    downsampling_ratio = 2048
    latent_dim = 128
    if not chunked:
        # default behavior. Encode the entire audio in parallel
        return vae_model.encode_export(audio)
    else:
        # CHUNKED ENCODING
        # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
        samples_per_latent = downsampling_ratio
        total_size = audio.shape[2] # in samples
        batch_size = audio.shape[0]
        chunk_size *= samples_per_latent # converting metric in latents to samples
        overlap *= samples_per_latent # converting metric in latents to samples
        hop_size = chunk_size - overlap
        chunks = []
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = audio[:,:,i:i+chunk_size]
            chunks.append(chunk)
        if i+chunk_size != total_size:
            # Final chunk
            chunk = audio[:,:,-chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # Note: y_size might be a different value from the latent length used in diffusion training
        # because we can encode audio of varying lengths
        # However, the audio should've been padded to a multiple of samples_per_latent by now.
        y_size = total_size // samples_per_latent
        # Create an empty latent, we will populate it with chunks as we encode them
        y_final = torch.zeros((batch_size,latent_dim,y_size)).to(audio.device)
        for i in range(num_chunks):
            x_chunk = chunks[i,:]
            # encode the chunk
            y_chunk = vae_model.encode_export(x_chunk)
            # figure out where to put the audio along the time domain
            if i == num_chunks-1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size // samples_per_latent
                t_end = t_start + chunk_size // samples_per_latent
            #  remove the edges of the overlaps
            ol = overlap//samples_per_latent//2
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks-1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
        return y_final

def prepare_model(max_frames, device, repo_id="ASLP-lab/DiffRhythm-1_2"):
    # prepare cfm model
    print(f"DEBUG: Preparing CFM model with max_frames={max_frames}...", flush=True)
    if max_frames == 2048:
        repo_id = "ASLP-lab/DiffRhythm-1_2"
    else:
        repo_id = "ASLP-lab/DiffRhythm-1_2-full"
        
    dit_ckpt_path = hf_hub_download(
        repo_id=repo_id, filename="cfm_model.pt", cache_dir=DEFAULT_CACHE_DIR
    )
        
    dit_config_path = "./config/diffrhythm-1b.json"
    with open(dit_config_path) as f:
        model_config = json.load(f)
    print(f"DEBUG: Initializing DiT model...", flush=True)
    dit_model_cls = DiT
    cfm = CFM(
        transformer=dit_model_cls(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    cfm = cfm.to(device)
    print(f"DEBUG: Loading CFM checkpoint...", flush=True)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)

    # prepare tokenizer
    print(f"DEBUG: Preparing CNENTokenizer...", flush=True)
    tokenizer = CNENTokenizer()

    # prepare muq
    print(f"DEBUG: Preparing MuQMuLan (from_pretrained)...", flush=True)
    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=DEFAULT_CACHE_DIR)
    muq = muq.to(device).eval()

    # prepare vae
    print(f"DEBUG: Preparing VAE...", flush=True)
    vae_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir=DEFAULT_CACHE_DIR,
    )
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)
    print(f"DEBUG: All models prepared successfully.", flush=True)

    return cfm, tokenizer, muq, vae


# for song edit, will be added in the future
def get_reference_latent(device, max_frames, edit, pred_segments, ref_song, vae_model):
    sampling_rate = 44100
    downsample_rate = 2048
    io_channels = 2
    if edit:
        input_audio, in_sr = torchaudio.load(ref_song)
        input_audio = prepare_audio(input_audio, in_sr=in_sr, target_sr=sampling_rate, target_length=None, target_channels=io_channels, device=device)
        input_audio = normalize_audio(input_audio, -6)
        
        with torch.no_grad():
            latent = encode_audio(input_audio, vae_model, chunked=True) # [b d t]
            mean, scale = latent.chunk(2, dim=1)
            prompt, _ = vae_sample(mean, scale)
            prompt = prompt.transpose(1, 2) # [b t d]
        
        pred_segments = json.loads(pred_segments)
        
        pred_frames = []
        for st, et in pred_segments:
            sf = 0 if st == -1 else int(st * sampling_rate / downsample_rate)
            ef = max_frames if et == -1 else int(et * sampling_rate / downsample_rate)
            pred_frames.append((sf, ef))

        return prompt, pred_frames
    else:
        prompt = torch.zeros(1, max_frames, 64).to(device)
        pred_frames = [(0, max_frames)]
        return prompt, pred_frames


def get_negative_style_prompt(device):
    file_path = "infer/example/vocal.npy"
    vocal_stlye = np.load(file_path)

    vocal_stlye = torch.from_numpy(vocal_stlye).to(device)  # [1, 512]

    # Use appropriate dtype based on device
    is_cpu = device == "cpu" or (hasattr(device, 'type') and device.type == 'cpu')
    if is_cpu:
        vocal_stlye = vocal_stlye.float()
    else:
        vocal_stlye = vocal_stlye.half()

    return vocal_stlye


@torch.no_grad()
def get_style_prompt(model, wav_path=None, prompt=None):
    """
    Get style embedding from audio or text prompt
    
    Args:
        model: MuQ-MuLan model
        wav_path: Path to audio file for style reference
        prompt: Text prompt for style
        
    Returns:
        torch.Tensor: Style embedding [1, 512]
        
    Raises:
        ValueError: If audio format not supported or too short
    """
    mulan = model

    if prompt is not None:
        print(f"   Using text prompt for style: '{prompt}'", flush=True)
        result = mulan(texts=prompt)
        # Use appropriate dtype based on device
        is_cpu = str(model.device) == "cpu" or (hasattr(model.device, 'type') and model.device.type == 'cpu')
        if is_cpu:
            return result.float()
        else:
            return result.half()

    # Audio-based style reference
    print(f"   Using audio file for style: {wav_path}", flush=True)
    
    # Validate audio file
    try:
        audio_info = validate_audio_file(wav_path)
    except Exception as e:
        print(f"     ✗ ERROR: {e}", flush=True)
        raise
    
    # Check duration
    if audio_info['duration'] < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio file too short: {audio_info['duration']:.1f}s\n"
            f"Minimum required: {MIN_AUDIO_DURATION}s"
        )
    
    print(f"     ✓ Audio file valid: {audio_info['duration']:.1f}s", flush=True)

    # Load audio for style extraction
    try:
        mid_time = audio_info['duration'] // 2
        start_time = mid_time - 5
        
        print(f"     Loading audio segment: {start_time:.1f}s to {start_time + 10:.1f}s", flush=True)
        wav, sr = librosa.load(wav_path, sr=24000, offset=start_time, duration=10)
        
        # Validate loaded audio
        try:
            audio_props = validate_audio_tensor_properties(wav, sr)
        except Exception as e:
            print(f"     ✗ ERROR validating loaded audio: {e}", flush=True)
            raise
        
        print(f"     ✓ Audio loaded: {audio_props['shape']}, sample rate: {sr} Hz", flush=True)

    except Exception as e:
        print(f"     ✗ ERROR loading audio: {e}", flush=True)
        print(f"     Check that:")
        print(f"       - File exists: {os.path.exists(wav_path)}")
        print(f"       - Format supported: {os.path.splitext(wav_path)[1]}")
        print(f"       - FFmpeg installed (for MP3/OGG): check_codec_pipeline.py")
        raise

    # Convert to tensor and get embedding
    try:
        wav = torch.tensor(wav).unsqueeze(0).to(model.device)

        print(f"     Extracting style embedding...", flush=True)
        with torch.no_grad():
            audio_emb = mulan(wavs=wav)  # [1, 512]

        # Use appropriate dtype based on device
        is_cpu = str(model.device) == "cpu" or (hasattr(model.device, 'type') and model.device.type == 'cpu')
        if is_cpu:
            audio_emb = audio_emb.float()
        else:
            audio_emb = audio_emb.half()

        print(f"     ✓ Style embedding extracted: shape={audio_emb.shape}", flush=True)

        return audio_emb

    except Exception as e:
        print(f"     ✗ ERROR extracting style embedding: {e}", flush=True)
        raise


def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split("\n"):
        try:
            time, lyric = line[1:9], line[10:]
            lyric = lyric.strip()
            mins, secs = time.split(":")
            secs = int(mins) * 60 + float(secs)
            lyrics_with_time.append((secs, lyric))
        except:
            continue
    return lyrics_with_time


class CNENTokenizer:
    def __init__(self):
        with open("./g2p/g2p/vocab.json", "r", encoding='utf-8') as file:
            self.phone2id: dict = json.load(file)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        from g2p.g2p_generation import chn_eng_g2p

        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])


def get_lrc_token(max_frames, text, tokenizer, max_secs, device):

    lyrics_shift = 0
    sampling_rate = 44100
    downsample_rate = 2048
    # max_secs = max_frames / (sampling_rate / downsample_rate)

    comma_token_id = 1
    period_token_id = 2

    lrc_with_time = parse_lyrics(text)

    modified_lrc_with_time = []
    for i in range(len(lrc_with_time)):
        time, line = lrc_with_time[i]
        line_token = tokenizer.encode(line)
        modified_lrc_with_time.append((time, line_token))
    lrc_with_time = modified_lrc_with_time

    lrc_with_time = [
        (time_start, line)
        for (time_start, line) in lrc_with_time
        if time_start < max_secs
    ]
    if max_frames == 2048:
        lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time
        
    end_frame = max_frames if max_frames == 2048 else int(max_secs * (sampling_rate / downsample_rate))
    end_frame = min(end_frame, max_frames) 

    normalized_start_time = 0.0
    
    normalized_duration = end_frame / max_frames

    lrc = torch.zeros((end_frame,), dtype=torch.long)

    tokens_count = 0
    last_end_pos = 0
    for time_start, line in lrc_with_time:
        tokens = [
            token if token != period_token_id else comma_token_id for token in line
        ] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        num_tokens = tokens.shape[0]

        gt_frame_start = int(time_start * sampling_rate / downsample_rate)

        frame_shift = random.randint(int(-lyrics_shift), int(lyrics_shift))

        frame_start = max(gt_frame_start - frame_shift, last_end_pos)
        frame_len = min(num_tokens, end_frame - frame_start)

        lrc[frame_start : frame_start + frame_len] = tokens[:frame_len]

        tokens_count += num_tokens
        last_end_pos = frame_start + frame_len

    lrc_emb = lrc.unsqueeze(0).to(device)

    normalized_start_time = torch.tensor(normalized_start_time).unsqueeze(0).to(device)
    normalized_duration = torch.tensor(normalized_duration).unsqueeze(0).to(device)

    # Use appropriate dtype based on device
    is_cpu = device == "cpu" or (hasattr(device, 'type') and device.type == 'cpu')
    if is_cpu:
        normalized_start_time = normalized_start_time.float()
        normalized_duration = normalized_duration.float()
    else:
        normalized_start_time = normalized_start_time.half()
        normalized_duration = normalized_duration.half()

    return lrc_emb, normalized_start_time, end_frame, normalized_duration


def load_checkpoint(model, ckpt_path, device, use_ema=True):
    # Determine optimal dtype based on device
    # CPU: use float32 for better performance (FP16 emulation is slow on CPU)
    # GPU: use float16 for faster inference and lower memory
    is_cpu = device == "cpu" or (hasattr(device, 'type') and device.type == 'cpu')

    if is_cpu:
        print(f"DEBUG: CPU detected - using float32 for better performance", flush=True)
        # Keep model in float32 for CPU
        model = model.float()
    else:
        print(f"DEBUG: GPU detected - using float16 for efficiency", flush=True)
        model = model.half()

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model.to(device)
