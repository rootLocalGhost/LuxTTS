import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional, List
import numpy as np
import safetensors.torch
import torch
import librosa
import torchaudio
from transformers import pipeline
from huggingface_hub import snapshot_download
from lhotse.utils import fix_random_seed

# Import core LuxTTS components
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm
from dataclasses import dataclass, field
from linacodec.vocoder.vocos import Vocos
from zipvoice.onnx_modeling import OnnxModel
from torch.nn.utils import parametrize

# Import XPU-specific utilities
from zipvoice.utils.common_xpu import AttributeDict, str2bool

@dataclass
class LuxTTSConfig:
    """Configuration class for LuxTTS model parameters"""
    model_dir: Optional[str] = None
    checkpoint_name: str = "model.pt"
    vocoder_path: Optional[str] = None
    trt_engine_path: Optional[str] = None
    tokenizer: str = "emilia"
    lang: str = "en-us"

@torch.inference_mode
def process_audio(audio, transcriber, tokenizer, feature_extractor, device, target_rms=0.1, duration=4, feat_scale=0.1):
    """Process reference audio to extract features for voice cloning"""
    # Load audio at two different sample rates for different purposes
    prompt_wav, sr = librosa.load(audio, sr=24000, duration=duration)
    prompt_wav2, sr = librosa.load(audio, sr=16000, duration=duration)

    # Transcribe the audio to get text content using Whisper
    prompt_text = transcriber(prompt_wav2)["text"]
    print(prompt_text)

    # Convert to tensor and normalize loudness
    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    # Extract acoustic features and move to target device
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=24000
    ).to(device)

    # Prepare features for the model
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    return prompt_tokens, prompt_features_lens, prompt_features, prompt_rms

def generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1):
    """Generate speech features and convert to audio waveform"""
    tokens = tokenizer.texts_to_token_ids([text])
    device = next(model.parameters()).device
    speed = speed * 1.3

    with torch.inference_mode():
        # Generate acoustic features using the ZipVoice model
        (pred_features, _, _, _) = model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration='predict',
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
    # Prepare features for vocoder
    pred_features = pred_features.permute(0, 2, 1) / 0.1
    # Convert features to audio waveform
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
    # Adjust volume to match reference audio
    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)
    return wav

def load_models_xpu(model_path=None):
    """Load all LuxTTS models on XPU (Intel Arc GPU) for accelerated inference"""
    params = LuxTTSConfig()
    if model_path is None:
        model_path = snapshot_download("YatharthS/LuxTTS")

    # Define paths to model files
    token_file = f"{model_path}/tokens.txt"
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"

    # Determine device to use (XPU if available, otherwise CPU)
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        print("Warning: Torch XPU not available. Falling back to CPU.")
        device_str = 'cpu'
    else:
        device_str = 'xpu'

    # Load Whisper for audio transcription
    print(f"Initializing Whisper on {device_str}...")
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device_str)

    # Load tokenizer
    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    # Load model configuration
    with open(model_config, "r") as f:
        model_config = json.load(f)

    # Initialize and load the ZipVoice model
    model = ZipVoiceDistill(
        **model_config["model"],
        **tokenizer_config,
    )

    load_checkpoint(filename=model_ckpt, model=model, strict=True)

    # Set the device for parameters
    params.device = torch.device(device_str, 0) if device_str == 'xpu' else torch.device('cpu')

    # Move model to target device
    model = model.to(params.device).eval()

    # Initialize feature extractor
    feature_extractor = VocosFbank()

    # Load and configure the vocoder
    print(f"Loading Vocos on {device_str}...")
    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml')
    vocos = vocos.to(params.device)

    # Remove parametrizations and load weights
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=params.device))

    params.sampling_rate = model_config["feature"]["sampling_rate"]
    return model, feature_extractor, vocos, tokenizer, transcriber

def load_models_cpu(model_path = None, num_thread=2):
    """Load all LuxTTS models on CPU for compatibility"""
    params = LuxTTSConfig()
    params.seed = 42
    model_path = snapshot_download('YatharthS/LuxTTS')
    token_file = f"{model_path}/tokens.txt"
    text_encoder_path = f"{model_path}/text_encoder.onnx"
    fm_decoder_path = f"{model_path}/fm_decoder.onnx"
    model_config  = f"{model_path}/config.json"

    # Load Whisper for audio transcription on CPU
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device='cpu')
    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    # Load model configuration
    with open(model_config, "r") as f:
        model_config = json.load(f)

    # Initialize ONNX model for CPU inference
    model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=num_thread)

    # Load and configure the vocoder
    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').eval()
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=torch.device('cpu')))

    # Initialize feature extractor
    feature_extractor = VocosFbank()
    params.sampling_rate = model_config["feature"]["sampling_rate"]
    params.onnx_int8 = True
    return model, feature_extractor, vocos, tokenizer, transcriber