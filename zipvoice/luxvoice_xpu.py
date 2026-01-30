from zipvoice.modeling_utils_xpu import process_audio, generate, load_models_xpu, load_models_cpu
from zipvoice.onnx_modeling import generate_cpu
import torch

class LuxTTS:
    """Main class for LuxTTS text-to-speech generation with XPU support"""

    def __init__(self, model_path='YatharthS/LuxTTS', device='xpu', threads=4):
        """Initialize the LuxTTS model with specified device and thread count"""
        if model_path == 'YatharthS/LuxTTS':
            model_path = None

        if device == 'cpu':
            # Load models on CPU for compatibility
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu(model_path, threads)
            print("Loading model on CPU")
        else:
            # Load models on XPU (Intel Arc GPU) for acceleration
            if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
                print("Warning: XPU device requested but torch.xpu is not available. Check your PyTorch installation.")
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_xpu(model_path)
            print("Loading model on XPU")

        # Store model components as instance variables
        self.model = model
        self.feature_extractor = feature_extractor
        self.vocos = vocos
        self.tokenizer = tokenizer
        self.transcriber = transcriber
        self.device = device
        # Set frequency range for audio output
        self.vocos.freq_range = 12000

    def encode_prompt(self, prompt_audio, duration=5, rms=0.001):
        """Convert reference audio to encoded features for voice cloning"""
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(
            prompt_audio, self.transcriber, self.tokenizer, self.feature_extractor, self.device, target_rms=rms, duration=duration
        )
        # Return a dictionary with all the encoded information
        encode_dict = {
            "prompt_tokens": prompt_tokens,
            'prompt_features_lens': prompt_features_lens,
            'prompt_features': prompt_features,
            'prompt_rms': prompt_rms
        }
        return encode_dict

    def generate_speech(self, text, encode_dict, num_steps=4, guidance_scale=3.0, t_shift=0.5, speed=1.0, return_smooth=False):
        """Generate speech from text using the encoded reference voice"""
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = encode_dict.values()

        # Configure vocoder output quality
        if return_smooth == True:
            self.vocos.return_48k = False
        else:
            self.vocos.return_48k = True

        # Choose the appropriate generation function based on device
        if self.device == 'cpu':
            final_wav = generate_cpu(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, self.model, self.vocos, self.tokenizer, num_step=num_steps, guidance_scale=guidance_scale, t_shift=t_shift, speed=speed)
        else:
            final_wav = generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, self.model, self.vocos, self.tokenizer, num_step=num_steps, guidance_scale=guidance_scale, t_shift=t_shift, speed=speed)

        # Return the audio tensor on CPU
        return final_wav.cpu()