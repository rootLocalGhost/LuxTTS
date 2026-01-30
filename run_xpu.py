import sys
import torch
import warnings
import soundfile as sf

# Patch PyTorch to work with Intel Arc GPUs (XPU)
# This fixes compatibility issues with external libraries that expect CUDA
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    # Create CUDA namespace if it doesn't exist so we can override autocast
    if not hasattr(torch, 'cuda'):
        import types
        torch.cuda = types.ModuleType('torch.cuda')
        torch.cuda.amp = types.ModuleType('torch.cuda.amp')
    elif not hasattr(torch.cuda, 'amp'):
        import types
        torch.cuda.amp = types.ModuleType('torch.cuda.amp')

    # Redirect CUDA autocast calls to XPU
    if hasattr(torch.amp, 'autocast'):
        # For newer PyTorch versions (2.4+)
        class XPUAutocast(torch.amp.autocast):
            def __init__(self, *args, **kwargs):
                # Always use XPU regardless of what the library requests
                if 'device_type' in kwargs:
                    kwargs['device_type'] = 'xpu'
                elif len(args) > 0 and args[0] == 'cuda':
                    args = ('xpu',) + args[1:]
                else:
                    kwargs['device_type'] = 'xpu'
                super().__init__(*args, **kwargs)
        torch.cuda.amp.autocast = XPUAutocast
    elif hasattr(torch.xpu.amp, 'autocast'):
        # For older PyTorch versions with IPEX
        torch.cuda.amp.autocast = torch.xpu.amp.autocast

# Suppress irrelevant warnings from linacodec
warnings.filterwarnings("ignore", category=FutureWarning, module="linacodec")

# Import our custom LuxTTS implementation for XPU
from zipvoice.luxvoice_xpu import LuxTTS

def main():
    # Load the LuxTTS model on Intel Arc GPU (XPU)
    # Using 2 threads for efficient model loading
    print("------------------------------------------------")
    print("Initializing LuxTTS on Intel Arc (XPU)...")
    lux_tts = LuxTTS('YatharthS/LuxTTS', device='xpu', threads=2)

    # Text to be spoken by the voice clone
    # Feel free to change this to whatever you want the model to say
    text = "The system is functioning perfectly. I am speaking to you from an Intel Arc A770 GPU. Isn't that amazing? I cant believe how far technology has come!"

    # Reference audio file that contains the voice to be cloned
    # Make sure this .wav file exists in the current directory
    prompt_audio = 'reference_voice.wav'

    # Convert the reference audio to features using Whisper
    # This captures both the content and style of the voice
    print("\n[1/2] Encoding prompt (Whisper + Feature Extractor)...")
    try:
        encoded_prompt = lux_tts.encode_prompt(prompt_audio, duration=5, rms=0.01)
    except FileNotFoundError:
        print(f"Error: Could not find '{prompt_audio}'. Please place a .wav file in this folder.")
        return

    # Generate speech using the encoded reference and target text
    # This combines ZipVoice for feature generation and Vocos for audio synthesis
    print("\n[2/2] Generating speech (ZipVoice + Vocos)...")
    final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4, speed=1.0)

    # Save the generated audio to a file
    output_filename = 'output_xpu.wav'
    final_wav = final_wav.numpy().squeeze()
    sf.write(output_filename, final_wav, 48000)
    print(f"\nSuccess! Audio saved to: {output_filename}")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()