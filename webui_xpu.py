import os
import sys
import warnings
import logging
import types
import gradio as gr # type: ignore
import torch

# Reduce console noise from various libraries
# Filter out common warnings that clutter the output
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*Transcription using a multilingual Whisper.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
warnings.filterwarnings("ignore", category=FutureWarning)

# Mute verbose loggers from transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.ERROR)

# Patch PyTorch to work with Intel Arc GPUs (XPU)
# This redirects CUDA calls to XPU equivalents
if hasattr(torch, 'xpu') and torch.xpu.is_available(): # type: ignore
    if not hasattr(torch, 'cuda'):
        torch.cuda = types.ModuleType('torch.cuda') # type: ignore

    if not hasattr(torch.cuda, 'amp'): # type: ignore
        torch.cuda.amp = types.ModuleType('torch.cuda.amp') # type: ignore

    # Determine the correct autocast class based on PyTorch version
    _BaseAutocast = None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'): # type: ignore
        _BaseAutocast = torch.amp.autocast # type: ignore
    elif hasattr(torch.xpu, 'amp') and hasattr(torch.xpu.amp, 'autocast'): # type: ignore
        _BaseAutocast = torch.xpu.amp.autocast # type: ignore

    if _BaseAutocast:
        class XPUAutocast(_BaseAutocast): # type: ignore
            def __init__(self, *args, **kwargs):
                # Always use XPU regardless of what the library requests
                if 'device_type' in kwargs:
                    kwargs['device_type'] = 'xpu'
                elif len(args) > 0 and args[0] == 'cuda':
                    args = ('xpu',) + args[1:]
                else:
                    kwargs['device_type'] = 'xpu'
                super().__init__(*args, **kwargs)

        # Replace the CUDA autocast with our XPU version
        setattr(torch.cuda.amp, 'autocast', XPUAutocast) # type: ignore

from zipvoice.luxvoice_xpu import LuxTTS

# Global variable to hold the LuxTTS model instance
lux_tts = None

def console_log(message):
    """Print log messages with LuxTTS prefix"""
    print(f" [LuxTTS] {message}")

def initialize_system():
    """Load the LuxTTS model on XPU when the app starts"""
    global lux_tts
    if lux_tts is None:
        console_log("-----------------------------------------")
        console_log("Initializing System on Intel Arc (XPU)...")
        # Use 4 threads to speed up the initial model loading
        lux_tts = LuxTTS('YatharthS/LuxTTS', device='xpu', threads=4)
        console_log("System Ready.")
        console_log("-----------------------------------------")

def generate_voice(text, reference_audio, speed, steps, guidance):
    """Generate speech from text using the reference voice"""
    if lux_tts is None:
        return None, "Error: Model not loaded."

    if not reference_audio:
        raise gr.Error("Reference audio required.")

    console_log(f"Processing: {text[:40]}...")

    try:
        # Extract voice characteristics from the reference audio
        encoded_prompt = lux_tts.encode_prompt(reference_audio, duration=5, rms=0.01)

        # Generate the speech based on text and reference voice
        final_wav = lux_tts.generate_speech(
            text,
            encoded_prompt,
            num_steps=int(steps),
            guidance_scale=float(guidance),
            speed=float(speed)
        )
    except Exception as e:
        console_log(f"Error: {str(e)}")
        return None, f"Error: {str(e)}"

    console_log("   -> Complete.")
    audio_data = final_wav.numpy().squeeze()
    return (48000, audio_data), "Success"

# Set up the UI theme with Intel Arc colors (cyan primary)
try:
    theme = gr.themes.Soft(
        primary_hue="cyan",      # Matches Intel Arc GPU branding
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="md"
    )
except:
    theme = None

# Create the Gradio interface
with gr.Blocks(title="LuxTTS XPU") as app:
    gr.Markdown("# ⚡ LuxTTS (Intel Arc)")

    with gr.Row():
        with gr.Column(scale=4):
            # Input for reference voice - can upload file or record
            ref_audio_input = gr.Audio(label="Reference Voice", type="filepath", sources=["upload", "microphone"])
            # Text input for the speech to be generated
            text_input = gr.Textbox(label="Text", placeholder="Text to generate...", lines=3, value="The quick brown fox jumps over the lazy dog.")

            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    # Speed: How fast the speech is generated
                    speed_slider = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")
                    # Steps: Number of generation steps (higher = more detailed but slower)
                    steps_slider = gr.Slider(2, 10, 4, step=1, label="Steps")
                    # Guidance: How similar to make the output to the reference voice
                    guidance_slider = gr.Slider(1.0, 5.0, 3.0, step=0.5, label="Similarity")

            gen_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            # Output audio player
            audio_output = gr.Audio(label="Result", autoplay=True)
            # Status indicator
            log_output = gr.Label(label="Status", value="Ready")

    # Connect the generate button to the processing function
    gen_btn.click(
        fn=generate_voice,
        inputs=[text_input, ref_audio_input, speed_slider, steps_slider, guidance_slider],
        outputs=[audio_output, log_output]
    )

if __name__ == "__main__":
    initialize_system()
    # Launch the web interface
    app.launch(server_name="0.0.0.0", share=False, theme=theme)