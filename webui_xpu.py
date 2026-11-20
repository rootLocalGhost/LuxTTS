import os
import sys
import warnings
import logging
import types
import gradio as gr
import torch

warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings(
    "ignore", message=".*Transcription using a multilingual Whisper.*"
)
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.ERROR)

if hasattr(torch, "xpu") and torch.xpu.is_available():
    if not hasattr(torch, "cuda"):
        torch.cuda = types.ModuleType("torch.cuda")
    if not hasattr(torch.cuda, "amp"):
        torch.cuda.amp = types.ModuleType("torch.cuda.amp")

    _BaseAutocast = None
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        _BaseAutocast = torch.amp.autocast
    elif hasattr(torch.xpu, "amp") and hasattr(torch.xpu.amp, "autocast"):
        _BaseAutocast = torch.xpu.amp.autocast

    if _BaseAutocast:

        class XPUAutocast(_BaseAutocast):
            def __init__(self, *args, **kwargs):
                if "device_type" in kwargs:
                    kwargs["device_type"] = "xpu"
                elif len(args) > 0 and args[0] == "cuda":
                    args = ("xpu",) + args[1:]
                else:
                    kwargs["device_type"] = "xpu"
                super().__init__(*args, **kwargs)

        setattr(torch.cuda.amp, "autocast", XPUAutocast)

from zipvoice.luxvoice_xpu import LuxTTS

lux_tts = None
_cached_ref_path = None
_cached_prompt = None


def console_log(message):
    print(f" [LuxTTS] {message}")


def initialize_system():
    """Load the LuxTTS model on XPU when the app starts"""
    global lux_tts
    if lux_tts is None:
        console_log("-----------------------------------------")
        console_log("Initializing System on Intel Arc (XPU)...")
        lux_tts = LuxTTS("YatharthS/LuxTTS", device="xpu", threads=4)
        console_log("System Ready.")
        console_log("-----------------------------------------")


def generate_voice(text, reference_audio, speed, steps, guidance):
    """Generate speech from text using the reference voice"""
    global _cached_ref_path, _cached_prompt
    if lux_tts is None:
        return None, "Error: Model not loaded."
    if not reference_audio:
        raise gr.Error("Reference audio required.")

    console_log(f"Processing: {text[:40]}...")
    try:
        if reference_audio != _cached_ref_path or _cached_prompt is None:
            console_log("Encoding new reference audio...")
            _cached_prompt = lux_tts.encode_prompt(
                reference_audio, duration=5, rms=0.01
            )
            _cached_ref_path = reference_audio

        final_wav = lux_tts.generate_speech(
            text,
            _cached_prompt,
            num_steps=int(steps),
            guidance_scale=float(guidance),
            speed=float(speed),
        )
    except Exception as e:
        console_log(f"Error: {str(e)}")
        return None, f"Error: {str(e)}"

    console_log("   -> Complete.")
    audio_data = final_wav.numpy().squeeze()
    return (48000, audio_data), "Success"


try:
    theme = gr.themes.Soft(
        primary_hue="cyan", secondary_hue="slate", neutral_hue="slate", radius_size="md"
    )
except Exception:
    theme = None

with gr.Blocks(title="LuxTTS XPU", theme=theme) as app:
    gr.Markdown("# ⚡ LuxTTS (Intel Arc)")
    with gr.Row():
        with gr.Column(scale=4):
            ref_audio_input = gr.Audio(
                label="Reference Voice",
                type="filepath",
                sources=["upload", "microphone"],
            )
            text_input = gr.Textbox(
                label="Text",
                placeholder="Text to generate...",
                lines=3,
                value="The quick brown fox jumps over the lazy dog.",
            )
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    speed_slider = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")
                    steps_slider = gr.Slider(2, 10, 4, step=1, label="Steps")
                    guidance_slider = gr.Slider(
                        1.0, 5.0, 3.0, step=0.5, label="Similarity"
                    )
            gen_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            audio_output = gr.Audio(label="Result", autoplay=True)
            log_output = gr.Label(label="Status", value="Ready")

    gen_btn.click(
        fn=generate_voice,
        inputs=[
            text_input,
            ref_audio_input,
            speed_slider,
            steps_slider,
            guidance_slider,
        ],
        outputs=[audio_output, log_output],
    )

if __name__ == "__main__":
    initialize_system()
    app.launch(server_name="0.0.0.0", share=False)
