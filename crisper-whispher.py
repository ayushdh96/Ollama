import os
import torch
import gradio as gr
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# Read your Hugging Face token from token.txt
with open("./data/token.txt", "r") as file:
    hf_token = file.read().strip()

# Device selection: Prefer MPS (Apple Silicon), then CUDA, otherwise CPU.
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32  # MPS works best with float32.
elif torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

print(f"Using device: {device} with torch_dtype: {torch_dtype}")

# For the pipeline, if using CPU or MPS, pass device=-1; if using CUDA, pass device index 0.
if device in ["cpu", "mps"]:
    pipeline_device = -1
else:
    pipeline_device = 0

model_name = "nyrahealth/CrisperWhisper"

# Load the processor and model with authentication (using token= as recommended).
processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    token=hf_token
)

# Move the model to the selected device.
model.to(device)

# Create the ASR pipeline.
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps='word',
    torch_dtype=torch_dtype,
    device=pipeline_device,
    token=hf_token
)

def transcribe(audio_file):
    """
    Transcribes an audio file using the CrisperWhisper ASR pipeline.
    """
    if audio_file is None:
        return "No audio provided."
    
    result = asr_pipe(audio_file)
    return result["text"]

# Create a Gradio interface.
# Note: We removed the "source" parameter.
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", interactive=True),
    outputs="text",
    title="Crisper Whisper ASR",
    description="Speak into your microphone (or record audio) and see your words transcribed by Crisper Whisper."
)

# Launch the Gradio app.
interface.launch()