import streamlit as st
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sys
import os

# Function to transcribe audio
def transcribe_audio(audio_path):
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz sampling rate
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"])

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
def main():
    st.title("Speech-to-Text with Whisper")
    st.write("Upload an MP3 file to transcribe it into text using the Whisper model.")

    uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

    if uploaded_file is not None:
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Transcribing...")
        transcription = transcribe_audio("temp_audio.mp3")

        st.write("**Transcription:**")
        st.write(transcription)

        os.remove("temp_audio.mp3")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        transcription = transcribe_audio(audio_path)
        print("Transcription:", transcription)
    else:
        main()