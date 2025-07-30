import argparse
import torch
import onnxruntime
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

# ====== Load tokenizer dari HuggingFace ======
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")

# ====== Load model ONNX ======
onnx_model_path = "mms_tts_ind_vits.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# ====== Fungsi: teks ke audio ======
def tts_onnx(text, out_wav="output.wav"):
    encoded = tokenizer(text, return_tensors="np")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    waveform = ort_session.run(None, ort_inputs)[0][0]

    sf.write(out_wav, waveform, samplerate=16000)
    print(f"✅ Suara disimpan ke: {out_wav}")

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech CLI using ONNX")
    parser.add_argument("-text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("-o", type=str, default="output.wav", help="Output WAV file (default: output.wav)")
    args = parser.parse_args()

    tts_onnx(args.text, args.o)

if __name__ == "__main__":
    main()