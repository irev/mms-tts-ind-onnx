import torch
import onnxruntime
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

# ====== Load tokenizer dari HuggingFace ======
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")
# Download dan simpan ke folder lokal
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind", cache_dir="./mms_tokenizer")

# ====== Load model ONNX ======
onnx_model_path = "mms_tts_ind_vits.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# ====== Fungsi: teks ke audio ======
def tts_onnx(text, out_wav="output.wav"):
    # Tokenisasi input
    encoded = tokenizer(text, return_tensors="np")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Inference ONNX
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    waveform = ort_session.run(None, ort_inputs)[0][0]  # [1, T] → ambil [T]

    # Simpan ke file
    sf.write(out_wav, waveform, samplerate=16000)
    print(f"✅ Suara disimpan ke: {out_wav}")

# ====== Contoh pakai ======
if __name__ == "__main__":
    teks = "selamat pagi dunia dari ONNX! Ini adalah contoh TTS. Mari kita uji apakah suara ini terdengar baik. Semoga berhasil! 😊"
    tts_onnx(teks)
