# MMS TTS IND VITS ONNX

Proyek ini melakukan konversi teks Bahasa Indonesia menjadi suara menggunakan model Facebook MMS-TTS dan ONNX.

## Struktur Folder
- `export_to_onnx.py`: Ekspor model MMS-TTS ke format ONNX.
- `run_tts_onnx.py`: Jalankan inferensi TTS menggunakan model ONNX.
- `cli.py`: Skrip CLI untuk TTS.
- `test_bahasa.wav`, `output.wav`: Contoh file audio.
- `mms_tts_ind_vits.onnx`: Model hasil ekspor ONNX.
- `mms_tokenizer/`: Folder tokenizer dan model.

## Instalasi
1. Install Python 3.8+.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Ekspor Model ke ONNX
Jalankan:
```bash
python export_to_onnx.py
```
Model ONNX akan tersimpan sebagai `mms_tts_ind_vits.onnx`.

## Cara Inferensi TTS
Jalankan:
```bash
python run_tts_onnx.py
```

## Catatan
- Model dan tokenizer diambil dari HuggingFace: [facebook/mms-tts-ind](https://huggingface.co/facebook/mms-tts-ind).
- Pastikan file dan folder model sudah tersedia sesuai struktur di atas.

## Lisensi
Proyek ini menggunakan model dari Facebook MMS-TTS. Ikuti ketentuan lisensi dari sumber terkait.
