import torch
from transformers import VitsModel, AutoTokenizer
import torch.onnx

# Load model & tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-ind")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")
model.eval()

# Wrapper: hanya ambil .waveform output
class VitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.waveform

# Bungkus model
onnx_model = VitsWrapper(model)

# Siapkan input dummy
text = "selamat pagi dunia"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Export ke ONNX
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "mms_tts_ind_vits.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["waveform"],
    dynamic_axes={
        "input_ids": {1: "sequence_length"},
        "attention_mask": {1: "sequence_length"},
        "waveform": {1: "audio_length"}
    },
    opset_version=13
)

print("✅ Sukses export ke mms_tts_ind_vits.onnx")
