import os, time, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "models/gemma-4-e4b-it"
print("Lade Modell ...", flush=True)
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"
)
print(f"Modell geladen in {time.time()-t0:.1f} s. Device={model.device}", flush=True)

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Keine Kamera gefunden"
time.sleep(1.0)
ret, frame = cap.read()
assert ret, "Kein Frame"
img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
print("Frame erhalten: %s" % (img.size,), flush=True)

prompt = processor.tokenizer.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe what you see. 2 to 3 short sentences."},
    ]}],
    tokenize=False, add_generation_prompt=True,
)
print("Prompt:", prompt, flush=True)

inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
p_len = inputs["input_ids"].shape[-1]
print(f"\nInferenz startet ... (prompt len = {p_len})", flush=True)
t1 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
dt = time.time() - t1
new = out[0, p_len:]
print(f"\n--- FERTIG nach {dt:.1f} s ---", flush=True)
print("Roh-IDs:", new.tolist(), flush=True)
text = processor.decode(new, skip_special_tokens=True).strip()
print("Entcode:", repr(text), flush=True)
print("OK" if text else "LEER", flush=True)
cap.release()
sys.exit(0 if text else 1)
