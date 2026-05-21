import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import uvicorn

MODEL_PATH = "intent_classifier_model"
LABELS     = ["legal_query", "situation_based"]
MAX_LENGTH = 128

app = FastAPI(title="Legal Intent Service")

# Load model once at startup
print("[IntentService] Loading model...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("[IntentService] Model ready on", device)


class QueryRequest(BaseModel):
    query: str


@app.post("/predict")
def predict(request: QueryRequest):
    inputs = tokenizer(
        request.query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    idx    = int(torch.argmax(logits, dim=1).item())
    intent = LABELS[idx]
    conf   = float(torch.softmax(logits, dim=1)[0][idx].item())

    print(f"[IntentService] '{request.query[:60]}' → {intent} ({conf:.2f})")
    return {"intent": intent, "confidence": round(conf, 4)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
