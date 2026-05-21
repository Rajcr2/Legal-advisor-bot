import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_PATH    = "legal_intent_dataset.json"
MODEL_SAVE_PATH = "intent_classifier_model"
BASE_MODEL      = "distilbert-base-uncased"
LABELS          = ["legal_query", "situation_based"]
EPOCHS          = 3
BATCH_SIZE      = 8
LEARNING_RATE   = 2e-5
MAX_LENGTH      = 128

class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


if __name__ == "__main__":
    # Load
    with open(DATASET_PATH) as f:
        data = json.load(f)
    texts  = [s["text"]  for s in data["samples"]]
    labels = [s["label"] for s in data["samples"]]
    print(f"[Dataset] {len(texts)} samples loaded")

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, stratify=labels, random_state=42)

    # Tokenize
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    #encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH)

    # Train
    model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            save_strategy="epoch",
            logging_steps=50,
            report_to="none"
        ),
        train_dataset=LegalDataset(train_encodings, y_train)
    ).train()

    # Evaluate
    model.eval()
    encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        y_pred = torch.argmax(model(**encodings).logits, dim=1).numpy()
    
    print("\n" + "="*50)
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))

    # Save
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"[Done] Model saved to '{MODEL_SAVE_PATH}/'")
