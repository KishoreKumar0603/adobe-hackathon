# train_model.py

import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_scheduler
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

MODEL_DIR = "./model"
DATA_PATH = "dataset/train_data.jsonl"
label_map = {"non-heading": 0, "H1": 1, "H2": 2, "H3": 3}

# Custom dataset
class HeadingDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encodings = tokenizer(
            [d["text"] + " [SEP] " + d["features"] for d in data],
            truncation=True, padding=True, max_length=128
        )
        self.labels = [label_map[d["label"]] for d in data]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_accuracy(preds, labels):
    preds = np.argmax(preds, axis=1)
    return (preds == labels).mean()

# Load and filter data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

filtered_data = [d for d in raw_data if d["label"] in label_map]

# Split data
train_data, val_data = train_test_split(
    filtered_data, test_size=0.2, random_state=42, stratify=[d["label"] for d in filtered_data]
)

# Tokenizer and datasets
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_dataset = HeadingDataset(train_data, tokenizer)
val_dataset = HeadingDataset(val_data, tokenizer)

# Model setup
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(label_map)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Learning rate scheduler
num_training_steps = len(train_loader) * 3
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"🧪 Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} Training Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(logits)
            all_labels.extend(labels)
    acc = compute_accuracy(np.array(all_preds), np.array(all_labels))
    print(f"🎯 Validation Accuracy after Epoch {epoch+1}: {acc:.4f}")

# Save model
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("✅ Final model saved to", MODEL_DIR)
