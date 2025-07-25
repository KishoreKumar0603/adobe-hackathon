import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

MODEL_DIR = "./model"
DATA_PATH = "dataset/train_data.jsonl"
label_map = {"non-heading": 0, "H1": 1, "H2": 2, "H3": 3}

# Custom dataset class
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

# Load and filter data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

filtered_data = [d for d in raw_data if d["label"] in label_map]

# Split data
train_data, val_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

# Load tokenizer and dataset
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_dataset = HeadingDataset(train_data, tokenizer)
val_dataset = HeadingDataset(val_data, tokenizer)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training settings
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")

# Save model and tokenizer
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("✅ Model training complete and saved!")
