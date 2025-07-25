# prepare_training_data.py

import os
import json
from span_extractor import extract_spans
from tqdm import tqdm

RAW_DIR = "dataset/raw"
OUTPUT_FILE = "dataset/train_data.jsonl"

# Labels to track
allowed_labels = {"H1", "H2", "H3", "non-heading"}

def load_ground_truth(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_data():
    samples = []
    for filename in tqdm(os.listdir(RAW_DIR)):
        if not filename.endswith(".pdf"):
            continue
        
        base = filename[:-4]
        pdf_path = os.path.join(RAW_DIR, base + ".pdf")
        json_path = os.path.join(RAW_DIR, base + ".json")

        if not os.path.exists(json_path):
            print(f"⚠️ Missing JSON for {filename}")
            continue

        ground_truth = load_ground_truth(json_path)
        spans = extract_spans(pdf_path)

        # Build mapping of ground truth headings
        truth_map = {
            (h["page"], h["text"].strip()): h["level"]
            for h in ground_truth["outline"]
            if h["level"] in allowed_labels
        }

        for span in spans:
            key = (span["page"], span["text"].strip())
            label = truth_map.get(key, "non-heading")

            # Filter only known labels
            if label not in allowed_labels:
                continue

            features = f"font_size:{span['size']} bold:{span['bold']} y_pos:{span['y']} page:{span['page']}"
            samples.append({
                "text": span["text"],
                "features": features,
                "label": label
            })

    # Save as JSON Lines
    os.makedirs("dataset", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            json.dump(sample, f)
            f.write("\n")

    print(f"✅ Created training data: {OUTPUT_FILE} with {len(samples)} entries.")

if __name__ == "__main__":
    prepare_data()
