import fitz  # PyMuPDF
import os
import json
from difflib import SequenceMatcher


def get_spans_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    spans.append({
                        "text": text,
                        "font_size": round(span["size"], 1),
                        "bold": "Bold" in span.get("font", ""),
                        "y_pos": span["origin"][1],
                        "page": page.number + 1
                    })
    return spans


def match_label(text, page, outline):
    best_score = 0
    best_label = "non-heading"
    for item in outline:
        if item["page"] != page:
            continue
        score = SequenceMatcher(None, text.lower(), item["text"].lower()).ratio()
        if score > best_score:
            best_score = score
            best_label = item["level"]
    return best_label if best_score > 0.75 else "non-heading"


def build_dataset(input_dir, output_dir, output_path):
    all_data = []
    files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    total = 0
    for pdf_file in files:
        pdf_path = os.path.join(input_dir, pdf_file)
        json_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".json"))
        if not os.path.exists(json_path):
            print(f"⚠️ Missing JSON for {pdf_file}, skipping.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            truth = json.load(f)

        outline = truth.get("outline", [])
        spans = get_spans_from_pdf(pdf_path)

        for span in spans:
            label = match_label(span["text"], span["page"], outline)
            features = f"font_size:{span['font_size']} bold:{span['bold']} y_pos:{span['y_pos']} page:{span['page']}"
            all_data.append({
                "text": span["text"],
                "features": features,
                "label": label
            })
            total += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Training data written to {output_path}")
    print(f"📊 Total entries: {total}")


# 🔁 Run this
build_dataset("input", "output", "dataset/train_data.jsonl")
