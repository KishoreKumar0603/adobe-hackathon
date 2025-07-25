import os
from span_extractor import extract_spans
from outline_builder import build_outline
from utils import save_json

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_pdf(input_path, output_path):
    spans = extract_spans(input_path)
    result = build_outline(spans, use_ml=True)
    save_json(result, output_path)

def batch_process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")
            print(f"📝 Processing {filename}...")
            process_pdf(input_path, output_path)

if __name__ == "__main__":
    batch_process()
