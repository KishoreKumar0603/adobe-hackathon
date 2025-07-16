import os
from span_extractor import extract_spans
from outline_builder import build_outline
from utils import list_pdfs_in_directory, save_json, ensure_directory

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def main():
    ensure_directory(OUTPUT_DIR)
    use_ml = True  # Enable ML reranking

    for filename in list_pdfs_in_directory(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".json"))

        spans = extract_spans(input_path)
        result = build_outline(spans, use_ml=use_ml)
        save_json(result, output_path)

    print("✅ Done!")

if __name__ == "__main__":
    main()
