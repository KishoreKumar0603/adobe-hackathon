import os
import json

def list_pdfs_in_directory(directory_path):
    return [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
