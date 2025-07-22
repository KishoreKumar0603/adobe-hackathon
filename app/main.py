import os
import json
from extractor import extract_outline

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def main():
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, file)
            data = extract_outline(pdf_path)

            output_file = os.path.splitext(file)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_file)

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
