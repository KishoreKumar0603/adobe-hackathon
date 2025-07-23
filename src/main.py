# src/main.py

import os
import json
import argparse
from tqdm import tqdm

# Import the logic from the other files
from round_1a import process_round_1a
from round_1b import process_round_1b

def run_1a(input_dir, output_dir):
    print(f"Running Round 1A: Processing all PDFs in {input_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs for Round 1A"):
        input_path = os.path.join(input_dir, pdf_file)
        result = process_round_1a(input_path)
        
        output_filename = os.path.splitext(pdf_file)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
            
    print(f"Round 1A complete. Outputs are in {output_dir}")


def run_1b(input_dir, output_dir, config_file):
    print(f"Running Round 1B with config: {config_file}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_path = os.path.join(input_dir, config_file)
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    persona = config.get("persona")
    job_to_be_done = config.get("job_to_be_done")
    doc_filenames = config.get("documents", [])
    
    if not all([persona, job_to_be_done, doc_filenames]):
        raise ValueError("Config file is missing persona, job_to_be_done, or documents list.")

    pdf_paths = [os.path.join(input_dir, fname) for fname in doc_filenames]
    
    result = process_round_1b(pdf_paths, persona, job_to_be_done)
    
    output_path = os.path.join(output_dir, 'challenge1b_output.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
        
    print(f"Round 1B complete. Output is in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adobe Hackathon 2025 - Connecting the Dots")
    parser.add_argument("round", choices=["round1a", "round1b"], help="Which round to execute.")
    parser.add_argument("--input_dir", required=True, help="Input directory path.")
    parser.add_argument("--output_dir", required=True, help="Output directory path.")
    parser.add_argument("--config_file", default="config.json", help="Config file for Round 1B.")
    
    args = parser.parse_args()
    
    if args.round == "round1a":
        run_1a(args.input_dir, args.output_dir)
    elif args.round == "round1b":
        run_1b(args.input_dir, args.output_dir, args.config_file)
