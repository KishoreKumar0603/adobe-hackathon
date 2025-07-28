import fitz  # PyMuPDF
import json
import re
import os
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np

DATASET_PATH = Path("dataset")
INPUT_DIR = Path("input")
LABELS_DIR = Path("labels")

DATASET_PATH.mkdir(exist_ok=True)

def aggressive_text_cleaning(text):
    """More aggressive text cleaning for better matching"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\w\s\-\.]', ' ', text)  # Keep only alphanumeric, spaces, hyphens, dots
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_text_blocks(pdf_path):
    """Extract text blocks with improved handling"""
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num in range(len(doc)):
        page_blocks = []
        
        for block in doc[page_num].get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                line_features = {
                    "font_sizes": [],
                    "bold_flags": [],
                    "y_positions": [],
                    "x_start": float('inf'),
                    "x_end": 0
                }
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        line_text += text + " "
                        line_features["font_sizes"].append(span["size"])
                        line_features["bold_flags"].append(bool(span["flags"] & 2**4))
                        line_features["y_positions"].append(span["bbox"][1])
                        line_features["x_start"] = min(line_features["x_start"], span["bbox"][0])
                        line_features["x_end"] = max(line_features["x_end"], span["bbox"][2])
                
                if line_text.strip() and len(line_text.strip()) >= 2:  # Reduced minimum length
                    page_blocks.append({
                        "text": line_text.strip(),
                        "page": page_num + 1,
                        "font_size": max(line_features["font_sizes"]) if line_features["font_sizes"] else 12,
                        "bold": any(line_features["bold_flags"]),
                        "y_pos": min(line_features["y_positions"]) if line_features["y_positions"] else 0,
                        "x_start": line_features["x_start"] if line_features["x_start"] != float('inf') else 0,
                        "width": line_features["x_end"] - line_features["x_start"] if line_features["x_start"] != float('inf') else 0
                    })
        
        all_blocks.extend(page_blocks)
    
    doc.close()
    return all_blocks

def enhanced_text_matching(block_text, heading_text, debug=False):
    """Enhanced text matching with multiple strategies and lower thresholds"""
    
    # Original texts for reference
    orig_block = block_text
    orig_heading = heading_text
    
    # Clean both texts aggressively
    block_clean = aggressive_text_cleaning(block_text)
    heading_clean = aggressive_text_cleaning(heading_text)
    
    if debug:
        print(f"  Original block: '{orig_block}'")
        print(f"  Original heading: '{orig_heading}'")
        print(f"  Cleaned block: '{block_clean}'")
        print(f"  Cleaned heading: '{heading_clean}'")
    
    # Skip if either is too short after cleaning
    if len(block_clean) < 2 or len(heading_clean) < 2:
        return False, 0.0
    
    # Strategy 1: Exact match after aggressive cleaning
    if block_clean == heading_clean:
        if debug: print("    → Exact match after cleaning!")
        return True, 1.0
    
    # Strategy 2: Remove numbering patterns and compare
    # Handle various numbering patterns: "1.", "1.1", "1.1.1", etc.
    block_no_num = re.sub(r'^\d+(\.\d+)*\.?\s*', '', block_clean).strip()
    heading_no_num = re.sub(r'^\d+(\.\d+)*\.?\s*', '', heading_clean).strip()
    
    if block_no_num and heading_no_num and len(block_no_num) >= 2 and len(heading_no_num) >= 2:
        if block_no_num == heading_no_num:
            if debug: print(f"    → Number-stripped exact match! '{block_no_num}' == '{heading_no_num}'")
            return True, 0.98
        
        # Fuzzy match on number-stripped text
        fuzzy_no_num = SequenceMatcher(None, block_no_num, heading_no_num).ratio()
        if fuzzy_no_num > 0.85:  # High threshold for number-stripped
            if debug: print(f"    → Number-stripped fuzzy match! Ratio: {fuzzy_no_num}")
            return True, fuzzy_no_num * 0.95
    
    # Strategy 3: Substring matching (more lenient)
    longer_text = block_clean if len(block_clean) > len(heading_clean) else heading_clean
    shorter_text = heading_clean if len(block_clean) > len(heading_clean) else block_clean
    
    if shorter_text in longer_text and len(shorter_text) >= 3:
        ratio = len(shorter_text) / len(longer_text)
        if ratio > 0.3:  # More lenient threshold
            if debug: print(f"    → Substring match! Ratio: {ratio}")
            return True, min(0.9, ratio + 0.3)  # Boost confidence but cap at 0.9
    
    # Strategy 4: Word-based matching
    block_words = set(block_clean.split())
    heading_words = set(heading_clean.split())
    
    if len(heading_words) > 0:
        common_words = block_words & heading_words
        word_overlap = len(common_words) / len(heading_words)
        
        # Also check reverse overlap
        if len(block_words) > 0:
            reverse_overlap = len(common_words) / len(block_words)
            word_overlap = max(word_overlap, reverse_overlap)
        
        if word_overlap > 0.5 and len(common_words) >= 1:  # More lenient
            if debug: print(f"    → Word overlap match! Ratio: {word_overlap}, Common words: {common_words}")
            return True, min(0.85, word_overlap + 0.2)  # Boost confidence
    
    # Strategy 5: Fuzzy matching with lower threshold
    fuzzy_ratio = SequenceMatcher(None, block_clean, heading_clean).ratio()
    if fuzzy_ratio > 0.6:  # Lower threshold
        if debug: print(f"    → Fuzzy match! Ratio: {fuzzy_ratio}")
        return True, fuzzy_ratio
    
    # Strategy 6: Handle common PDF extraction issues
    # Remove extra spaces and try again
    block_compressed = re.sub(r'\s+', '', block_clean)
    heading_compressed = re.sub(r'\s+', '', heading_clean)
    
    if block_compressed == heading_compressed:
        if debug: print("    → Space-compressed exact match!")
        return True, 0.95
    
    compressed_fuzzy = SequenceMatcher(None, block_compressed, heading_compressed).ratio()
    if compressed_fuzzy > 0.8:
        if debug: print(f"    → Space-compressed fuzzy match! Ratio: {compressed_fuzzy}")
        return True, compressed_fuzzy * 0.9
    
    if debug: print(f"    → No match. Best fuzzy ratio: {fuzzy_ratio}")
    return False, fuzzy_ratio

def label_blocks(blocks, headings, debug=False):
    """Label text blocks with improved matching and page-aware logic"""
    labeled = []
    match_stats = {"exact": 0, "fuzzy": 0, "none": 0}
    
    if debug:
        print(f"\n=== Labeling {len(blocks)} blocks against {len(headings)} headings ===")
        print("Sample headings from ground truth:")
        for i, h in enumerate(headings[:10]):  # Show more examples
            print(f"  {i+1}. '{h['text']}' (level: {h['level']}, page: {h.get('page', 'N/A')})")
    
    for i, block in enumerate(blocks):
        label = "non-heading"
        best_score = 0
        matched_heading = None
        
        if debug and i < 30:  # Debug more blocks
            print(f"\nBlock {i}: '{block['text']}' (page {block['page']}, font: {block['font_size']}, bold: {block['bold']})")
        
        # Try to match with each heading
        for heading in headings:
            # Give slight preference to same-page matches
            page_bonus = 0.05 if heading.get('page') == block['page'] else 0
            
            is_match, score = enhanced_text_matching(
                block["text"], 
                heading["text"], 
                debug=(debug and i < 30)
            )
            
            final_score = score + page_bonus
            
            if is_match and final_score > best_score:
                label = heading["level"]
                best_score = final_score
                matched_heading = heading
                
                if debug and i < 30:
                    print(f"    ✓ MATCHED with '{heading['text']}' (score: {final_score:.3f}, page bonus: {page_bonus})")
        
        # Additional heuristics for missed headings
        if label == "non-heading":
            # Check if this looks like a heading based on formatting
            median_font = np.median([b["font_size"] for b in blocks])
            is_large_font = block["font_size"] > median_font * 1.2
            is_bold = block["bold"]
            is_short = len(block["text"]) < 100
            starts_with_number = bool(re.match(r'^\d+', block["text"]))
            
            # Heuristic scoring for potential headings
            heuristic_score = 0
            if is_large_font: heuristic_score += 0.3
            if is_bold: heuristic_score += 0.2
            if is_short: heuristic_score += 0.1
            if starts_with_number: heuristic_score += 0.1
            
            # If it looks like a heading but no match found, try fuzzy matching with lower threshold
            if heuristic_score > 0.4:
                for heading in headings:
                    fuzzy_score = SequenceMatcher(None, 
                                                aggressive_text_cleaning(block["text"]), 
                                                aggressive_text_cleaning(heading["text"])).ratio()
                    if fuzzy_score > 0.4:  # Very low threshold for heuristic matches
                        label = heading["level"]
                        best_score = fuzzy_score * 0.7  # Lower confidence
                        matched_heading = heading
                        if debug and i < 30:
                            print(f"    ✓ HEURISTIC MATCH with '{heading['text']}' (score: {fuzzy_score:.3f})")
                        break
        
        # Calculate additional features
        median_font = np.median([b["font_size"] for b in blocks])
        
        labeled_block = {
            "text": block["text"],
            "features": {
                "font_size": block["font_size"],
                "bold": block["bold"],
                "y_pos": block["y_pos"],
                "page": block["page"],
                "font_ratio": block["font_size"] / median_font,
                "text_length": len(block["text"]),
                "has_numbers": bool(re.search(r'\d', block["text"])),
                "is_upper": block["text"].isupper(),
                "starts_with_number": bool(re.match(r'^\d+', block["text"])),
                "x_start": block.get("x_start", 0),
                "width": block.get("width", 0),
                "word_count": len(block["text"].split()),
                "ends_with_colon": block["text"].strip().endswith(':'),
                "has_special_chars": bool(re.search(r'[^\w\s]', block["text"]))
            },
            "label": label,
            "confidence": min(1.0, best_score)  # Cap confidence at 1.0
        }
        
        labeled.append(labeled_block)
        
        # Update stats
        if label != "non-heading":
            if best_score >= 0.9:
                match_stats["exact"] += 1
            else:
                match_stats["fuzzy"] += 1
        else:
            match_stats["none"] += 1
        
        if debug and i < 30:
            print(f"  → Final Label: {label} (confidence: {best_score:.3f})")
    
    if debug:
        print(f"\n=== Match Statistics ===")
        print(f"High-confidence matches (≥0.9): {match_stats['exact']}")
        print(f"Fuzzy matches (<0.9): {match_stats['fuzzy']}")
        print(f"No matches: {match_stats['none']}")
        
        # Show detailed label distribution
        label_counts = {}
        confidence_by_label = {}
        for item in labeled:
            label = item["label"]
            conf = item["confidence"]
            
            label_counts[label] = label_counts.get(label, 0) + 1
            
            if label not in confidence_by_label:
                confidence_by_label[label] = []
            confidence_by_label[label].append(conf)
        
        print(f"\n📊 Label Distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(labeled)) * 100
            avg_conf = np.mean(confidence_by_label[label]) if confidence_by_label[label] else 0
            print(f"  {label}: {count} ({percentage:.1f}%) - Avg confidence: {avg_conf:.3f}")
    
    return labeled

def main():
    all_data = []
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    for i, pdf_file in enumerate(pdf_files):
        label_file = LABELS_DIR / f"{pdf_file.stem}.json"
        
        if not label_file.exists():
            print(f"⚠️ Skipping {pdf_file.name} — no matching label found.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})")
        
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_json = json.load(f)
            
            headings = label_json.get("outline", [])
            print(f"Found {len(headings)} ground truth headings")
            
            if not headings:
                print("⚠️ No headings in ground truth, skipping...")
                continue
            
            blocks = extract_text_blocks(pdf_file)
            print(f"Extracted {len(blocks)} text blocks")
            
            # Enable debug for first file or specify file
            debug_mode = (i == 0)  # or pdf_file.name == "specific_file.pdf"
            labeled_blocks = label_blocks(blocks, headings, debug=debug_mode)
            
            all_data.extend(labeled_blocks)
            
            # Show quick stats for this file
            heading_count = sum(1 for item in labeled_blocks if item["label"] != "non-heading")
            avg_confidence = np.mean([item["confidence"] for item in labeled_blocks if item["label"] != "non-heading"])
            print(f"📈 File stats: {heading_count} headings found, avg confidence: {avg_confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save training data
    output_path = DATASET_PATH / "train_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as out_file:
        for item in all_data:
            json.dump(item, out_file, ensure_ascii=False)
            out_file.write("\n")
    
    # Print final comprehensive statistics
    print(f"\n{'='*60}")
    print(f"✅ Training data prepared: {len(all_data)} entries")
    print(f"📁 Saved to: {output_path}")
    
    # Calculate comprehensive stats
    label_counts = {}
    confidence_stats = {}
    
    for item in all_data:
        label = item["label"]
        conf = item["confidence"]
        
        label_counts[label] = label_counts.get(label, 0) + 1
        
        if label not in confidence_stats:
            confidence_stats[label] = []
        confidence_stats[label].append(conf)
    
    print(f"\n📊 Final Comprehensive Statistics:")
    total_headings = sum(count for label, count in label_counts.items() if label != "non-heading")
    
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(all_data)) * 100
        
        if confidence_stats[label]:
            avg_conf = np.mean(confidence_stats[label])
            min_conf = np.min(confidence_stats[label])
            max_conf = np.max(confidence_stats[label])
            high_conf_count = sum(1 for c in confidence_stats[label] if c >= 0.8)
            
            print(f"  {label}: {count} ({percentage:.1f}%)")
            print(f"    Confidence - Avg: {avg_conf:.3f}, Min: {min_conf:.3f}, Max: {max_conf:.3f}")
            print(f"    High confidence (≥0.8): {high_conf_count}/{count} ({100*high_conf_count/count:.1f}%)")
        else:
            print(f"  {label}: {count} ({percentage:.1f}%) - No confidence data")
    
    print(f"\n🎯 Overall heading detection rate: {total_headings} headings found")
    if total_headings > 0:
        overall_avg_conf = np.mean([item["confidence"] for item in all_data if item["label"] != "non-heading"])
        print(f"🎯 Overall average confidence for headings: {overall_avg_conf:.3f}")

if __name__ == "__main__":
    main()