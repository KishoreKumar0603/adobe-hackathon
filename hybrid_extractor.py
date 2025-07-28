import json
import re
import numpy as np
from pathlib import Path
import joblib
import fitz  # PyMuPDF
from typing import List, Dict, Optional
from difflib import SequenceMatcher

class HybridPDFOutlineExtractor:
    def __init__(self, model_path: str = "models/heading_classifier.joblib"):
        self.model_path = Path(model_path)
        self.scaler_path = Path(model_path).parent / "feature_scaler.joblib"
        self.metadata_path = Path(model_path).parent / "model_metadata.json"
        
        # Load ML model if available
        self.ml_model = None
        self.scaler = None
        self.feature_names = None
        
        if (self.model_path.exists() and 
            self.scaler_path.exists() and 
            self.metadata_path.exists()):
            try:
                self.ml_model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata['feature_names']
                print("✅ ML model loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load ML model: {e}")
                print("Falling back to heuristic-only mode")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text to fix extraction issues"""
        if not text:
            return ""
        
        # Fix common OCR/extraction errors
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Space between letter and number
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Space between number and letter
        
        # Remove repeated characters that might be OCR errors
        text = re.sub(r'(.)\1{3,}', r'\1', text)  # Remove 4+ repeated chars
        
        # Fix common word fragments
        replacements = {
            r'\bquest\s*f\b': 'quest for',
            r'\bPr\s*r\b': 'Pr',
            r'\boposal\b': 'oposal',
            r'\bY\s*ou\s*T\s*HERE\b': 'You THERE',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()

    def is_corrupted_text(self, text: str) -> bool:
        """Detect if text appears to be corrupted/fragmented"""
        if not text or len(text) < 3:
            return True
            
        # Check for excessive fragmentation
        fragments = len(re.findall(r'\b\w{1,2}\b', text))  # Single/double letter words
        total_words = len(text.split())
        
        if total_words > 0 and fragments / total_words > 0.4:  # More than 40% fragments
            return True
        
        # Check for repeated patterns (like "RFP: R RFP:")
        words = text.split()
        if len(words) > 2:
            for i in range(len(words) - 1):
                if words[i] == words[i + 1] and len(words[i]) > 1:
                    return True
        
        # Check for excessive repeated characters
        if re.search(r'(.)\1{4,}', text):  # 5+ repeated chars
            return True
            
        return False

    def normalize_heading_text(self, text: str) -> str:
        """Normalize heading text for comparison"""
        text = self.clean_text(text)
        text = re.sub(r'^\d+\.?\s*', '', text)  # Remove numbering
        text = re.sub(r'^(chapter|section|part)\s+\d+\.?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        a_norm = self.normalize_heading_text(a)
        b_norm = self.normalize_heading_text(b)
        return SequenceMatcher(None, a_norm, b_norm).ratio()

    def is_duplicate_or_similar(self, new_heading: str, existing_headings: List[str], threshold: float = 0.8) -> bool:
        """Enhanced duplicate detection"""
        if not new_heading or not existing_headings:
            return False
            
        new_norm = self.normalize_heading_text(new_heading)
        
        # Skip very short normalized text
        if len(new_norm) < 3:
            return True
        
        for existing in existing_headings:
            existing_norm = self.normalize_heading_text(existing)
            
            # Skip if existing is also very short
            if len(existing_norm) < 3:
                continue
                
            similarity_score = self.similarity(new_heading, existing)
            
            # Check for substring relationships
            if len(new_norm) > 0 and len(existing_norm) > 0:
                # One contains the other and they're reasonably similar in length
                if (new_norm in existing_norm and len(new_norm) > len(existing_norm) * 0.7) or \
                   (existing_norm in new_norm and len(existing_norm) > len(new_norm) * 0.7):
                    similarity_score = max(similarity_score, 0.85)
            
            if similarity_score >= threshold:
                return True
        
        return False

    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract and clean text blocks"""
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_blocks = page.get_text("dict")["blocks"]
            
            for block in text_blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    if not line.get("spans"):
                        continue
                    
                    # Collect spans and sort by x position
                    line_spans = []
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            line_spans.append({
                                "text": text,
                                "font_size": span["size"],
                                "bold": bool(span["flags"] & 2**4),
                                "bbox": span["bbox"],
                                "x_pos": span["bbox"][0]
                            })
                    
                    if not line_spans:
                        continue
                    
                    # Sort spans by x position for proper reading order
                    line_spans.sort(key=lambda x: x["x_pos"])
                    
                    # Reconstruct text with proper spacing
                    line_text = ""
                    for i, span in enumerate(line_spans):
                        if i > 0:
                            # Add space if there's a gap between spans
                            prev_span = line_spans[i-1]
                            gap = span["x_pos"] - (prev_span["bbox"][2])
                            if gap > 3:  # Significant gap
                                line_text += " "
                        line_text += span["text"]
                    
                    # Clean the text
                    line_text = self.clean_text(line_text)
                    
                    # Skip corrupted or very short text
                    if len(line_text) < 3 or self.is_corrupted_text(line_text):
                        continue
                    
                    # Get formatting info
                    font_sizes = [span["font_size"] for span in line_spans]
                    bold_flags = [span["bold"] for span in line_spans]
                    
                    # Use largest font and any bold
                    dominant_font_size = max(font_sizes)
                    is_bold = any(bold_flags)
                    
                    # Bounding box
                    x_coords = [span["bbox"][0] for span in line_spans] + [span["bbox"][2] for span in line_spans]
                    y_coords = [span["bbox"][1] for span in line_spans] + [span["bbox"][3] for span in line_spans]
                    
                    all_blocks.append({
                        "text": line_text,
                        "page": page_num + 1,
                        "font_size": dominant_font_size,
                        "bold": is_bold,
                        "y_pos": min(y_coords),
                        "x_start": min(x_coords),
                        "width": max(x_coords) - min(x_coords)
                    })
        
        doc.close()
        return all_blocks

    def extract_text_features(self, text: str) -> Dict:
        """Extract text-based features for ML model"""
        features = {}
        
        words = text.split()
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['char_count'] = len(text.strip())
        
        # Capitalization features
        features['is_all_caps'] = text.isupper()
        features['is_title_case'] = text.istitle()
        features['starts_with_capital'] = text[0].isupper() if text else False
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Pattern features
        features['has_numbers'] = bool(re.search(r'\d', text))
        features['starts_with_number'] = bool(re.match(r'^\d+', text))
        features['has_colon'] = ':' in text
        features['has_period'] = '.' in text
        features['ends_with_colon'] = text.strip().endswith(':')
        features['ends_with_period'] = text.strip().endswith('.')
        
        # Numbering patterns
        features['numbered_h1'] = bool(re.match(r'^\d+\.?\s', text))
        features['numbered_h2'] = bool(re.match(r'^\d+\.\d+\.?\s', text))
        features['numbered_h3'] = bool(re.match(r'^\d+\.\d+\.\d+\.?\s', text))
        features['has_numbering'] = features['numbered_h1'] or features['numbered_h2'] or features['numbered_h3']
        
        # Common heading words
        heading_words = {
            'introduction', 'conclusion', 'summary', 'overview', 'chapter', 
            'section', 'background', 'methodology', 'results', 'discussion',
            'abstract', 'references', 'appendix', 'bibliography', 'contents',
            'table', 'revision', 'history', 'acknowledgments', 'preface'
        }
        text_lower = text.lower()
        features['has_heading_words'] = any(word in text_lower for word in heading_words)
        features['heading_word_count'] = sum(1 for word in heading_words if word in text_lower)
        
        # Length categories
        features['is_very_short'] = len(text) < 10
        features['is_short'] = 10 <= len(text) < 50
        features['is_medium'] = 50 <= len(text) < 100
        features['is_long'] = len(text) >= 100
        
        # Special characters
        features['has_parentheses'] = '(' in text or ')' in text
        features['has_brackets'] = '[' in text or ']' in text
        features['has_quotes'] = '"' in text or "'" in text
        features['special_char_ratio'] = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        
        return features

    def is_likely_non_heading(self, text: str) -> bool:
        """Identify text that's unlikely to be a heading"""
        text_lower = text.lower().strip()
        
        # Common non-heading patterns
        non_heading_patterns = [
            r'^(page|fig|figure|table)\s+\d+',
            r'^\d+$',  # Just a number
            r'^www\.',  # URLs
            r'^http',   # URLs
            r'@\w+\.',  # Email addresses
            r'^\(\d+\)',  # Phone numbers
            r'^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # Phone numbers
            r'^[-=_]{3,}$',  # Separator lines
            r'^\s*[-•·]\s*$',  # Bullet points
        ]
        
        for pattern in non_heading_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Too many special characters
        special_chars = len(re.findall(r'[^\w\s]', text))
        if len(text) > 0 and special_chars / len(text) > 0.5:
            return True
            
        return False

    def heuristic_classification(self, block: Dict, median_font: float, all_blocks: List[Dict]) -> Dict:
        """Enhanced heuristic classification with context"""
        text = block["text"]
        font_size = block["font_size"]
        bold = block.get("bold", False)
        
        result = {
            "is_heading": False,
            "level": None,
            "confidence": 0.0,
            "reasons": []
        }
        
        # Quick exclusions
        if self.is_likely_non_heading(text):
            return result
        
        # Font size check - stricter threshold
        font_ratio = font_size / median_font
        if font_ratio < 1.1:  # Must be at least 10% larger
            return result
        
        confidence = 0.0
        reasons = []
        
        # Strong numbering patterns (highest priority)
        if re.match(r'^\d+\.\d+\.\d+\.?\s+\w', text):  # 1.1.1 Something
            result["level"] = "H3"
            confidence += 0.95
            reasons.append("numbered_h3")
        elif re.match(r'^\d+\.\d+\.?\s+\w', text):  # 1.1 Something
            result["level"] = "H2"
            confidence += 0.95
            reasons.append("numbered_h2")
        elif re.match(r'^\d+\.?\s+\w', text):  # 1. Something
            result["level"] = "H1"
            confidence += 0.95
            reasons.append("numbered_h1")
        
        # Strong heading keywords (case-insensitive)
        if not result["level"]:
            strong_patterns = [
                (r'^(introduction|conclusion|summary|overview|abstract)$', 'H1', 0.9),
                (r'^(table\s+of\s+contents|contents)$', 'H1', 0.9),
                (r'^(chapter|section)\s+\d+', 'H1', 0.85),
                (r'^(background|methodology|results|discussion|analysis)$', 'H1', 0.8),
                (r'^(references|appendix|bibliography|acknowledgments)$', 'H1', 0.8),
            ]
            
            text_clean = re.sub(r'[^\w\s]', '', text.lower().strip())
            for pattern, level, conf in strong_patterns:
                if re.match(pattern, text_clean):
                    result["level"] = level
                    confidence += conf
                    reasons.append(f"strong_pattern")
                    break
        
        # Font-based classification (more conservative)
        if not result["level"]:
            if font_ratio >= 1.5 and bold:
                result["level"] = "H1"
                confidence += 0.7
                reasons.append("very_large_bold_font")
            elif font_ratio >= 1.3:
                result["level"] = "H1" if bold else "H2"
                confidence += 0.6
                reasons.append("large_font")
            elif font_ratio >= 1.2 and bold:
                result["level"] = "H2"
                confidence += 0.5
                reasons.append("medium_bold_font")
        
        # Additional checks for confidence
        if result["level"]:
            # Bold text bonus
            if bold:
                confidence += 0.1
                reasons.append("bold")
            
            # ALL CAPS bonus (but not too much)
            if text.isupper() and 5 <= len(text) <= 50:
                confidence += 0.15
                reasons.append("all_caps")
            
            # Title case bonus
            if text.istitle() and len(text.split()) <= 8:
                confidence += 0.1
                reasons.append("title_case")
            
            # Length penalties/bonuses
            if len(text) > 150:
                confidence -= 0.3
                reasons.append("too_long")
            elif 5 <= len(text) <= 80:
                confidence += 0.05
                reasons.append("good_length")
            elif len(text) < 5:
                confidence -= 0.2
                reasons.append("too_short")
        
        # Final decision - higher threshold
        if result["level"] and confidence > 0.4:
            result["is_heading"] = True
            result["confidence"] = min(confidence, 1.0)
            result["reasons"] = reasons
        
        return result
    
    def ml_classification(self, block: Dict, median_font: float) -> Dict:
        """ML-based classification"""
        if self.ml_model is None or self.scaler is None:
            return {"is_heading": False, "level": None, "confidence": 0.0, "reasons": ["no_ml_model"]}
        
        try:
            text_features = self.extract_text_features(block["text"])
            
            feature_vector = {
                'font_size': float(block.get('font_size', 12)),
                'bold': int(block.get('bold', False)),
                'font_ratio': float(block.get('font_size', 12) / median_font),
                'y_pos': float(block.get('y_pos', 0)),
                'x_start': float(block.get('x_start', 0)),
                'width': float(block.get('width', 0)),
                'page': int(block.get('page', 1)),
                'word_count_feat': len(block["text"].split()),
                'ends_with_colon_feat': int(block["text"].strip().endswith(':')),
                'has_special_chars_feat': int(bool(re.search(r'[^\w\s]', block["text"]))),
                **{k: (int(v) if isinstance(v, bool) else float(v)) for k, v in text_features.items()}
            }
            
            feature_array = np.array([[feature_vector.get(feat, 0) for feat in self.feature_names]])
            feature_array = np.nan_to_num(feature_array, nan=0.0)
            
            feature_array_scaled = self.scaler.transform(feature_array)
            
            prediction = self.ml_model.predict(feature_array_scaled)[0]
            probabilities = self.ml_model.predict_proba(feature_array_scaled)[0]
            
            class_idx = list(self.ml_model.classes_).index(prediction)
            confidence = probabilities[class_idx]
            
            return {
                "is_heading": prediction != "non-heading",
                "level": prediction if prediction != "non-heading" else None,
                "confidence": float(confidence),
                "reasons": ["ml_prediction"]
            }
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return {"is_heading": False, "level": None, "confidence": 0.0, "reasons": ["ml_error"]}
    
    def detect_title(self, blocks: List[Dict]) -> str:
        """Improved title detection"""
        if not blocks:
            return ""
        
        first_page_blocks = [b for b in blocks if b["page"] == 1]
        if not first_page_blocks:
            return ""
        
        # Sort by y position (top to bottom)
        first_page_blocks.sort(key=lambda x: x["y_pos"])
        
        # Look for title in first few blocks with larger fonts
        median_font = np.median([b["font_size"] for b in first_page_blocks])
        
        title_candidates = []
        for block in first_page_blocks[:10]:  # Check first 10 blocks
            text = block["text"].strip()
            font_ratio = block["font_size"] / median_font
            
            # Good title criteria
            if (5 <= len(text) <= 200 and  # Reasonable length
                font_ratio >= 1.1 and  # Larger font
                not self.is_likely_non_heading(text) and  # Not obvious non-heading
                not self.is_corrupted_text(text) and  # Not corrupted
                not re.match(r'^\d+\.', text)):  # Not numbered
                
                title_candidates.append({
                    "text": text,
                    "font_ratio": font_ratio,
                    "y_pos": block["y_pos"],
                    "bold": block.get("bold", False)
                })
        
        if not title_candidates:
            return ""
        
        # Sort by font size (descending) then by position (ascending)
        title_candidates.sort(key=lambda x: (-x["font_ratio"], x["y_pos"]))
        
        return title_candidates[0]["text"]
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """Main extraction method"""
        try:
            blocks = self.extract_text_blocks(pdf_path)
            if not blocks:
                return {"title": "", "outline": []}
            
            # Detect title
            title = self.detect_title(blocks)
            
            # Calculate median font size for the document
            median_font = np.median([b["font_size"] for b in blocks])
            
            # Classify each block
            headings = []
            seen_heading_texts = []
            
            for block in blocks:
                text = block["text"].strip()
                
                # Skip very short texts or corrupted text
                if len(text) < 3 or self.is_corrupted_text(text):
                    continue
                
                # Skip if it's the title (avoid duplication)
                if title and self.similarity(text, title) > 0.9:
                    continue
                
                # Check for duplicates
                if self.is_duplicate_or_similar(text, seen_heading_texts):
                    continue
                
                # Get classifications
                heuristic_result = self.heuristic_classification(block, median_font, blocks)
                ml_result = self.ml_classification(block, median_font)
                
                # Combine classifications
                final_decision = self.combine_classifications(heuristic_result, ml_result)
                
                if final_decision["is_heading"] and final_decision["confidence"] > 0.5:  # Higher threshold
                    headings.append({
                        "level": final_decision["level"],
                        "text": text,
                        "page": block["page"],
                        "y_pos": block.get("y_pos", 0)
                    })
                    seen_heading_texts.append(text)
            
            # Sort headings by page and position
            headings.sort(key=lambda x: (x["page"], x.get("y_pos", 0)))
            
            # Clean output
            clean_headings = [
                {
                    "level": h["level"],
                    "text": h["text"],
                    "page": h["page"]
                }
                for h in headings
            ]
            
            return {"title": title, "outline": clean_headings}
            
        except Exception as e:
            print(f"Error extracting outline from {pdf_path}: {e}")
            return {"title": "", "outline": []}
    
    def combine_classifications(self, heuristic_result: Dict, ml_result: Dict) -> Dict:
        """Combine heuristic and ML classifications"""
        
        # If no ML model, use heuristic only
        if self.ml_model is None:
            return {**heuristic_result, "method": "heuristic_only"}
        
        # Both say not a heading
        if not heuristic_result["is_heading"] and not ml_result["is_heading"]:
            return {
                "is_heading": False,
                "level": None,
                "confidence": 0.0,
                "method": "both_agree_no"
            }
        
        # Both agree it's a heading
        if heuristic_result["is_heading"] and ml_result["is_heading"]:
            # Prefer heuristic for numbered patterns
            if any("numbered" in reason for reason in heuristic_result.get("reasons", [])):
                return {**heuristic_result, "method": "heuristic_numbered"}
            elif ml_result["confidence"] > heuristic_result["confidence"]:
                return {**ml_result, "method": "ml_higher_conf"}
            else:
                return {**heuristic_result, "method": "heuristic_higher_conf"}
        
        # Disagreement - use weighted confidence
        heuristic_weighted = heuristic_result["confidence"] * 0.6  # Favor heuristic slightly
        ml_weighted = ml_result["confidence"] * 0.4
        
        # Strong boost for numbered patterns
        if any("numbered" in reason for reason in heuristic_result.get("reasons", [])):
            heuristic_weighted *= 1.8
        
        if heuristic_weighted > ml_weighted and heuristic_result["is_heading"]:
            return {**heuristic_result, "method": "heuristic_wins"}
        elif ml_result["is_heading"] and ml_result["confidence"] > 0.7:  # High ML confidence
            return {**ml_result, "method": "ml_wins"}
        else:
            return {
                "is_heading": False,
                "level": None,
                "confidence": 0.0,
                "method": "neither_confident"
            }


# Main processing function
def process():
    """Process all PDFs in input directory"""
    from pathlib import Path
    
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    extractor = HybridPDFOutlineExtractor()
    
    for pdf_file in input_dir.glob("*.pdf"):
        try:
            print(f"Processing {pdf_file.name}...")
            result = extractor.extract_outline(str(pdf_file))
            
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {pdf_file.name} -> {len(result['outline'])} headings found")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            error_result = {"title": "", "outline": []}
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    process()