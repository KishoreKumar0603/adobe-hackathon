import json
import re
import numpy as np
from pathlib import Path
import joblib
import fitz  # PyMuPDF
from typing import List, Dict, Optional
from collections import Counter

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
    
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with comprehensive features"""
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            
            for block in page.get_text("dict")["blocks"]:
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
                    
                    if line_text.strip() and len(line_text.strip()) >= 3:
                        # Calculate relative position on page
                        y_relative = line_features["y_positions"][0] / page_rect.height if line_features["y_positions"] else 0
                        x_relative = line_features["x_start"] / page_rect.width if line_features["x_start"] != float('inf') else 0
                        
                        all_blocks.append({
                            "text": line_text.strip(),
                            "page": page_num + 1,
                            "font_size": max(line_features["font_sizes"]) if line_features["font_sizes"] else 12,
                            "bold": any(line_features["bold_flags"]),
                            "y_pos": min(line_features["y_positions"]) if line_features["y_positions"] else 0,
                            "x_start": line_features["x_start"] if line_features["x_start"] != float('inf') else 0,
                            "width": line_features["x_end"] - line_features["x_start"] if line_features["x_start"] != float('inf') else 0,
                            "y_relative": y_relative,
                            "x_relative": x_relative
                        })
        
        doc.close()
        return all_blocks
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract text-based features (same as training)"""
        features = {}
        
        # Basic text features
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
    
    def is_likely_table_header_or_form_field(self, text: str) -> bool:
        """Detect table headers, form fields, and other non-heading content"""
        text_lower = text.lower().strip()
        
        # Table headers and form fields
        table_indicators = [
            'name', 'age', 'date', 'signature', 's.no', 'amount', 'relationship',
            'remarks', 'version', 'identifier', 'days', 'goals', 'mission statement'
        ]
        
        # Check if it's a single word table header
        if len(text.split()) <= 2 and any(indicator in text_lower for indicator in table_indicators):
            return True
            
        # Check for form-like patterns
        if re.match(r'^[a-z\s]+:?\s*$', text_lower) and len(text) < 30:
            return True
            
        # Check for dotted lines (table separators)
        if re.match(r'^\.{5,}$', text.strip()):
            return True
            
        # Check for URLs and technical identifiers
        if 'www.' in text_lower or '.com' in text_lower or re.match(r'^[A-Z0-9\-_]+$', text):
            return True
            
        return False
    
    def is_title_candidate(self, block: Dict, all_blocks: List[Dict]) -> bool:
        """Improved title detection"""
        text = block["text"].strip()
        
        # Skip if it's likely a table header or form field
        if self.is_likely_table_header_or_form_field(text):
            return False
            
        # Must be on first page
        if block["page"] != 1:
            return False
            
        # Must be reasonable length for a title
        if len(text) < 5 or len(text) > 200:
            return False
            
        # Must not be a numbered heading
        if re.match(r'^\d+\.', text):
            return False
            
        # Check if it's positioned like a title (top area of first page)
        if block.get("y_relative", 1) > 0.3:  # Not in top 30% of page
            return False
            
        # Check font size relative to other text on first page
        first_page_blocks = [b for b in all_blocks if b["page"] == 1]
        if first_page_blocks:
            font_sizes = [b["font_size"] for b in first_page_blocks]
            median_font = np.median(font_sizes)
            if block["font_size"] < median_font * 1.1:  # Not significantly larger
                return False
        
        return True
    
    def heuristic_classification(self, block: Dict, median_font: float, all_blocks: List[Dict]) -> Dict:
        """Enhanced heuristic classification with better filtering"""
        text = block["text"]
        font_size = block["font_size"]
        bold = block.get("bold", False)
        
        result = {
            "is_heading": False,
            "level": None,
            "confidence": 0.0,
            "reasons": []
        }
        
        # Skip table headers and form fields
        if self.is_likely_table_header_or_form_field(text):
            return result
        
        # Font size check - more lenient for numbered headings
        font_ratio = font_size / median_font
        has_numbering = bool(re.match(r'^\d+\.', text))
        
        if not has_numbering and font_ratio < 1.05:
            return result
        
        confidence = 0.0
        reasons = []
        
        # Strong numbering patterns (most reliable)
        if re.match(r'^\d+\.\d+\.\d+\.?\s+\w', text):  # 1.1.1 Title
            result["level"] = "H3"
            confidence += 0.95
            reasons.append("numbered_h3")
        elif re.match(r'^\d+\.\d+\.?\s+\w', text):  # 1.1 Title
            result["level"] = "H2"
            confidence += 0.95
            reasons.append("numbered_h2")
        elif re.match(r'^\d+\.?\s+\w', text):  # 1. Title
            result["level"] = "H1"
            confidence += 0.95
            reasons.append("numbered_h1")
        
        # Common heading patterns (must have good font size or be bold)
        if not result["level"] and (font_ratio >= 1.1 or bold):
            heading_indicators = [
                (r'^(table of contents|contents)$', 'H1', 0.9),
                (r'^(introduction|conclusion|summary|overview)$', 'H1', 0.8),
                (r'^(chapter|section)\s+\d+', 'H1', 0.8),
                (r'^(background|methodology|results|discussion)$', 'H1', 0.7),
                (r'^(abstract|references|appendix|bibliography)$', 'H1', 0.8),
                (r'^revision\s+history$', 'H1', 0.8),
                (r'^acknowledgements?$', 'H1', 0.8),
            ]
            
            text_lower = text.lower().strip()
            for pattern, level, conf in heading_indicators:
                if re.match(pattern, text_lower):
                    result["level"] = level
                    confidence += conf
                    reasons.append(f"pattern_{level}")
                    break
        
        # Font-based classification (only if no pattern match)
        if not result["level"] and font_ratio >= 1.1:
            if font_ratio >= 1.4:
                result["level"] = "H1"
                confidence += 0.6
                reasons.append("very_large_font")
            elif font_ratio >= 1.25:
                result["level"] = "H1"
                confidence += 0.5
                reasons.append("large_font")
            elif font_ratio >= 1.15:
                result["level"] = "H2"
                confidence += 0.4
                reasons.append("medium_font")
            else:
                result["level"] = "H3"
                confidence += 0.3
                reasons.append("small_font")
        
        # Additional confidence boosters
        if bold and result["level"]:
            confidence += 0.15
            reasons.append("bold")
        
        if text.isupper() and len(text) > 5 and len(text) < 100:
            confidence += 0.15
            reasons.append("all_caps")
        
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text) and result["level"]:
            confidence += 0.1
            reasons.append("title_case")
        
        # Length penalties and bonuses
        if len(text) > 150:
            confidence -= 0.3
            reasons.append("too_long")
        elif len(text) > 100:
            confidence -= 0.1
            reasons.append("long")
        elif 15 <= len(text) <= 80:
            confidence += 0.05
            reasons.append("good_length")
        elif len(text) < 5:
            confidence -= 0.2
            reasons.append("too_short")
        
        # Position-based confidence (headings often at top of sections)
        if block.get("y_relative", 0.5) < 0.1 and result["level"]:  # Top 10% of page
            confidence += 0.1
            reasons.append("top_position")
        
        # Final decision with higher threshold
        if result["level"] and confidence > 0.3:
            result["is_heading"] = True
            result["confidence"] = min(confidence, 1.0)
            result["reasons"] = reasons
        
        return result
    
    def ml_classification(self, block: Dict, median_font: float) -> Dict:
        """ML-based classification"""
        if self.ml_model is None or self.scaler is None:
            return {"is_heading": False, "level": None, "confidence": 0.0, "reasons": ["no_ml_model"]}
        
        try:
            # Extract features (same as training)
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
            
            # Ensure all required features are present and in correct order
            feature_array = np.array([[feature_vector.get(feat, 0) for feat in self.feature_names]])
            feature_array = np.nan_to_num(feature_array, nan=0.0)
            
            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Predict
            prediction = self.ml_model.predict(feature_array_scaled)[0]
            probabilities = self.ml_model.predict_proba(feature_array_scaled)[0]
            
            # Get confidence
            class_idx = list(self.ml_model.classes_).index(prediction)
            confidence = probabilities[class_idx]
            
            return {
                "is_heading": prediction != "non-heading",
                "level": prediction if prediction != "non-heading" else None,
                "confidence": float(confidence),
                "reasons": ["ml_prediction"],
                "all_probabilities": dict(zip(self.ml_model.classes_, probabilities))
            }
            
        except Exception as e:
            print(f"ML classification error: {e}")
            return {"is_heading": False, "level": None, "confidence": 0.0, "reasons": ["ml_error"]}
    
    def detect_title(self, blocks: List[Dict]) -> str:
        """Improved title detection"""
        if not blocks:
            return "Untitled Document"
        
        # Get all potential title candidates
        title_candidates = []
        for block in blocks:
            if self.is_title_candidate(block, blocks):
                title_candidates.append(block)
        
        if not title_candidates:
            # Fallback: look for largest font on first page that's not a form field
            first_page = [b for b in blocks if b["page"] == 1]
            if first_page:
                # Filter out likely form fields and table headers
                content_blocks = [b for b in first_page if not self.is_likely_table_header_or_form_field(b["text"])]
                if content_blocks:
                    max_size = max(b["font_size"] for b in content_blocks)
                    largest_blocks = [b for b in content_blocks if b["font_size"] >= max_size * 0.95]
                    # Choose the one that appears earliest on the page
                    largest_blocks.sort(key=lambda x: x.get("y_pos", 0))
                    if largest_blocks and 5 < len(largest_blocks[0]["text"]) < 200:
                        return largest_blocks[0]["text"]
            return "Untitled Document"
        
        # Sort candidates by font size (descending) and position (ascending)
        title_candidates.sort(key=lambda x: (-x["font_size"], x.get("y_pos", 0)))
        
        # Return the best candidate
        return title_candidates[0]["text"]
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """Main extraction method with duplicate removal"""
        try:
            blocks = self.extract_text_blocks(pdf_path)
            if not blocks:
                return {"title": "Empty Document", "outline": []}
            
            # Detect title
            title = self.detect_title(blocks)
            
            # Calculate median font size
            median_font = np.median([b["font_size"] for b in blocks])
            
            # Classify each block
            headings = []
            seen_texts = set()
            
            for block in blocks:
                text = block["text"].strip()
                
                # Skip duplicates, very short texts, and title repetition
                if (text in seen_texts or 
                    len(text) < 3 or 
                    text == title or
                    self.is_likely_table_header_or_form_field(text)):
                    continue
                
                # Get classifications
                heuristic_result = self.heuristic_classification(block, median_font, blocks)
                ml_result = self.ml_classification(block, median_font)
                
                # Decision logic: combine both approaches
                final_decision = self.combine_classifications(heuristic_result, ml_result)
                
                if final_decision["is_heading"]:
                    headings.append({
                        "level": final_decision["level"],
                        "text": text,
                        "page": block["page"],
                        "confidence": final_decision["confidence"],
                        "method": final_decision.get("method", "unknown"),
                        "y_pos": block.get("y_pos", 0)
                    })
                    seen_texts.add(text)
            
            # Sort headings by page and position
            headings.sort(key=lambda x: (x["page"], x.get("y_pos", 0)))
            
            # Additional post-processing: remove very similar headings
            filtered_headings = []
            for heading in headings:
                is_duplicate = False
                for existing in filtered_headings:
                    # Check for similar text (could be OCR artifacts)
                    if (abs(len(heading["text"]) - len(existing["text"])) < 5 and
                        heading["page"] == existing["page"] and
                        abs(heading.get("y_pos", 0) - existing.get("y_pos", 0)) < 20):
                        # Keep the one with higher confidence
                        if heading["confidence"] > existing["confidence"]:
                            filtered_headings.remove(existing)
                        else:
                            is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_headings.append(heading)
            
            # Remove confidence and method from final output
            clean_headings = [
                {
                    "level": h["level"],
                    "text": h["text"],
                    "page": h["page"]
                }
                for h in filtered_headings
            ]
            
            return {"title": title, "outline": clean_headings}
            
        except Exception as e:
            print(f"Error extracting outline from {pdf_path}: {e}")
            return {"title": "Error", "outline": []}
    
    def combine_classifications(self, heuristic_result: Dict, ml_result: Dict) -> Dict:
        """Combine heuristic and ML classifications with better logic"""
        
        # If no ML model, use heuristic only
        if self.ml_model is None:
            return {**heuristic_result, "method": "heuristic_only"}
        
        # Both say it's not a heading
        if not heuristic_result["is_heading"] and not ml_result["is_heading"]:
            return {
                "is_heading": False,
                "level": None,
                "confidence": 0.0,
                "method": "both_agree_no"
            }
        
        # Strong heuristic patterns override ML (numbered headings are very reliable)
        if (heuristic_result["is_heading"] and 
            any("numbered" in reason for reason in heuristic_result.get("reasons", [])) and
            heuristic_result["confidence"] > 0.8):
            return {**heuristic_result, "method": "heuristic_numbered_override"}
        
        # Both agree it's a heading
        if heuristic_result["is_heading"] and ml_result["is_heading"]:
            # Use ML if it has higher confidence, otherwise use heuristic
            if ml_result["confidence"] > heuristic_result["confidence"]:
                return {**ml_result, "method": "ml_higher_conf"}
            else:
                return {**heuristic_result, "method": "heuristic_higher_conf"}
        
        # Disagreement: use weighted confidence
        heuristic_weighted = heuristic_result["confidence"] * 0.4
        ml_weighted = ml_result["confidence"] * 0.6
        
        # Boost for strong patterns
        if any("numbered" in reason or "pattern" in reason 
               for reason in heuristic_result.get("reasons", [])):
            heuristic_weighted *= 1.3
        
        # Only accept if confidence is reasonably high
        min_confidence_threshold = 0.4
        
        if (heuristic_weighted > ml_weighted and 
            heuristic_result["is_heading"] and 
            heuristic_result["confidence"] > min_confidence_threshold):
            return {**heuristic_result, "method": "heuristic_wins"}
        elif (ml_result["is_heading"] and 
              ml_result["confidence"] > min_confidence_threshold):
            return {**ml_result, "method": "ml_wins"}
        else:
            return {
                "is_heading": False,
                "level": None,
                "confidence": 0.0,
                "method": "neither_confident"
            }


# Main processing function (same interface as your original)
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
            print(f"   Title: {result['title']}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            # Create error output
            error_result = {"title": "Error", "outline": []}
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    process()