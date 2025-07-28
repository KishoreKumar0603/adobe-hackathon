import json
import re
import numpy as np
from pathlib import Path
import joblib
import fitz  # PyMuPDF
from typing import List, Dict, Optional

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
                    
                    if line_text.strip() and len(line_text.strip()) >= 3:
                        all_blocks.append({
                            "text": line_text.strip(),
                            "page": page_num + 1,
                            "font_size": max(line_features["font_sizes"]) if line_features["font_sizes"] else 12,
                            "bold": any(line_features["bold_flags"]),
                            "y_pos": min(line_features["y_positions"]) if line_features["y_positions"] else 0,
                            "x_start": line_features["x_start"] if line_features["x_start"] != float('inf') else 0,
                            "width": line_features["x_end"] - line_features["x_start"] if line_features["x_start"] != float('inf') else 0
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
    
    def heuristic_classification(self, block: Dict, median_font: float) -> Dict:
        """Enhanced heuristic classification"""
        text = block["text"]
        font_size = block["font_size"]
        bold = block.get("bold", False)
        
        result = {
            "is_heading": False,
            "level": None,
            "confidence": 0.0,
            "reasons": []
        }
        
        # Font size check
        font_ratio = font_size / median_font
        if font_ratio < 1.02:  # Very lenient threshold
            return result
        
        confidence = 0.0
        reasons = []
        
        # Strong numbering patterns
        if re.match(r'^\d+\.\d+\.\d+\.?\s', text):  # 1.1.1
            result["level"] = "H3"
            confidence += 0.9
            reasons.append("numbered_h3")
        elif re.match(r'^\d+\.\d+\.?\s', text):  # 1.1
            result["level"] = "H2"
            confidence += 0.9
            reasons.append("numbered_h2")
        elif re.match(r'^\d+\.?\s', text):  # 1.
            result["level"] = "H1"
            confidence += 0.9
            reasons.append("numbered_h1")
        
        # Common heading patterns
        if not result["level"]:
            heading_indicators = [
                (r'^(table of contents|contents)$', 'H1', 0.8),
                (r'^(introduction|conclusion|summary|overview)$', 'H1', 0.7),
                (r'^(chapter|section)\s+\d+', 'H1', 0.8),
                (r'^(background|methodology|results|discussion)$', 'H1', 0.6),
                (r'^(abstract|references|appendix|bibliography)$', 'H1', 0.7),
                (r'^revision\s+history$', 'H1', 0.8),
            ]
            
            text_lower = text.lower().strip()
            for pattern, level, conf in heading_indicators:
                if re.match(pattern, text_lower):
                    result["level"] = level
                    confidence += conf
                    reasons.append(f"pattern_{pattern}")
                    break
        
        # Font-based classification
        if not result["level"]:
            if font_ratio >= 1.5:
                result["level"] = "H1"
                confidence += 0.6
                reasons.append("very_large_font")
            elif font_ratio >= 1.3:
                result["level"] = "H1"
                confidence += 0.5
                reasons.append("large_font")
            elif font_ratio >= 1.2:
                result["level"] = "H2"
                confidence += 0.4
                reasons.append("medium_font")
            elif font_ratio >= 1.1:
                result["level"] = "H3"
                confidence += 0.3
                reasons.append("small_font")
        
        # Additional confidence boosters
        if bold:
            confidence += 0.2
            reasons.append("bold")
        
        if text.isupper() and len(text) > 5:
            confidence += 0.2
            reasons.append("all_caps")
        
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text):
            confidence += 0.15
            reasons.append("title_case")
        
        # Length penalty for very long text
        if len(text) > 200:
            confidence -= 0.4
            reasons.append("too_long")
        elif 10 <= len(text) <= 80:
            confidence += 0.1
            reasons.append("good_length")
        
        # Final decision
        if result["level"] and confidence > 0.2:
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
        """Detect document title"""
        first_page = [b for b in blocks if b["page"] == 1]
        if not first_page:
            return "Untitled Document"
        
        # Look for largest font on first page
        max_size = max(b["font_size"] for b in first_page)
        title_candidates = [b for b in first_page if b["font_size"] >= max_size * 0.95]
        
        for b in title_candidates:
            if 5 < len(b["text"]) < 200 and not re.match(r'^\d+\.', b["text"]):
                return b["text"]
        
        return "Untitled Document"
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """Main extraction method"""
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
                
                # Skip duplicates and very short texts
                if text in seen_texts or len(text) < 3:
                    continue
                
                # Get classifications
                heuristic_result = self.heuristic_classification(block, median_font)
                ml_result = self.ml_classification(block, median_font)
                
                # Decision logic: combine both approaches
                final_decision = self.combine_classifications(heuristic_result, ml_result)
                
                if final_decision["is_heading"]:
                    headings.append({
                        "level": final_decision["level"],
                        "text": text,
                        "page": block["page"],
                        "confidence": final_decision["confidence"],
                        "method": final_decision.get("method", "unknown")
                    })
                    seen_texts.add(text)
            
            # Sort headings by page and position
            headings.sort(key=lambda x: (x["page"], x.get("y_pos", 0)))
            
            # Remove confidence and method from final output (keeping it clean for submission)
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
            return {"title": "Error", "outline": []}
    
    def combine_classifications(self, heuristic_result: Dict, ml_result: Dict) -> Dict:
        """Combine heuristic and ML classifications"""
        
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
        
        # Both agree it's a heading
        if heuristic_result["is_heading"] and ml_result["is_heading"]:
            # Use the one with higher confidence, but prefer heuristic for numbered patterns
            if any("numbered" in reason for reason in heuristic_result.get("reasons", [])):
                return {**heuristic_result, "method": "heuristic_numbered"}
            elif ml_result["confidence"] > heuristic_result["confidence"]:
                return {**ml_result, "method": "ml_higher_conf"}
            else:
                return {**heuristic_result, "method": "heuristic_higher_conf"}
        
        # Disagreement: use confidence-based decision with weights
        heuristic_weighted = heuristic_result["confidence"] * 0.4
        ml_weighted = ml_result["confidence"] * 0.6
        
        # Special boost for strong heuristic patterns
        if any("numbered" in reason for reason in heuristic_result.get("reasons", [])):
            heuristic_weighted *= 1.5
        
        if heuristic_weighted > ml_weighted and heuristic_result["is_heading"]:
            return {**heuristic_result, "method": "heuristic_wins"}
        elif ml_result["is_heading"]:
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
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            # Create error output
            error_result = {"title": "Error", "outline": []}
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    process()