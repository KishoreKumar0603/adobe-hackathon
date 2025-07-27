import json, re
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np

class PDFOutlineExtractor:
    def extract_text_blocks(self, pdf_path):
        doc = fitz.open(pdf_path)
        blocks = []
        for page_num in range(len(doc)):
            for b in doc[page_num].get_text("dict")["blocks"]:
                if "lines" not in b: continue
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            blocks.append({
                                "text": text,
                                "page": page_num + 1,
                                "font_size": span["size"],
                                "font_flags": span["flags"],
                                "font": span["font"]
                            })
        return blocks

    def detect_title(self, blocks):
        first_page = [b for b in blocks if b["page"] == 1]
        if not first_page: return "Untitled Document"
        max_size = max(b["font_size"] for b in first_page)
        title_candidates = [b for b in first_page if b["font_size"] >= max_size * 0.95]
        for b in title_candidates:
            if 5 < len(b["text"]) < 200: return b["text"]
        return "Untitled Document"

    def is_heading_candidate(self, block, median_font):
        text = block["text"]
        if block["font_size"] < median_font * 1.05: return False
        if re.match(r'^(\d+[\.\d]*)\s+.+', text): return True
        if text.isupper() and len(text) > 5: return True
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text): return True
        return False

    def classify_level(self, text):
        if re.match(r'^\d+\.\d+\.\d+', text): return "H3"
        elif re.match(r'^\d+\.\d+', text): return "H2"
        elif re.match(r'^\d+', text): return "H1"
        return None

    def extract_outline(self, pdf_path):
        blocks = self.extract_text_blocks(pdf_path)
        if not blocks:
            return {"title": "Empty", "outline": []}
        title = self.detect_title(blocks)
        median_font = np.median([b["font_size"] for b in blocks])
        headings = []
        seen = set()
        for b in blocks:
            text = b["text"].strip()
            if text in seen: continue
            if self.is_heading_candidate(b, median_font):
                level = self.classify_level(text)
                if not level:
                    fs = b["font_size"]
                    p75, p85, p95 = np.percentile([x["font_size"] for x in blocks], [75, 85, 95])
                    if fs >= p95: level = "H1"
                    elif fs >= p85: level = "H2"
                    elif fs >= p75: level = "H3"
                if level:
                    headings.append({
                        "level": level,
                        "text": text,
                        "page": b["page"]
                    })
                    seen.add(text)
        return {"title": title, "outline": headings}

def process():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(exist_ok=True)
    extractor = PDFOutlineExtractor()
    for pdf_file in input_dir.glob("*.pdf"):
        try:
            result = extractor.extract_outline(str(pdf_file))
        except Exception:
            result = {"title": "Error", "outline": []}
        with open(output_dir / f"{pdf_file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process()
