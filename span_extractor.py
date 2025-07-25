# span_extractor.py

import fitz  # PyMuPDF

def extract_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    spans.append({
                        "text": text,
                        "size": round(span["size"], 1),
                        "bold": "bold" in span.get("font", "").lower(),
                        "x": span["bbox"][0],
                        "y": span["bbox"][1],
                        "page": page.number + 1,
                        "font": span.get("font", "")
                    })

    return spans
