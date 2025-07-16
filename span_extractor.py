import fitz  # PyMuPDF

def extract_spans(pdf_path):
    doc = fitz.open(pdf_path)
    spans = []

    for page in doc:
        page_number = page.number + 1
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = round(span["size"], 1)
                    y = span["origin"][1]
                    font = span.get("font", "")
                    bold = "Bold" in font or "bold" in font.lower()
                    spans.append({
                        "text": text,
                        "size": size,
                        "bold": bold,
                        "y": y,
                        "font": font,
                        "page": page_number
                    })
    return spans
