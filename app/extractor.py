import fitz
import unicodedata

def is_non_latin(text):
    for ch in text:
        if unicodedata.category(ch)[0] in ["L", "N"]:
            if 'LATIN' not in unicodedata.name(ch, ''):
                return True
    return False

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    title = doc.metadata.get('title', '') or 'Untitled Document'
    outline = []

    font_sizes = []
    font_counts = {}

    # Pass 1: Collect font statistics
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    font = span["font"]
                    font_sizes.append(size)
                    font_counts[font] = font_counts.get(font, 0) + 1

    avg_size = sum(font_sizes) / max(len(font_sizes), 1)

    # Pass 2: Heading detection using combined heuristics
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    size = round(span["size"], 1)
                    font_flags = span["flags"]  # bold = 2, italic = 1
                    font = span["font"]

                    bold = (font_flags & 2) != 0
                    unique_font = font_counts.get(font, 0) <= 2

                    if len(text) < 5 and not is_non_latin(text):
                        continue

                    # Combine factors:
                    importance = 0
                    if size >= avg_size:
                        importance += 1
                    if bold:
                        importance += 1
                    if unique_font:
                        importance += 1

                    level = None
                    if importance >= 3:
                        level = "H1"
                    elif importance == 2:
                        level = "H2"
                    elif importance == 1:
                        level = "H3"

                    if level:
                        outline.append({
                            "level": level,
                            "text": text,
                            "page": page_num
                        })

    return {"title": title, "outline": outline}
