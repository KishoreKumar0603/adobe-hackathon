import fitz  # PyMuPDF
from collections import defaultdict

def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    font_stats = defaultdict(int)
    text_count = defaultdict(int)
    all_spans = []

    # Step 1: Collect font stats and spans with metadata
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    text = span["text"].strip()
                    y = span["origin"][1]
                    is_bold = "Bold" in span.get("font", "")
                    if text:
                        font_stats[size] += 1
                        text_count[text] += 1
                        all_spans.append({
                            "text": text,
                            "size": size,
                            "y": y,
                            "bold": is_bold,
                            "font": span.get("font", ""),
                            "page": page.number + 1
                        })

    # Step 2: Assign heading levels based on font size
    sorted_sizes = sorted(font_stats.items(), key=lambda x: (-x[0]))
    font_map = {}
    if sorted_sizes:
        font_map[sorted_sizes[0][0]] = "Title"
    if len(sorted_sizes) > 1:
        font_map[sorted_sizes[1][0]] = "H1"
    if len(sorted_sizes) > 2:
        font_map[sorted_sizes[2][0]] = "H2"
    if len(sorted_sizes) > 3:
        font_map[sorted_sizes[3][0]] = "H3"

    outline = []
    title = ""

    # Step 3: Use filters to find clean headings
    for span in all_spans:
        text = span["text"]
        size = span["size"]
        y = span["y"]
        level = font_map.get(size)
        page = span["page"]
        is_bold = span["bold"]

        # Heuristics to skip false positives
        if len(text) < 3:
            continue
        if text_count[text] > 2:
            continue  # likely a header/footer
        if y < 50 or y > 750:
            continue  # skip top/bottom of page
        if any(x in text.lower() for x in ["page", "copyright", "generated", "date"]):
            continue
        if not is_bold and level != "Title":
            continue

        if level == "Title" and not title:
            title = text
        elif level in {"H1", "H2", "H3"}:
            outline.append({
                "level": level,
                "text": text,
                "page": page
            })

    return {
        "title": title,
        "outline": outline
    }
