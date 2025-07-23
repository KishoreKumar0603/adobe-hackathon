import fitz  # PyMuPDF
import json
import os
import re
from collections import Counter

def get_font_styles(doc):
    """Analyzes the document to find common font sizes and styles."""
    styles = {}
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        font_size = round(s["size"])
                        font_name = s["font"]
                        styles[font_size] = styles.get(font_size, []) + [font_name]
    
    # Count font occurrences to find the most common (body text)
    font_counts = {size: Counter(names) for size, names in styles.items()}
    sorted_sizes = sorted(font_counts.keys(), reverse=True)
    
    if not sorted_sizes:
        return 12, "H1", "H2", "H3" # Default fallback

    # Heuristic: Body text is the most frequent font size
    body_size = max(font_counts, key=lambda s: len(font_counts[s]))
    
    # Heuristic: Headings are larger than body text
    heading_sizes = [s for s in sorted_sizes if s > body_size]
    
    # Assign H1, H2, H3 based on descending size
    h1_size = heading_sizes[0] if len(heading_sizes) > 0 else body_size + 2
    h2_size = heading_sizes[1] if len(heading_sizes) > 1 else body_size + 1
    h3_size = heading_sizes[2] if len(heading_sizes) > 2 else body_size

    return body_size, h1_size, h2_size, h3_size

def is_bold(font_name):
    return any(x in font_name.lower() for x in ["bold", "black", "heavy"])

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def process_round_1a(pdf_path):
    """
    Extracts title and H1, H2, H3 headings from a PDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"error": f"Could not open or process PDF: {e}"}

    body_size, h1_size, h2_size, h3_size = get_font_styles(doc)
    
    output = {"title": "", "outline": []}
    
    # Title Heuristic: First large, bold text on the first page
    title_found = False
    first_page = doc[0]
    blocks = sorted(first_page.get_text("dict")["blocks"], key=lambda b: b['bbox'][1])
    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    font_sz = round(s["size"])
                    if font_sz >= h1_size and is_bold(s["font"]):
                        output["title"] = clean_text(s["text"])
                        title_found = True
                        break
                if title_found: break
            if title_found: break
    
    if not output["title"] and doc.metadata['title']:
         output["title"] = doc.metadata['title']


    # Heading Extraction
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    # A heading is usually a single span in a line
                    if len(l["spans"]) == 1:
                        span = l["spans"][0]
                        text = clean_text(span["text"])
                        font_sz = round(span["size"])

                        level = None
                        # Check font size against our determined heading sizes
                        if abs(font_sz - h1_size) < 1 and is_bold(span["font"]):
                            level = "H1"
                        elif abs(font_sz - h2_size) < 1:
                            level = "H2"
                        elif abs(font_sz - h3_size) < 1:
                            level = "H3"
                        
                        # Fallback for documents that don't follow size hierarchy well
                        elif font_sz > body_size and is_bold(span["font"]) and not level:
                            if font_sz >= (body_size + 3): level = "H1"
                            elif font_sz >= (body_size + 1.5): level = "H2"
                            else: level = "H3"
                            
                        # Rule: Headings are short and not empty
                        if level and 2 < len(text) < 150:
                            output["outline"].append({
                                "level": level,
                                "text": text,
                                "page": page_num + 1
                            })

    doc.close()
    return output
