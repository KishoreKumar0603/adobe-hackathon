from collections import defaultdict
from heading_classifier import HeadingClassifier
from context_utils import get_context, validate_hierarchy

def build_outline(spans, use_ml=True):
    font_stats = defaultdict(int)
    text_count = defaultdict(int)

    for span in spans:
        font_stats[span["size"]] += 1
        text_count[span["text"]] += 1

    # Identify top font sizes (largest = title)
    sorted_fonts = sorted(font_stats.items(), key=lambda x: -x[0])
    font_map = {}
    levels = ["Title", "H1", "H2", "H3"]

    for i, (size, _) in enumerate(sorted_fonts[:len(levels)]):
        font_map[size] = levels[i]

    candidates = []
    title = ""

    for idx, span in enumerate(spans):
        text = span["text"]
        if (
            len(text) < 3 or
            text_count[text] > 2 or
            span["y"] < 50 or span["y"] > 750 or
            any(x in text.lower() for x in ["page", "copyright", "date"])
        ):
            continue

        # Map span size to level (initial guess)
        level = font_map.get(span["size"], "non-heading")
        span["level"] = level

        # Save title if not already set
        if level == "Title" and not title:
            title = text
        else:
            candidates.append((idx, span))

    if use_ml:
        classifier = HeadingClassifier()
        refined_candidates = []

        for idx, span in candidates:
            context = get_context(spans, idx)
            label = classifier.classify_span(span, context)

            if label != "non-heading":
                refined_candidates.append({
                    "level": label,
                    "text": span["text"],
                    "page": span["page"],
                    "y": span["y"]
                })

        # Fix heading hierarchy (e.g., H1 followed by H3 → change to H2)
        outline = validate_hierarchy(refined_candidates)
    else:
        # Just use font-based candidates
        outline = [
            {"level": span["level"], "text": span["text"], "page": span["page"], "y": span["y"]}
            for _, span in candidates if span["level"] in {"H1", "H2", "H3"}
        ]

    # Final sort by page and vertical position
    sorted_outline = sorted(outline, key=lambda x: (x["page"], x["y"]))

    # Remove 'y' before returning
    return {
        "title": title,
        "outline": [
            {"level": item["level"], "text": item["text"], "page": item["page"]}
            for item in sorted_outline
        ]
    }
