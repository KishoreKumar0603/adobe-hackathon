from collections import defaultdict
from heading_classifier import HeadingClassifier

def build_outline(spans, use_ml=False):
    font_stats = defaultdict(int)
    text_count = defaultdict(int)
    for span in spans:
        font_stats[span["size"]] += 1
        text_count[span["text"]] += 1

    sorted_fonts = sorted(font_stats.items(), key=lambda x: -x[0])
    font_map = {}
    levels = ["Title", "H1", "H2", "H3"]
    for i, (size, _) in enumerate(sorted_fonts[:len(levels)]):
        font_map[size] = levels[i]

    candidates = []
    title = ""

    for span in spans:
        text = span["text"]
        if (
            len(text) < 3 or
            text_count[text] > 2 or
            span["y"] < 50 or span["y"] > 750 or
            any(x in text.lower() for x in ["page", "copyright", "date"])
        ):
            continue

        level = font_map.get(span["size"])
        if not level:
            continue

        if level == "Title" and not title:
            title = text
        elif level in {"H1", "H2", "H3"}:
            span["level"] = level
            candidates.append(span)

    if use_ml:
        scorer = HeadingClassifier()
        scored = scorer.score_candidates(candidates)
        # Sort by ML confidence
        candidates = sorted(
            [dict(c, score=s) for c, (_, s) in zip(candidates, scored)],
            key=lambda x: -x["score"]
        )

    # Final output
    outline = []
    for c in candidates:
        outline.append({
            "level": c["level"],
            "text": c["text"],
            "page": c["page"]
        })

    return {
        "title": title,
        "outline": outline
    }
