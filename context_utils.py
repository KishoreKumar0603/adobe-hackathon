import numpy as np

def get_context(spans: list, current_idx: int, window_size: int = 2) -> str:
    """
    Extract surrounding text as context for better classification
    Args:
        spans: List of all text spans
        current_idx: Index of current span being classified
        window_size: Number of spans to consider before/after
    Returns:
        Context string with surrounding text
    """
    start = max(0, current_idx - window_size)
    end = min(len(spans), current_idx + window_size + 1)
    
    context_parts = []
    for i in range(start, end):
        if i != current_idx:
            context_parts.append(spans[i]['text'])
    
    return " [CONTEXT] ".join(context_parts)

def validate_hierarchy(outline: list) -> list:
    """
    Ensure proper heading hierarchy (H2 follows H1, etc.)
    Args:
        outline: Extracted outline with potential hierarchy issues
    Returns:
        Validated outline with fixed hierarchy
    """
    if not outline:
        return []
    
    validated = []
    last_level = 0
    
    for entry in outline:
        current_level = int(entry['level'][1:])
        
        # Fix hierarchy jumps (e.g., H1 directly to H3)
        if current_level > last_level + 1:
            current_level = last_level + 1
            entry['level'] = f"H{current_level}"
        
        validated.append(entry)
        last_level = current_level
    
    return validated
