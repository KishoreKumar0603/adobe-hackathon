# src/round_1b.py

import fitz  # PyMuPDF
import spacy
import json
import time
from collections import defaultdict

# Load the spaCy model. This will be pre-downloaded in the Docker container.
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Model 'en_core_web_md' not found. Please run 'python -m spacy download en_core_web_md'")
    nlp = spacy.load("en_core_web_sm") # Fallback to small model
print("spaCy model loaded.")

def get_document_sections(pdf_path):
    """Splits a PDF into sections based on H1/H2 headings."""
    doc = fitz.open(pdf_path)
    sections = []
    current_section_text = ""
    current_section_title = "Introduction" # Default for text before first heading
    current_page = 1

    # Using a simplified heading detection for section splitting
    # A more robust version would reuse logic from round_1a
    for page_num, page in enumerate(doc):
        blocks = sorted(page.get_text("blocks"), key=lambda b: b[1]) # Sort by vertical position
        for block in blocks:
            block_text = block[4]
            # Simple heuristic: A short line in all caps or ending without a period might be a heading.
            lines = block_text.strip().split('\n')
            is_heading = (len(lines) == 1 and len(lines[0].split()) < 10 and not lines[0].endswith('.'))

            if is_heading:
                # Save the previous section
                if current_section_text.strip():
                    sections.append({
                        "title": current_section_title,
                        "text": current_section_text.strip(),
                        "page": current_page
                    })
                # Start a new section
                current_section_title = lines[0]
                current_page = page_num + 1
                current_section_text = ""
            else:
                current_section_text += block_text + "\n"
    
    # Add the last section
    if current_section_text.strip():
        sections.append({
            "title": current_section_title,
            "text": current_section_text.strip(),
            "page": current_page
        })

    doc.close()
    return sections

def rank_sections(sections_by_doc, persona, job_to_be_done):
    """Ranks sections from all documents based on semantic similarity to the query."""
    query = f"{persona}. {job_to_be_done}"
    query_doc = nlp(query)
    
    ranked_sections = []
    
    for doc_name, sections in sections_by_doc.items():
        for section in sections:
            section_doc = nlp(section["text"][:nlp.max_length]) # Truncate to model's max length
            similarity = query_doc.similarity(section_doc)
            
            ranked_sections.append({
                "document": doc_name,
                "page_number": section["page"],
                "section_title": section["title"],
                "importance_rank": similarity,
                "full_text": section["text"] # Keep full text for sub-section analysis
            })
            
    # Sort by importance_rank (similarity score)
    ranked_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
    return ranked_sections

def get_sub_section_analysis(section_text, query_doc):
    """Finds the most relevant sentences within a section."""
    section_doc = nlp(section_text)
    
    sentences = []
    for sent in section_doc.sents:
        if sent.text.strip():
            similarity = query_doc.similarity(sent)
            sentences.append({"text": sent.text, "score": similarity})
            
    # Get top 3 sentences or all sentences with score > 0.6
    sentences.sort(key=lambda x: x["score"], reverse=True)
    top_sentences = [s["text"] for s in sentences if s["score"] > 0.6][:3]
    
    if not top_sentences and sentences: # if no sentence scores high, take the best one
        top_sentences = [sentences[0]["text"]]
        
    return " ".join(s.strip() for s in top_sentences)


def process_round_1b(pdf_paths, persona, job_to_be_done):
    """Main logic for Round 1B."""
    start_time = time.time()
    
    # 1. Extract sections from all documents
    all_sections = {}
    for pdf_path in pdf_paths:
        doc_name = pdf_path.split('/')[-1]
        all_sections[doc_name] = get_document_sections(pdf_path)

    # 2. Rank sections
    query_doc = nlp(f"{persona}. {job_to_be_done}")
    ranked_sections = rank_sections(all_sections, persona, job_to_be_done)
    
    # 3. Perform sub-section analysis on top sections
    sub_section_results = []
    for section in ranked_sections[:5]: # Analyze top 5 sections
        refined_text = get_sub_section_analysis(section["full_text"], query_doc)
        sub_section_results.append({
            "document": section["document"],
            "page_number": section["page_number"],
            "refined_text": refined_text
        })

    # 4. Format the final output
    output = {
        "metadata": {
            "input_documents": [p.split('/')[-1] for p in pdf_paths],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "extracted_sections": [
            {k: v for k, v in sec.items() if k != 'full_text'} for sec in ranked_sections
        ],
        "sub_section_analysis": sub_section_results
    }
    
    print(f"Round 1B processing took {time.time() - start_time:.2f} seconds.")
    return output

