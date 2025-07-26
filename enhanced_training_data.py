#!/usr/bin/env python3
"""
Enhanced Training Data Generator for Heading Classification

This script implements an improved approach for generating training data by comparing
PDF content with JSON files, with special handling for cases with empty outlines.
It includes advanced feature extraction, sophisticated matching algorithms, and
validation mechanisms to ensure high-quality training data.
"""

import os
import json
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import re
import logging
from typing import List, Dict, Tuple, Any, Optional, Set
import random
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HEADING_LEVELS = ["H1", "H2", "H3"]
NON_HEADING = "non-heading"
SIMILARITY_THRESHOLD = 0.75  # Default threshold, will be adjusted dynamically
MIN_HEADING_LENGTH = 2
MAX_HEADING_LENGTH = 100
HEADING_LEVEL_COLORS = {
    "H1": (1, 0.7, 0.7),  # Light red
    "H2": (0.7, 0.7, 1),  # Light blue
    "H3": (0.7, 1, 0.7),  # Light green
}


class EnhancedTrainingDataGenerator:
    """
    Enhanced generator for heading classification training data.
    """

    def __init__(self, input_dir: str, output_dir: str, output_path: str, 
                 validation_dir: str = None, confidence_threshold: float = 0.6):
        """
        Initialize the training data generator.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory containing JSON files
            output_path: Path to save the training data
            validation_dir: Directory to save validation files
            confidence_threshold: Minimum confidence score for automatic acceptance
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_path = output_path
        self.validation_dir = validation_dir or os.path.join(os.path.dirname(output_path), "validation")
        self.confidence_threshold = confidence_threshold
        
        # Create validation directory if it doesn't exist
        if not os.path.exists(self.validation_dir):
            os.makedirs(self.validation_dir, exist_ok=True)

    def build_dataset(self):
        """
        Build the training dataset from PDF and JSON files.
        """
        all_data = []
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".pdf")]
        total = 0
        
        for pdf_file in files:
            pdf_path = os.path.join(self.input_dir, pdf_file)
            json_path = os.path.join(self.output_dir, pdf_file.replace(".pdf", ".json"))
            
            if not os.path.exists(json_path):
                logger.warning(f"⚠️ Missing JSON for {pdf_file}, skipping.")
                continue
            
            logger.info(f"Processing {pdf_file}...")
            
            # Load JSON data
            with open(json_path, "r", encoding="utf-8") as f:
                truth = json.load(f)
            
            # Extract spans from PDF
            spans = self.extract_enhanced_spans(pdf_path)
            
            # Process the document
            document_data = self.process_document(spans, truth, pdf_path)
            all_data.extend(document_data)
            total += len(document_data)
        
        # Save the training data
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for entry in all_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"✅ Training data written to {self.output_path}")
        logger.info(f"📊 Total entries: {total}")
        
        # Validate the dataset
        self.validate_dataset(all_data)
        
        return all_data

    def extract_enhanced_spans(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract spans from PDF with enhanced features.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of spans with enhanced features
        """
        doc = fitz.open(pdf_path)
        spans = []
        
        # First pass: extract basic spans and document statistics
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        spans.append({
                            "text": text,
                            "size": round(span["size"], 1),
                            "bold": "bold" in span.get("font", "").lower(),
                            "italic": "italic" in span.get("font", "").lower(),
                            "font": span.get("font", ""),
                            "x": span["bbox"][0],
                            "y": span["bbox"][1],
                            "width": span["bbox"][2] - span["bbox"][0],
                            "height": span["bbox"][3] - span["bbox"][1],
                            "page": page.number + 1,
                            "block_idx": block.get("number", 0),
                            "line_idx": line.get("number", 0),
                            "span_idx": span.get("number", 0),
                            "color": span.get("color", 0),
                        })
        
        # Calculate document statistics
        if spans:
            # Font size statistics
            font_sizes = [span["size"] for span in spans]
            most_common_size = Counter(font_sizes).most_common(1)[0][0]
            
            # Second pass: add relative features
            for span in spans:
                # Add relative font size
                span["rel_size"] = span["size"] / most_common_size
                
                # Add capitalization features
                span["is_title_case"] = self.is_title_case(span["text"])
                span["is_upper_case"] = span["text"].isupper()
                
                # Add length features
                span["char_count"] = len(span["text"])
                span["word_count"] = len(span["text"].split())
                
                # Add position features
                span["rel_y"] = span["y"] / page.rect.height
                span["rel_x"] = span["x"] / page.rect.width
                
                # Add structural features
                span["has_number_prefix"] = bool(re.match(r'^\d+[.\s]', span["text"]))
                span["ends_with_punct"] = bool(re.search(r'[.!?:]$', span["text"]))
        
        return spans

    def process_document(self, spans: List[Dict[str, Any]], truth: Dict[str, Any], 
                         pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a document to generate training data.
        
        Args:
            spans: List of spans from the PDF
            truth: JSON data for the document
            pdf_path: Path to the PDF file
            
        Returns:
            List of training data entries
        """
        document_data = []
        outline = truth.get("outline", [])
        title = truth.get("title", "")
        
        # Determine processing approach based on outline availability
        if outline:
            logger.info(f"Using outline-based approach with {len(outline)} outline items")
            labeled_spans = self.process_with_outline(spans, outline)
        else:
            logger.info(f"Using heuristic approach with title: '{title}'")
            labeled_spans = self.process_without_outline(spans, title)
        
        # Convert to training data format
        for span, label, confidence in labeled_spans:
            features = (
                f"font_size:{span['size']} "
                f"rel_size:{span['rel_size']:.2f} "
                f"bold:{span['bold']} "
                f"italic:{span['italic']} "
                f"y_pos:{span['y']} "
                f"rel_y:{span['rel_y']:.2f} "
                f"page:{span['page']} "
                f"char_count:{span['char_count']} "
                f"word_count:{span['word_count']} "
                f"is_title_case:{span['is_title_case']} "
                f"is_upper_case:{span['is_upper_case']} "
                f"has_number_prefix:{span['has_number_prefix']} "
                f"ends_with_punct:{span['ends_with_punct']}"
            )
            
            document_data.append({
                "text": span["text"],
                "features": features,
                "label": label,
                "confidence": confidence,
                "page": span["page"],
                "y": span["y"],
                "manually_verified": False
            })
        
        # Generate validation files
        self.generate_validation_files(pdf_path, document_data)
        
        return document_data

    def process_with_outline(self, spans: List[Dict[str, Any]], 
                             outline: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str, float]]:
        """
        Process spans using the outline for matching.
        
        Args:
            spans: List of spans from the PDF
            outline: Outline data from the JSON
            
        Returns:
            List of (span, label, confidence) tuples
        """
        labeled_spans = []
        
        # Enhanced matching algorithm
        for span in spans:
            best_match = self.find_best_match(span, outline)
            if best_match:
                outline_item, similarity = best_match
                label = outline_item["level"]
                confidence = similarity
            else:
                label = NON_HEADING
                confidence = 1.0  # High confidence for non-headings when no match
            
            labeled_spans.append((span, label, confidence))
        
        return labeled_spans

    def process_without_outline(self, spans: List[Dict[str, Any]], 
                                title: str) -> List[Tuple[Dict[str, Any], str, float]]:
        """
        Process spans without using an outline.
        
        Args:
            spans: List of spans from the PDF
            title: Document title from the JSON
            
        Returns:
            List of (span, label, confidence) tuples
        """
        # Apply heuristic detection
        heuristic_headings = self.detect_headings_heuristic(spans, title)
        
        # Apply structure analysis
        structure_headings = self.analyze_document_structure(spans)
        
        # Combine results
        combined_headings = self.combine_heading_results([
            (heuristic_headings, 0.7),
            (structure_headings, 0.3)
        ])
        
        return combined_headings

    def detect_headings_heuristic(self, spans: List[Dict[str, Any]], 
                                  title: str) -> List[Tuple[Dict[str, Any], str, float]]:
        """
        Detect headings using heuristics.
        
        Args:
            spans: List of spans from the PDF
            title: Document title
            
        Returns:
            List of (span, label, confidence) tuples
        """
        # Get document statistics
        font_sizes = [span["size"] for span in spans]
        font_size_counts = Counter(font_sizes)
        common_font_size = font_size_counts.most_common(1)[0][0]
        
        # Sort font sizes in descending order for heading level assignment
        sorted_sizes = sorted(set(font_sizes), reverse=True)
        
        # Find title span if possible
        title_span = None
        if title:
            title_candidates = []
            for span in spans:
                similarity = SequenceMatcher(None, span["text"].lower(), title.lower()).ratio()
                if similarity > 0.8:
                    title_candidates.append((span, similarity))
            
            if title_candidates:
                title_span, _ = max(title_candidates, key=lambda x: x[1])
        
        # Score each span
        potential_headings = []
        for span in spans:
            score = 0
            
            # Font size analysis
            if span["size"] > common_font_size:
                score += (span["size"] / common_font_size - 1) * 10
                
            # Bold text analysis
            if span["bold"]:
                score += 5
                
            # Length analysis
            if MIN_HEADING_LENGTH < len(span["text"]) < MAX_HEADING_LENGTH:
                score += 3
                
            # Capitalization patterns
            if span["is_title_case"] or span["is_upper_case"]:
                score += 2
                
            # Structural features
            if span["has_number_prefix"]:
                score += 2
                
            if not span["ends_with_punct"]:
                score += 1
            
            # Title similarity if title_span exists
            if title_span and span != title_span:
                if abs(span["size"] - title_span["size"]) < 2:
                    score += 3
                if span["bold"] == title_span["bold"]:
                    score += 2
            
            # If score exceeds threshold, consider it a heading
            confidence = min(score / 20, 1.0)  # Normalize to [0, 1]
            if score > 5:
                potential_headings.append((span, score, confidence))
        
        # Sort by score and classify heading levels
        potential_headings.sort(key=lambda x: (-x[1], x[0]["page"], x[0]["y"]))
        
        # Assign heading levels based on font size and position
        labeled_spans = []
        for span, score, confidence in potential_headings:
            # Determine heading level based on font size
            if len(sorted_sizes) >= 3:
                if span["size"] >= sorted_sizes[0]:
                    label = "H1"
                elif span["size"] >= sorted_sizes[1]:
                    label = "H2"
                else:
                    label = "H3"
            else:
                # If we have limited font sizes, use relative size
                rel_size = span["rel_size"]
                if rel_size > 1.5:
                    label = "H1"
                elif rel_size > 1.2:
                    label = "H2"
                else:
                    label = "H3"
            
            labeled_spans.append((span, label, confidence))
        
        # Add non-headings
        # Create a list of span identifiers (text, page, y) that are headings
        heading_span_ids = {(span["text"], span["page"], span["y"]) for span, _, _ in labeled_spans}
        
        for span in spans:
            # Create a unique identifier for this span
            span_id = (span["text"], span["page"], span["y"])
            if span_id not in heading_span_ids:
                labeled_spans.append((span, NON_HEADING, 1.0))
        
        return labeled_spans

    def analyze_document_structure(self, spans: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str, float]]:
        """
        Analyze document structure to identify headings.
        
        Args:
            spans: List of spans from the PDF
            
        Returns:
            List of (span, label, confidence) tuples
        """
        # Group spans by page
        spans_by_page = defaultdict(list)
        for span in spans:
            spans_by_page[span["page"]].append(span)
        
        # Sort spans within each page by y-position
        for page, page_spans in spans_by_page.items():
            spans_by_page[page] = sorted(page_spans, key=lambda s: s["y"])
        
        # Identify potential section starts
        section_starts = []
        for page, page_spans in spans_by_page.items():
            for i, span in enumerate(page_spans):
                # Skip first span on page
                if i == 0:
                    continue
                
                # Calculate vertical gap with previous span
                prev_span = page_spans[i-1]
                gap = span["y"] - (prev_span["y"] + prev_span["height"])
                
                # Check if this could be a section start
                is_potential_start = (
                    gap > 10 and  # Significant gap
                    MIN_HEADING_LENGTH < len(span["text"]) < MAX_HEADING_LENGTH and  # Reasonable length
                    (span["bold"] or span["rel_size"] > 1.1)  # Emphasized text
                )
                
                if is_potential_start:
                    confidence = min((gap / 20) * (span["rel_size"] - 1) * 5, 1.0)
                    section_starts.append((span, confidence))
        
        # Cluster section starts by formatting
        if section_starts:
            # Extract features for clustering
            features = []
            for span, _ in section_starts:
                features.append([
                    span["size"],
                    1 if span["bold"] else 0,
                    1 if span["italic"] else 0,
                    span["rel_y"],
                    span["char_count"],
                    1 if span["is_title_case"] else 0,
                    1 if span["is_upper_case"] else 0,
                    1 if span["has_number_prefix"] else 0
                ])
            
            # Normalize features
            X = StandardScaler().fit_transform(features)
            
            # Cluster
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
            labels = clustering.labels_
            
            # Group by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  # Skip noise
                    span, confidence = section_starts[i]
                    clusters[label].append((span, confidence))
            
            # Assign heading levels based on clusters
            heading_levels = {}
            sorted_clusters = sorted(clusters.items(),
                                    key=lambda x: -np.mean([s["size"] for s, _ in x[1]]))
            
            for i, (cluster, items) in enumerate(sorted_clusters):
                if i < len(HEADING_LEVELS):
                    level = HEADING_LEVELS[i]
                else:
                    level = HEADING_LEVELS[-1]
                
                for span, confidence in items:
                    # Use a tuple of identifying properties as the key
                    span_key = (span["text"], span["page"], span["y"])
                    heading_levels[span_key] = (level, confidence, span)
            
            # Create labeled spans
            labeled_spans = []
            for span in spans:
                span_key = (span["text"], span["page"], span["y"])
                if span_key in heading_levels:
                    level, confidence, _ = heading_levels[span_key]
                    labeled_spans.append((span, level, confidence))
                else:
                    labeled_spans.append((span, NON_HEADING, 1.0))
            
            return labeled_spans
        
        # Fallback if clustering didn't work
        return [(span, NON_HEADING, 1.0) for span in spans]

    def combine_heading_results(self, results_with_weights: List[Tuple[List[Tuple[Dict[str, Any], str, float]], float]]) -> List[Tuple[Dict[str, Any], str, float]]:
        """
        Combine heading detection results from multiple methods.
        
        Args:
            results_with_weights: List of (results, weight) tuples
            
        Returns:
            Combined list of (span, label, confidence) tuples
        """
        if not results_with_weights:
            return []
        
        # Create a mapping of span identifiers to weighted votes
        span_votes = defaultdict(lambda: defaultdict(float))
        
        for results, weight in results_with_weights:
            for span, label, confidence in results:
                span_key = (span["text"], span["page"], span["y"])
                span_votes[span_key][label] += confidence * weight
        
        # Determine final label for each span
        final_results = []
        all_spans = {}  # Use a dictionary instead of a set
        
        for results, _ in results_with_weights:
            for span, _, _ in results:
                span_key = (span["text"], span["page"], span["y"])
                all_spans[span_key] = span  # Store the span with its key
        
        for span_key, span in all_spans.items():
            votes = span_votes[span_key]
            
            if votes:
                # Get label with highest weighted vote
                final_label = max(votes.items(), key=lambda x: x[1])[0]
                final_confidence = votes[final_label] / sum(votes.values())
            else:
                final_label = NON_HEADING
                final_confidence = 1.0
            
            final_results.append((span, final_label, final_confidence))
        
        return final_results

    def find_best_match(self, span: Dict[str, Any], outline: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Find the best matching outline item for a span.
        
        Args:
            span: Span from the PDF
            outline: Outline data from the JSON
            
        Returns:
            Tuple of (outline_item, similarity) or None if no good match
        """
        best_score = 0
        best_item = None
        
        # Filter outline items by page
        page_outline = [item for item in outline if item["page"] == span["page"]]
        
        for item in page_outline:
            # Basic text similarity
            text_similarity = SequenceMatcher(None, span["text"].lower(), item["text"].lower()).ratio()
            
            # Enhanced similarity with additional features
            feature_bonus = 0
            
            # Position similarity if y-coordinate is available
            if "y" in item and abs(span["y"] - item["y"]) < 20:
                feature_bonus += 0.1
            
            # Heading characteristics bonus
            if span["bold"] and len(span["text"]) < MAX_HEADING_LENGTH:
                feature_bonus += 0.05
            
            if span["is_title_case"] or span["is_upper_case"]:
                feature_bonus += 0.05
            
            # Combine similarities
            combined_score = text_similarity + feature_bonus
            
            if combined_score > best_score:
                best_score = combined_score
                best_item = item
        
        # Dynamic threshold based on span characteristics
        threshold = SIMILARITY_THRESHOLD
        
        # Adjust threshold based on span properties
        if span["bold"] and span["rel_size"] > 1.2:
            threshold -= 0.1  # Lower threshold for likely headings
        
        if span["char_count"] > 50:
            threshold += 0.1  # Higher threshold for long text
        
        if best_score > threshold:
            return best_item, best_score
        
        return None

    def generate_validation_files(self, pdf_path: str, document_data: List[Dict[str, Any]]):
        """
        Generate validation files for the document.
        
        Args:
            pdf_path: Path to the PDF file
            document_data: Training data for the document
        """
        base_name = os.path.basename(pdf_path).replace(".pdf", "")
        
        # Generate annotated PDF
        annotated_pdf_path = os.path.join(self.validation_dir, f"{base_name}_annotated.pdf")
        self.generate_annotated_pdf(pdf_path, document_data, annotated_pdf_path)
        
        # Generate validation report
        report_path = os.path.join(self.validation_dir, f"{base_name}_validation.json")
        self.generate_validation_report(document_data, report_path)
        
        logger.info(f"✅ Validation files generated in {self.validation_dir}")

    def generate_annotated_pdf(self, pdf_path: str, document_data: List[Dict[str, Any]], 
                               output_path: str):
        """
        Generate an annotated PDF with highlighted headings.
        
        Args:
            pdf_path: Path to the PDF file
            document_data: Training data for the document
            output_path: Path to save the annotated PDF
        """
        # Load the PDF
        doc = fitz.open(pdf_path)
        
        # Create a copy for annotation
        annotated_pdf = fitz.open()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            annotated_page = annotated_pdf.new_page(width=page.rect.width, height=page.rect.height)
            
            # Copy original content
            annotated_page.show_pdf_page(annotated_page.rect, doc, page_num)
            
            # Add annotations for detected headings
            page_entries = [e for e in document_data if e["page"] == page_num + 1]
            
            for entry in page_entries:
                if entry["label"] != NON_HEADING:
                    # Find text on page
                    text_instances = page.search_for(entry["text"])
                    
                    if text_instances:
                        for rect in text_instances:
                            # Create colored highlight based on heading level
                            color = HEADING_LEVEL_COLORS.get(entry["label"], (0.7, 0.7, 0.7))
                            
                            # Create highlight annotation
                            highlight = annotated_page.add_highlight_annot(rect)
                            
                            # Set the color after creation
                            highlight.set_colors(stroke=color)
                            highlight.update()
                            
                            # Add label annotation
                            text_annot = annotated_page.add_text_annot(
                                rect.top_right,
                                f"Level: {entry['label']}, Conf: {entry['confidence']:.2f}"
                            )
                            text_annot.update()
        
        # Save the annotated PDF
        annotated_pdf.save(output_path)

    def generate_validation_report(self, document_data: List[Dict[str, Any]], output_path: str):
        """
        Generate a validation report for the document.
        
        Args:
            document_data: Training data for the document
            output_path: Path to save the validation report
        """
        # Count labels
        label_counts = Counter([entry["label"] for entry in document_data])
        
        # Count low confidence entries
        low_confidence = [
            entry for entry in document_data 
            if entry["confidence"] < self.confidence_threshold
        ]
        
        # Generate report
        report = {
            "label_counts": dict(label_counts),
            "total_entries": len(document_data),
            "low_confidence_count": len(low_confidence),
            "low_confidence_entries": [
                {
                    "text": entry["text"],
                    "label": entry["label"],
                    "confidence": entry["confidence"],
                    "page": entry["page"],
                    "y": entry["y"]
                }
                for entry in low_confidence
            ]
        }
        
        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def validate_dataset(self, all_data: List[Dict[str, Any]]):
        """
        Validate the entire dataset.
        
        Args:
            all_data: Complete training dataset
        """
        # Check for duplicate entries
        texts = [entry["text"] for entry in all_data]
        text_counts = Counter(texts)
        duplicates = {text: count for text, count in text_counts.items() if count > 1}
        
        if duplicates:
            logger.warning(f"⚠️ Found {len(duplicates)} duplicate text entries")
        
        # Check label distribution
        label_counts = Counter([entry["label"] for entry in all_data])
        logger.info(f"📊 Label distribution: {dict(label_counts)}")
        
        # Check confidence distribution
        confidence_values = [entry["confidence"] for entry in all_data]
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        low_confidence = sum(1 for c in confidence_values if c < self.confidence_threshold)
        
        logger.info(f"📊 Average confidence: {avg_confidence:.2f}")
        logger.info(f"📊 Low confidence entries: {low_confidence} ({low_confidence/len(all_data)*100:.1f}%)")
        
        # Generate overall validation report
        report_path = os.path.join(self.validation_dir, "dataset_validation.json")
        
        report = {
            "total_entries": len(all_data),
            "label_distribution": dict(label_counts),
            "average_confidence": avg_confidence,
            "low_confidence_count": low_confidence,
            "low_confidence_percentage": low_confidence/len(all_data)*100 if all_data else 0,
            "duplicate_count": len(duplicates)
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Overall validation report saved to {report_path}")

    @staticmethod
    def is_title_case(text: str) -> bool:
        """Check if text is in title case."""
        if not text:
            return False
        
        words = text.split()
        if not words:
            return False
        
        # Check if first word starts with uppercase
        if not words[0][0].isupper():
            return False
        
        # Check if at least 50% of words start with uppercase
        uppercase_count = sum(1 for word in words if word and word[0].isupper())
        return uppercase_count / len(words) >= 0.5


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Generate enhanced training data for heading classification")
    parser.add_argument("--input-dir", default="input", help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="output", help="Directory containing JSON files")
    parser.add_argument("--output-path", default="dataset/train_data.jsonl", help="Path to save the training data")
    parser.add_argument("--validation-dir", default=None, help="Directory to save validation files")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="Minimum confidence score for automatic acceptance")
    
    args = parser.parse_args()
    
    generator = EnhancedTrainingDataGenerator(
        args.input_dir,
        args.output_dir,
        args.output_path,
        args.validation_dir,
        args.confidence_threshold
    )
    
    generator.build_dataset()


if __name__ == "__main__":
    main()