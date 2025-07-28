# PDF Outline Extractor (Rule-Based)

## Overview
Extracts title and heading structure from PDFs using font size rules.

## Usage

### Build Docker Image
```
docker build --platform linux/amd64 -t pdf_extractor_rule .
```

### Run Container
```
docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output --network none pdf_extractor_rule
```

## Output
JSON files in `/output` with extracted structure (title, H1/H2/H3 headings and page numbers).
