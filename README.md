# Adobe India Hackathon 2025: Connecting the Dots

This project is a solution for the "Connecting the Dots" hackathon, aiming to build an intelligent PDF experience. It contains implementations for both Round 1A (Document Structure Extraction) and Round 1B (Persona-Driven Intelligence).

## Project Structure

```
adobe-hackathon/
├── Dockerfile
├── README.md
├── requirements.txt
├── input/
├── output/
└── src/
    ├── main.py        # Main entry point
    ├── round_1a.py    # Logic for structure extraction
    └── round_1b.py    # Logic for persona-driven analysis
```

## Approach

### Round 1A: Understand Your Document

The goal is to extract a structured outline (Title, H1, H2, H3) from a PDF.

1.  **PDF Parsing**: The solution uses the `PyMuPDF` (fitz) library, which is extremely fast and provides detailed text block information, including font size, name, and weight.
2.  **Style Analysis**: A preliminary pass over the document identifies the hierarchy of font sizes. It heuristically determines the common body text size and then assigns larger font sizes to H1, H2, and H3 levels.
3.  **Heading Detection**: It iterates through text blocks, applying a set of rules to identify headings:
    *   **Font Size & Weight**: Must be larger than the body text and often bold.
    *   **Line Content**: Headings are typically short, single-line blocks.
    *   **Context**: The first large, bold text on the first page is assumed to be the title.
4.  **JSON Output**: The extracted information is formatted into the required JSON structure.

### Round 1B: Persona-Driven Document Intelligence

The goal is to extract and rank sections from multiple documents relevant to a given user persona and their task.

1.  **NLP Model**: The solution uses `spaCy` with the `en_core_web_md` model for all NLP tasks. This model provides a good balance between performance and size (< 1GB), and is baked into the Docker image for offline use.
2.  **Document Chunking**: Each PDF is broken down into semantic "sections". A section is defined as a heading and all the text that follows it, until the next heading.
3.  **Relevance Ranking**:
    *   A "query document" is created by combining the `persona` and `job_to_be_done` text.
    *   The semantic similarity (using cosine similarity on word vectors) is calculated between the query and each document section.
    *   Sections from all documents are ranked in descending order of this similarity score.
4.  **Sub-section Analysis**: For the top-ranked sections, the most relevant sentences are identified by calculating their individual similarity to the query. The top few sentences are extracted to form a "refined text" summary.
5.  **JSON Output**: The final ranked list and sub-section analysis are compiled into the required JSON format.

## How to Build and Run

The entire solution is containerized with Docker for easy and consistent execution.

### 1. Build the Docker Image

Navigate to the project's root directory (`adobe-hackathon/`) and run the build command. This will install all dependencies and download the NLP model.

```sh
docker build -t adobe-hackathon-solution .
```

### 2. Run the Solution

#### For Round 1A:

Place your input PDFs inside the `input/round1a/` directory.

```sh
docker run --rm \
  -v $(pwd)/input/round1a:/app/input \
  -v $(pwd)/output:/app/output \
  adobe-hackathon-solution \
  round1a --input_dir /app/input --output_dir /app/output
```
The output JSON files will appear in the `output/` directory on your host machine.

#### For Round 1B:

1. Place your input PDFs inside the `input/round1b/` directory.
2. Create a `config.json` file in `input/round1b/` with the persona, job, and document list. Example:

```json
{
  "persona": "PhD Researcher in Computational Biology",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for Graph Neural Networks in Drug Discovery.",
  "documents": [
    "doc1_gnn_intro.pdf",
    "doc2_gnn_methods.pdf"
  ]
}
```

3. Run the container:
```sh
docker run --rm \
  -v $(pwd)/input/round1b:/app/input \
  -v $(pwd)/output:/app/output \
  adobe-hackathon-solution \
  round1b --input_dir /app/input --output_dir /app/output --config_file config.json
```
The `challenge1b_output.json` file will appear in the `output/` directory.
