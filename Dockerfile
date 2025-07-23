# Dockerfile

# Use a stable, slim Python base image compatible with amd64
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install system dependencies needed for PyMuPDF and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model so it's baked into the image (for offline use)
# en_core_web_md is a good balance of size and performance (< 1GB constraint)
RUN python -m spacy download en_core_web_md

# Copy the rest of the application source code
COPY ./src ./src

# Set the entry point to run the main script
# The arguments (round1a/round1b, --input, --output) will be passed in the `docker run` command
ENTRYPOINT ["python", "src/main.py"]
