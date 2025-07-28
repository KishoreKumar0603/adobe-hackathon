# Use official Python image
FROM --platform=linux/amd64 python:3.10-slim

FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and model files
COPY . .

# Expose output directory as a volume (optional)
VOLUME ["/app/output"]

# Set default command to run the extractor
CMD ["python", "hybrid_extractor.py"]