FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY outline_extractor.py .

RUN mkdir -p /app/input /app/output

CMD ["python", "outline_extractor.py"]




