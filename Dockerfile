FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY data/ ./data/
COPY models/ ./models/
COPY training/ ./training/
COPY main.py .

# Default command
CMD ["python", "main.py", "--epochs", "5", "--batch_size", "64"]
