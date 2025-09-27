# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (keep minimal; add as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and assets
COPY app.py /app/app.py
COPY chatbotQA.py /app/chatbotQA.py
COPY database.py /app/database.py
COPY templates /app/templates
COPY static /app/static
COPY data /app/data

# Optional: persist databases outside the container
VOLUME ["/app/data", "/app"]

# Expose Flask port
EXPOSE 5000

# The app needs GEMINI_API_KEY at runtime; pass it via -e or docker-compose
# Default command
CMD ["python", "app.py"]
