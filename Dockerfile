# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - poppler-utils: Required for pdf2image (PDF to image conversion)
# - libgl1 and libglib2.0-0: Required for OpenCV (used by ultralytics/YOLO)
# - libsm6, libxext6, libxrender-dev: Additional X11 libs for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY callables.py .
COPY medical_report_processor.py .
COPY llm_logger.py .

# Copy the fine-tuned YOLO model
COPY fine_tune_yolo/runs/detect/train4/weights/best.pt ./fine_tune_yolo/runs/detect/train4/weights/

# Create directories for output and logs
RUN mkdir -p /app/output /app/logs

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
