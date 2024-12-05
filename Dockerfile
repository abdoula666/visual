FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ResNet50 weights
RUN python -c "from tensorflow.keras.applications import ResNet50; ResNet50(weights='imagenet', include_top=False)"

# Copy the rest of the application
COPY . .

# Create directories for uploads and ensure proper permissions
RUN mkdir -p uploads product_images
RUN chmod -R 777 uploads product_images

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV WORKERS=1
ENV THREADS=4
ENV TIMEOUT=300
ENV MAX_REQUESTS=1000
ENV KEEP_ALIVE=5

# Command to run the application using gunicorn with optimized settings
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--workers", "1", \
     "--threads", "4", \
     "--max-requests", "1000", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--worker-class", "gthread", \
     "app:app"]
