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

# Command to run the application using gunicorn with increased timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "300", "--workers", "1", "--threads", "4", "app:app"]
