FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data
RUN mkdir -p /app/results

# Copy application code
COPY . .

EXPOSE 5002

# Increase timeout to 300 seconds (5 minutes) for processing larger files
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--timeout", "300", "image_processor:app"]
