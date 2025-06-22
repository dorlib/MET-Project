FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Increase timeout to 300 seconds (5 minutes) for handling NIfTI file uploads and processing
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "api:app"]
