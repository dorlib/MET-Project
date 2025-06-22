FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Make the script executable
RUN chmod +x app.py

# Expose port
EXPOSE 5003

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "app:app"]
