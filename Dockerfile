# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

ENV PYTHONPATH="/app:/app/src"


# Expose port (customize this if your app runs on another port)
EXPOSE 8080

# Run the app (adjust this if app.py has a different entry point)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

