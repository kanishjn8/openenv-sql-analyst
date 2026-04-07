# Dockerfile — Container definition for the SQL Analyst environment (Person A)
#
# Builds a slim Python 3.11 image, installs dependencies, seeds the
# database, and starts the FastAPI server on port 7860 (HF Spaces default).

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Generate the database at build time (deterministic seed)
RUN python data/seed.py

# Expose the Hugging Face Spaces default port
EXPOSE 7860

# Start the FastAPI server via uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
