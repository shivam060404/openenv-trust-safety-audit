FROM python:3.11-slim-bookworm

# Metadata
LABEL org.opencontainers.image.title="trust-safety-audit-env"
LABEL org.opencontainers.image.description="OpenEnv Trust & Safety Audit Environment"

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Expose the API port (HF Spaces default)
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]