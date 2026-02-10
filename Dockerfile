# ============================================================
# Stage 1: Base image with Python
# ============================================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install curl for Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Stage 2: Install dependencies (cached layer)
# ============================================================
# Copy ONLY requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python packages
# --no-cache-dir: don't store pip cache (smaller image)
# We install CPU-only PyTorch to save ~1.5GB
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    faiss-cpu \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    streamlit \
    Pillow \
    numpy \
    pandas \
    requests

# ============================================================
# Stage 3: Copy application code and data
# ============================================================
# Copy API code
COPY api/ ./api/

# Copy frontend code
COPY frontend/ ./frontend/

# Copy model weights
COPY models/ ./models/

# Copy data files (images, metadata, embeddings)
COPY data/ ./data/

# Copy FAISS index
COPY indexing/ ./indexing/

# ============================================================
# Stage 4: Default command (API server)
# ============================================================
# Expose port 8000 for the API
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
