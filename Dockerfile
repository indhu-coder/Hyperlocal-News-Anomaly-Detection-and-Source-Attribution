FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install only required system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker cache
COPY requirements.txt .

# Install CPU-only torch first, then remaining requirements
# IMPORTANT: remove torch from requirements.txt to avoid CUDA/GPU install
RUN pip install --upgrade pip && \
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy project files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]

