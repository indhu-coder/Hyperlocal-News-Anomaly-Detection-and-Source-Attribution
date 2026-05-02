FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some ML/NLP libs)
RUN apt-get update && \
    apt-get install -y build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Install spaCy small model (IMPORTANT)
RUN python -m spacy download en_core_web_sm

# Copy all files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
