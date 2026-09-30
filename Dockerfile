# MultiMix AI — Cloud Deployment Container
# Operating System: Linux (Debian slim)
# Runtime: Python 3.12
FROM python:3.12-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

# Install Linux system dependencies required for speech/audio processing and C-extension building
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration, and scripts
# Note: Models are excluded via .dockerignore and should be mounted or provisioned separately
COPY . .

# Ensure Linux execute permissions for startup script
RUN chmod +x scripts/start.sh

# Expose default Streamlit port (can be overridden via $PORT environment variable)
EXPOSE 8501

# Execute startup script
CMD ["./scripts/start.sh"]
