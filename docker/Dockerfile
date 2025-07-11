FROM python:3.10.14-slim

# Set working directory
WORKDIR /workspaces/interview_analyzer_chaining

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    redis-tools \
    # Upgrade installed packages
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy project files
COPY . ./.

# Install Python dependencies (requirements.txt is the single source of truth)
RUN pip install --no-cache-dir -r requirements.txt pip-audit

# Scan installed packages for vulnerabilities
RUN pip-audit

# Install spaCy model (must match requirements.txt version)
RUN python -m spacy download en_core_web_sm

# command to run when the container starts
CMD ["python", "src/main.py"]
