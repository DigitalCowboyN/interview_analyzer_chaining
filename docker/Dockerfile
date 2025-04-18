FROM python:3.10-slim

# Set working directory
WORKDIR /workspaces/interview_analyzer_chaining

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set OpenAI API key as an environment variable
ENV OPENAI_API_KEY=""

# Copy project files
COPY . ./.

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# command to run when the container starts
CMD ["python", "src/main.py"]
