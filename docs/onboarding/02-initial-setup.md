# Initial Setup

This guide walks you through cloning the repository, configuring environment variables, and building the Docker images.

**Prerequisites:** Complete [01-prerequisites.md](./01-prerequisites.md) first.

---

## 1. Choose a Workspace Directory

First, decide where you want to keep your code projects.

### Recommended Structure

```bash
# Create a workspace directory if you don't have one
mkdir -p ~/workspace
cd ~/workspace
```

**Note:** This directory will contain ~3GB of code, Docker images, and data.

**You can use any directory you prefer. Common alternatives:**

- `~/Developer` - If using this: `mkdir -p ~/Developer && cd ~/Developer`
- `~/Projects` - If using this: `mkdir -p ~/Projects && cd ~/Projects`
- `~/Code` - If using this: `mkdir -p ~/Code && cd ~/Code`

**For this guide, we'll use `~/workspace/` as the default path.**

---

## 2. Clone the Repository

### Clone Options

You can clone the repository using HTTPS, SSH, or GitHub CLI:

**Option 1: HTTPS (Recommended for most users)**

```bash
cd ~/workspace
git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining.git
cd interview_analyzer_chaining
```

**Option 2: SSH (If you have SSH keys set up with GitHub)**

```bash
cd ~/workspace
git clone git@github.com:DigitalCowboyN/interview_analyzer_chaining.git
cd interview_analyzer_chaining
```

**Option 3: GitHub CLI (If you have `gh` installed)**

```bash
cd ~/workspace
gh repo clone DigitalCowboyN/interview_analyzer_chaining
cd interview_analyzer_chaining
```

**For this guide, we'll use HTTPS (Option 1).**

### Verification

```bash
ls -la
```

**Expected output:** You should see files like:

- `docker-compose.yml`
- `Makefile`
- `requirements.txt`
- `config.yaml`
- `src/`
- `tests/`
- `docs/`

---

## 3. Create Environment File

The project requires a `.env` file to store secrets and configuration. This file is **not** in version control for security reasons.

### Copy the Template

From the project root directory:

```bash
# Copy the template from docs/onboarding/
cp docs/onboarding/env.example .env
```

### Open in Cursor

```bash
cursor .env
```

Or open Cursor, then: File → Open → Navigate to `interview_analyzer_chaining/.env`

### Update the Placeholder Values

The template contains placeholders in brackets. Replace them with your actual values:

```bash
# ============================================
# API Keys (Use your personal keys from Prerequisites step)
# ============================================
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
GEMINI_API_KEY=[YOUR_GEMINI_API_KEY]

# ============================================
# Neo4j Database Configuration
# ============================================
# For local development, check the password in docker-compose.yml
# Look for the line: NEO4J_AUTH: "neo4j/[PASSWORD]"
NEO4J_PASSWORD=[CHECK_DOCKER_COMPOSE_YML]

# ============================================
# Environment Detection (Usually auto-detected, but can override)
# ============================================
# ENVIRONMENT=host  # Uncomment if auto-detection fails
```

### ⚠️ Security Warning

**NEVER commit your `.env` file to Git!** It contains secret API keys that can cost you money if exposed.

- ✓ `.env` is already in `.gitignore` (safe by default)
- ✓ Use `.env.example` as a template (no real secrets)
- ✗ Never share your `.env` in Slack/email
- ✗ Never copy/paste your keys in public channels

**See [SECURITY-WARNING.md](./SECURITY-WARNING.md) for complete security guidance.**

### Replace Placeholder Values

1. **OPENAI_API_KEY:** Replace `[YOUR_OPENAI_API_KEY]` with your actual OpenAI key from Prerequisites step
2. **GEMINI_API_KEY:** Replace `[YOUR_GEMINI_API_KEY]` with your actual Gemini key from Prerequisites step
3. **NEO4J_PASSWORD:** Replace `[CHECK_DOCKER_COMPOSE_YML]` with the password from docker-compose.yml (look for NEO4J_AUTH line)

### Example (DO NOT copy these placeholders)

```bash
# Example only - use YOUR actual keys
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
GEMINI_API_KEY=[YOUR_GEMINI_API_KEY]
NEO4J_PASSWORD=[CHECK_DOCKER_COMPOSE_YML]
```

### Save and Close

In Cursor: `Cmd+S` to save, then close the file.

### Security Check

Verify `.env` is in `.gitignore`:

```bash
grep "^\.env$" .gitignore
```

**Expected output:** `.env`

This confirms your secrets won't be committed to Git.

---

## 4. Review Configuration Files

Before building, let's understand the key configuration files.

### docker-compose.yml

This defines 7 services:

```bash
cat docker-compose.yml | grep "^  [a-z]" | grep -v "^  #"
```

**Services you'll see:**

1. `app` - FastAPI application (port 8000)
2. `redis` - Message broker (port 6379)
3. `worker` - Celery background worker
4. `neo4j` - Main graph database (ports 7474, 7687)
5. `neo4j-test` - Test database (ports 7475, 7688)
6. `eventstore` - EventStoreDB (ports 2113, 1113)
7. `projection-service` - Event projection service

### config.yaml

Application-level configuration. Let's verify it exists:

```bash
head -20 config.yaml
```

**You should see:**

- OpenAI model configuration
- Gemini configuration
- Path settings
- Pipeline workers configuration
- Neo4j connection settings

**Note:** Environment variables in `config.yaml` (like `${OPENAI_API_KEY}`) are automatically replaced with values from your `.env` file at runtime.

---

## 5. Build Docker Images

Now we'll build the Docker images for the application services.

### Start Build Process

From the project root:

```bash
docker compose build app worker
```

**What this does:**

- Reads `docker/Dockerfile`
- Creates a Python 3.10 environment
- Installs all dependencies from `requirements.txt`
- Downloads spaCy language model
- Runs security audit
- Builds images for both `app` and `worker` services

### Expected Duration

**First build:** 10-15 minutes (downloads and installs everything)
**Subsequent builds:** 2-5 minutes (uses cache)

### Monitor Progress

You'll see output like:

```
[+] Building 234.5s (18/18) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 1.23kB
 => [internal] load .dockerignore
 => CACHED [1/10] FROM docker.io/library/python:3.10.14-slim
 => [2/10] WORKDIR /workspaces/interview_analyzer_chaining
 ...
```

### Potential Warnings

You might see:

```
Warning: pip-audit found vulnerabilities. Review above output.
```

**This is expected** - some dependency vulnerabilities may not have fixes yet. The build continues.

### Verification

After successful build:

```bash
docker images | grep interview_analyzer
```

**Expected output:**

```
interview_analyzer_chaining-app       latest    abc123def456   2 minutes ago   2.1GB
interview_analyzer_chaining-worker    latest    abc123def456   2 minutes ago   2.1GB
```

**Note:** Both images share the same base, so they'll have the same image ID.

---

## 6. Verify Build Success

### Check Docker Images

```bash
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "REPOSITORY|interview_analyzer"
```

### Test Import (Optional)

Verify Python dependencies are correctly installed:

```bash
docker compose run --rm app python -c "import fastapi; import openai; import neo4j; print('✓ Core imports successful')"
```

**Expected output:** `✓ Core imports successful`

If you see import errors, the build may have failed. Try rebuilding:

```bash
docker compose build --no-cache app worker
```

---

## 7. Project Structure Overview

Now that you have the code, let's review the structure.

**Option 1 - Using find (built-in command):**

```bash
find . -type d -maxdepth 2 -not -path '*/\.*' | sort
```

**Option 2 - Install tree (optional):**

```bash
# Install tree for better visualization
brew install tree

# Then run:
tree -L 2 -I '__pycache__|htmlcov|*.pyc|.pytest_cache' -d
```

**Key directories:**

```
.
├── data/               # Data files (input, output, maps)
│   ├── input/         # Place .txt interview transcripts here
│   ├── output/        # Analysis results (.jsonl files)
│   └── maps/          # Intermediate sentence maps
├── docker/            # Dockerfile
├── docs/              # Documentation (you're here!)
│   └── onboarding/   # This onboarding guide
├── logs/              # Application logs
├── prompts/           # LLM prompt templates (YAML)
│   ├── domain_prompts.yaml
│   └── task_prompts.yaml
├── src/               # Source code
│   ├── agents/       # LLM interaction logic
│   ├── api/          # FastAPI routes and schemas
│   ├── commands/     # Command handlers (CQRS)
│   ├── events/       # Event sourcing (aggregates, events)
│   ├── io/           # Input/Output abstractions
│   ├── models/       # Pydantic data models
│   ├── persistence/  # Database persistence layer
│   ├── projections/  # Event projection handlers
│   ├── services/     # Business logic services
│   └── utils/        # Utilities, helpers, config
└── tests/             # Test suite (673 tests)
    ├── api/          # API endpoint tests
    ├── commands/     # Command handler tests
    ├── events/       # Event sourcing tests
    ├── integration/  # Integration tests
    └── projections/  # Projection tests
```

---

## Setup Checklist

Before proceeding, verify:

- [ ] Repository cloned successfully
- [ ] `.env` file created with your actual API keys
- [ ] `.env` contains `NEO4J_PASSWORD=[password_from_docker-compose.yml]`
- [ ] Docker images built successfully (`docker images | grep interview_analyzer` shows 2 images)
- [ ] Test import succeeded
- [ ] You understand the project directory structure

---

## Troubleshooting

### "Permission denied" when cloning

**If SSH clone fails, use HTTPS instead:**

```bash
git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining.git
```

### Docker build fails with "no space left on device"

```bash
# Clean up Docker
docker system prune -a --volumes
# Then retry build
docker compose build app worker
```

### Build fails with "unable to resolve image"

- Check your internet connection
- Verify Docker Desktop is running
- Try manually pulling the base image first (downloads ~150MB):
  ```bash
  docker pull python:3.10.14-slim
  ```
- Then retry the build:
  ```bash
  docker compose build app worker
  ```

### ".env file not working"

- Ensure no spaces around `=` in `.env`
- No quotes needed around values
- Save file as `.env` exactly (not `.env.txt`)

### Cursor won't open .env file

```bash
# Open in default editor
open -e .env
# Or use nano in terminal
nano .env
```

---

## What's Next?

Your development environment is configured!

Next: [Running the System →](./03-running-the-system.md)

Learn how to start all services and verify everything works.
