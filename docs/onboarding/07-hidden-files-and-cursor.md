# Hidden Files and Cursor Setup

This guide covers the "hidden" configuration files that make the project work, including environment variables, Cursor AI configuration, and Git settings.

**Time Required:** 5-10 minutes

---

## 1. The `.env` File (Environment Variables)

### What It Is

The `.env` file stores **secret** configuration like API keys and passwords. This file is **gitignored** (never committed to Git) because it contains sensitive data.

### File Location

```bash
# Actual config file (NEVER commit this)
/workspaces/interview_analyzer_chaining/.env

# Template file (safe to commit)
/workspaces/interview_analyzer_chaining/docs/onboarding/env.example
```

### How to Create It

If you followed **02-initial-setup.md**, you already created this file. If not:

```bash
cd ~/Developer/interview_analyzer_chaining

# Copy template from docs/onboarding/
cp docs/onboarding/env.example .env
```

Now edit it with your actual API keys:

```bash
cursor .env
```

### Complete Example (with your keys filled in)

```bash
# Redis Configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND_URL=redis://redis:6379/1

# OpenAI API (REQUIRED)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]

# Google Gemini API (OPTIONAL)
# Get from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=[YOUR_GEMINI_API_KEY]

# Neo4j Database Password
# Check docker-compose.yml for the password (NEO4J_AUTH line)
NEO4J_PASSWORD=[CHECK_DOCKER_COMPOSE_YML]

# EventStoreDB Connection
ESDB_CONNECTION_STRING=esdb://eventstore:2113?tls=false

# Projection Service
ENABLE_PROJECTION_SERVICE=true
PROJECTION_LANE_COUNT=12

# Optional: Force environment type (auto-detected by default)
# ENVIRONMENT=host  # Options: host, docker, ci
```

### Security Best Practices

✓ **DO:**

- Keep `.env` local only (never commit to Git)
- Store API keys in a password manager (1Password, LastPass, Bitwarden)
- Use separate keys for each developer
- Rotate keys if they're exposed

✗ **DON'T:**

- Share your `.env` file in Slack/email
- Commit `.env` to Git (it's gitignored for a reason)
- Use production keys for local development
- Store keys in plain text notes

### Verifying Your `.env` File

Check that your file exists and has the right keys:

```bash
cd ~/Developer/interview_analyzer_chaining

# Verify file exists
ls -la .env

# Check it has your keys (shows first 20 lines)
head -20 .env
```

**Expected output:**

```
-rw-r--r--  1 yourname  staff  1234 Nov  9 14:30 .env
```

And you should see your actual API keys (not "your-actual-key-here").

### Troubleshooting `.env` Issues

**Problem: "API key not found"**

Check that your `.env` file has the key without extra spaces:

```bash
# WRONG (has spaces around =)
OPENAI_API_KEY = [YOUR_KEY]

# RIGHT (no spaces)
OPENAI_API_KEY=[YOUR_KEY]
```

**Problem: "Permission denied"**

Make the file readable:

```bash
chmod 600 .env  # Only you can read/write
```

**Problem: "API key invalid"**

Verify your key works:

```bash
# Test OpenAI API key (replace [YOUR_KEY] with actual key)
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer [YOUR_KEY]" | head -20
```

If you see `"error": {"message": "Incorrect API key"...}`, regenerate your key at https://platform.openai.com/api-keys.

---

## 2. The `.cursorrules` File (Cursor AI Configuration)

### What It Is

The `.cursorrules` file tells Cursor's AI assistant about your project's architecture, patterns, and preferences. This makes AI suggestions **much better** because it understands your codebase.

### File Location

```bash
# Actual config file (optional, can be in root)
/workspaces/interview_analyzer_chaining/.cursorrules

# Template file (in docs/onboarding/)
/workspaces/interview_analyzer_chaining/docs/onboarding/cursorrules.example
```

### How to Create It

Copy the template to your project root:

```bash
cd ~/Developer/interview_analyzer_chaining
cp docs/onboarding/cursorrules.example .cursorrules
```

**Expected output:**

```
-rw-r--r--  1 yourname  staff  7890 Nov  9 14:35 .cursorrules
```

**Note:** The .cursorrules file is optional but highly recommended for better AI assistance.

```bash
cat > .cursorrules << 'EOF'
# Interview Analyzer - Cursor AI Rules

## Project Context
This is an event-sourced interview analysis system using:
- Python 3.11+
- FastAPI, Neo4j, EventStoreDB, Celery, Redis

## Code Style
- Max line length: 120 characters
- Type hints required everywhere
- PEP 8 compliant (checked with flake8)

## Architecture
- Event sourcing: Write to EventStoreDB first
- Projections: Sync events to Neo4j
- Always use correlation_id for event tracing
- Environment-aware config (Docker/CI/Host)

(See full file in repository for complete rules)
EOF
```

### What It Does

When you use Cursor's AI features (`Cmd+K` for inline edits, `Cmd+L` for chat), the AI:

- ✓ Suggests code that matches your architecture (event sourcing, projections)
- ✓ Uses correct connection strings for your environment
- ✓ Follows PEP 8 style (120 char line length, type hints)
- ✓ Knows about Neo4j transaction handling quirks
- ✓ Suggests proper test structure with pytest

**Example: Without `.cursorrules`**

AI suggests:

```python
# Generic, wrong for this project
def get_data():
    return db.query("SELECT * FROM users")
```

**With `.cursorrules`**

AI suggests:

```python
# Correct for event-sourced architecture
async def get_interview_projection(interview_id: str, session: AsyncSession) -> Interview:
    """Load interview read model from Neo4j."""
    query = "MATCH (i:Interview {id: $id}) RETURN i"
    result = await session.run(query, id=interview_id)
    # ... proper Neo4j handling
```

### How Cursor Uses `.cursorrules`

1. **Inline Edit (Cmd+K):** AI reads rules before suggesting code changes
2. **AI Chat (Cmd+L):** AI understands project context in conversations
3. **Auto-complete:** Suggestions follow your patterns (type hints, line length)
4. **Refactoring:** AI maintains architectural patterns (event sourcing, etc.)

### Customizing for Your Workflow

You can add personal preferences to `.cursorrules`:

```bash
cursor .cursorrules
```

Add at the bottom:

```
## Personal Preferences
- Prefer single-line docstrings for simple functions
- Always add TODO comments for incomplete features
- Use explicit variable names (no abbreviations)
```

The AI will incorporate these into suggestions.

---

## 3. The `.gitignore` File (Git Exclusions)

### What It Is

The `.gitignore` file tells Git which files to **never** track. This prevents accidental commits of secrets, build artifacts, and IDE settings.

### File Location

```bash
# Actual file
/workspaces/interview_analyzer_chaining/.gitignore

# Template (for reference)
docs/onboarding/gitignore.example
```

### Purpose

Prevents Git from tracking:

- **Secrets** (`.env`, API keys)
- **Generated files** (`__pycache__`, `*.pyc`, `htmlcov/`)
- **OS files** (`.DS_Store`, `Thumbs.db`)
- **IDE files** (`.idea/`, `.vscode/*`)
- **Dependencies** (`venv/`, `node_modules/`)

### Key Patterns Explained

**Critical (Security):**

```gitignore
.env                            # Environment variables with API keys
.devcontainer/devcontainer.env  # DevContainer secrets
*.key                           # Private keys
*.pem                           # SSL certificates
```

**Python artifacts:**

```gitignore
__pycache__/                    # Compiled Python files
*.pyc, *.pyo, *.pyd             # Bytecode files
.pytest_cache/                  # Test cache
htmlcov/                        # Coverage reports
*.egg-info/                     # Package metadata
```

**IDE/Editor:**

```gitignore
.vscode/*                       # VS Code settings
!.vscode/settings.json          # EXCEPT this one (allow it)
.idea/                          # PyCharm settings
.DS_Store                       # macOS Finder info
```

**Virtual environments:**

```gitignore
venv/                           # Virtual environment
.venv/                          # Alternative venv name
env/                            # Another common name
```

### Pattern Syntax

```gitignore
# Ignore specific file
secret.txt

# Ignore all files with extension
*.pyc

# Ignore directory
build/

# Ignore files anywhere in tree
**/.DS_Store

# DON'T ignore (exception to previous rule)
!important.txt

# Ignore only in root
/config.json

# Ignore in any subdirectory
**/logs/
```

### Verifying `.gitignore` Works

Test that `.env` is ignored:

```bash
cd ~/Developer/interview_analyzer_chaining

# Create or modify .env
echo "TEST_VAR=secret" >> .env

# Check git status
git status
```

**Expected output:**

```
On branch main
nothing to commit, working tree clean
```

**If `.env` appears in git status:**

```bash
# It might already be tracked. Remove from tracking:
git rm --cached .env

# Verify .env is in .gitignore
grep "^\.env$" .gitignore

# If not there, add it
echo ".env" >> .gitignore

# Commit the change
git add .gitignore
git commit -m "Ensure .env is gitignored"
```

### Common `.gitignore` Mistakes

❌ **Mistake 1: File already committed**

If you committed `.env` before adding to `.gitignore`:

```bash
# Remove from tracking (keeps local file)
git rm --cached .env
git commit -m "Remove .env from tracking"
git push
```

⚠️ **This doesn't remove from history!** See `SECURITY-WARNING.md` for history cleaning.

❌ **Mistake 2: Wrong directory patterns**

```gitignore
# WRONG - Only ignores venv/ in root
/venv/

# RIGHT - Ignores venv/ anywhere
venv/
# Or explicitly:
**/venv/
```

❌ **Mistake 3: Trailing whitespace**

```gitignore
.env      # WRONG - has trailing spaces
.env      # RIGHT - no trailing spaces
```

Git won't match patterns with extra whitespace.

❌ **Mistake 4: Ignoring too much**

```gitignore
# BAD - Ignores ALL json files (including config templates)
*.json

# BETTER - Be specific
config.local.json
secrets.json
```

### Customizing for Your Needs

**To ignore project-specific files:**

```bash
# Edit .gitignore
cursor .gitignore

# Add at bottom:
# My project files
/my_experiments/
temp_*.py
scratch/
```

**To check what's being ignored:**

```bash
# List all ignored files
git status --ignored

# Check if specific file is ignored
git check-ignore -v .env
# Output: .gitignore:80:.env    .env
```

**To force-add ignored file (rare):**

```bash
# If you really need to commit an ignored file
git add -f special_config.json
```

### Project-Specific Ignores

This project ignores:

```gitignore
# Data directories (may want to track samples)
data/
output/

# AI assistant files
.aider*
repomix-output.xml

# DevContainer secrets (but not devcontainer.json!)
.devcontainer/devcontainer.env
```

### Verification Commands

```bash
# Test .env is ignored
echo "test" > .env
git status | grep .env  # Should show nothing

# Test .pyc is ignored
touch test.pyc
git status | grep test.pyc  # Should show nothing

# Clean up
rm test.pyc

# See all ignored files
git status --ignored --short

# Check why file is ignored
git check-ignore -v htmlcov/index.html
# Shows which .gitignore rule matched
```

### When to Update `.gitignore`

Update when:

- ✅ Adding new tools (e.g., adding Poetry → ignore `poetry.lock`)
- ✅ New generated files appear (e.g., `.mypy_cache/`)
- ✅ New IDE (e.g., adding Sublime Text → ignore `*.sublime-*`)
- ✅ New OS (e.g., adding Linux → ignore `*~`)

Don't update when:

- ❌ Files should be tracked (code, configs without secrets)
- ❌ Already covered by existing patterns
- ❌ One-off temporary files (just delete them)

---

## 4. Cursor IDE Configuration

### Workspace Settings

Cursor stores workspace-specific settings in `.vscode/settings.json` (this file is gitignored, so everyone can customize).

Create it for better Python support:

```bash
cd ~/Developer/interview_analyzer_chaining
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "/usr/local/bin/python3",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=120"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "120"],
  "editor.formatOnSave": true,
  "editor.rulers": [120],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    "htmlcov": true
  }
}
EOF
```

**What this does:**

- Enables Flake8 linting (shows style errors in real-time)
- Auto-formats code with Black on save (120 char line length)
- Shows a ruler at 120 characters
- Hides Python artifacts (`__pycache__`, etc.)

### Enabling Format on Save

1. Open Cursor
2. Press `Cmd+,` (opens Settings)
3. Search for "format on save"
4. Check "Format On Save"

Now when you press `Cmd+S`, Black automatically formats your code.

### Python Extension Setup

Cursor should auto-install the Python extension. Verify:

1. Click Extensions icon (left sidebar) or press `Cmd+Shift+X`
2. Search "Python"
3. Confirm "Python" extension is installed (by Microsoft)

If not installed:

```
ms-python.python
```

Paste this ID in Extensions search, click Install.

---

## 5. Docker Configuration Files

### `docker-compose.yml`

Defines all services (Neo4j, EventStore, Redis, API, etc.).

**Location:** `/workspaces/interview_analyzer_chaining/docker-compose.yml`

**View it:**

```bash
cd ~/Developer/interview_analyzer_chaining
cat docker-compose.yml | head -50
```

**Key sections:**

- **neo4j**: Graph database (port 7687 for driver, 7474 for browser)
- **eventstore**: Event store (port 2113)
- **redis**: Message broker (port 6379)
- **app**: Python API (port 8000)
- **projection-service**: Syncs EventStore → Neo4j

**Customizing resources:**

If your Mac is low on RAM, edit resource limits:

```bash
cursor docker-compose.yml
```

Find the `neo4j` service:

```yaml
neo4j:
  # ...
  environment:
    NEO4J_dbms_memory_heap_initial__size: 512M # Default: 512M
    NEO4J_dbms_memory_heap_max__size: 2G # Default: 2G
```

For a Mac with 8GB RAM, reduce to:

```yaml
NEO4J_dbms_memory_heap_initial__size: 256M
NEO4J_dbms_memory_heap_max__size: 1G
```

Then rebuild:

```bash
make db-down
make build
make db-up
```

### `Dockerfile`

Defines the Python app container.

**Location:** `/workspaces/interview_analyzer_chaining/Dockerfile`

**View it:**

```bash
cat Dockerfile
```

You usually **don't need to edit this** unless adding system dependencies.

**Example: Adding ImageMagick**

```dockerfile
# Add after "RUN apt-get update"
RUN apt-get install -y imagemagick
```

Then rebuild:

```bash
make build
```

---

## 6. Development Configuration Files

### `Makefile`

Provides convenient commands like `make test`, `make lint`, etc.

**Location:** `/workspaces/interview_analyzer_chaining/Makefile`

**View all commands:**

```bash
make help
```

**Example output:**

```
Available targets:
  build      Build Docker images
  run        Run pipeline on sample file
  test       Run all tests
  coverage   Generate HTML coverage report
  lint       Check code style with flake8
  format     Auto-format code with black
  db-up      Start all services
  db-down    Stop all services
  clean      Remove all volumes (fresh start)
```

**How to use:**

```bash
make test    # Runs: docker compose run --rm app pytest -v
make lint    # Runs: docker compose run --rm app flake8 src tests
```

### `requirements.txt`

Lists all Python dependencies.

**Location:** `/workspaces/interview_analyzer_chaining/requirements.txt`

**View it:**

```bash
cat requirements.txt | head -20
```

**Adding a new dependency:**

```bash
cursor requirements.txt
```

Add at the bottom:

```
# New dependency
requests==2.31.0
```

Then rebuild:

```bash
make build
```

**Checking installed versions:**

```bash
docker compose run --rm app pip list | grep requests
```

### `pytest.ini`

Configures pytest behavior.

**Location:** `/workspaces/interview_analyzer_chaining/pytest.ini`

**View it:**

```bash
cat pytest.ini
```

**Example content:**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
markers =
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

**Running specific markers:**

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration
```

### `.flake8`

Configures Flake8 linter for Python code quality checks.

**Location:**

```bash
# Actual file
/workspaces/interview_analyzer_chaining/.flake8

# Template (for reference)
docs/onboarding/flake8.example
```

**Purpose:** Controls code style enforcement, line length limits, and which errors to ignore.

**Key settings:**

```ini
max-line-length = 120      # Allow 120 chars per line (not 79)
exclude = .git,__pycache__ # Don't lint these directories
ignore = E203,W503,F401    # Ignore specific error codes
```

**Ignored errors explained:**

- **E203:** Whitespace before ':' (conflicts with Black formatter)
- **W503:** Line break before binary operator (deprecated warning)
- **F401:** Module imported but unused (needed for `__init__.py` exports)

**Using Flake8:**

```bash
# Run linting
make lint
# Or manually:
flake8 src tests

# Check specific file
flake8 src/pipeline.py
```

**Common Flake8 errors:**

- **E501:** Line too long (>120 characters)
- **F841:** Local variable assigned but never used
- **E302:** Expected 2 blank lines, found 1
- **W291:** Trailing whitespace

**Customizing:**

To ignore errors in specific lines:

```python
# Ignore specific error on this line
some_long_url = "https://..."  # noqa: E501

# Ignore all errors on this line
debug_code = True  # noqa
```

### `.coveragerc`

Configures code coverage measurement with coverage.py.

**Location:**

```bash
# Actual file
/workspaces/interview_analyzer_chaining/.coveragerc

# Template (for reference)
docs/onboarding/coveragerc.example
```

**Purpose:** Controls what code is measured, how reports are generated, and what lines to exclude from coverage.

**Key settings:**

**[run] section:**

```ini
branch = True          # Measure branch coverage (if/else paths)
source = src           # Only measure code in src/ directory
omit = */tests/*       # Don't measure test files themselves
```

**[report] section:**

```ini
show_missing = True    # Show which lines aren't covered
precision = 1          # Show coverage as XX.X%
sort = Cover           # Sort by coverage percentage
```

**[html] section:**

```ini
directory = htmlcov    # Generate HTML reports in htmlcov/
```

**Using coverage:**

```bash
# Run tests with coverage
make coverage
# Or manually:
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

**Customizing:**

To exclude specific lines from coverage, add comment:

```python
def debug_function():  # pragma: no cover
    """This won't be counted in coverage."""
    print("Debug info")
```

To exclude specific files, edit `.coveragerc`:

```ini
omit =
    */tests/*
    */my_experimental_module.py  # Add your file here
```

---

## 7. Environment Detection System

The application auto-detects whether it's running in Docker, CI, or on your Mac.

### How It Works

**Code:** `src/config.py`

```python
def _detect_environment() -> str:
    """Auto-detect runtime environment."""
    if os.path.exists("/.dockerenv"):
        return "docker"
    if os.getenv("CI") == "true":
        return "ci"
    return "host"
```

### Connection String Examples

**Docker (inside container):**

- Neo4j: `neo4j://neo4j:7687` (uses container hostname)
- EventStore: `esdb://eventstore:2113?tls=false`

**Host (your Mac):**

- Neo4j: `neo4j://localhost:7687` (Docker port-forwarded)
- EventStore: `esdb://localhost:2113?tls=false`

### Overriding Detection

If auto-detection fails, force environment type:

```bash
cursor .env
```

Add:

```bash
ENVIRONMENT=host  # Options: host, docker, ci
```

Then restart services:

```bash
make db-down
make db-up
```

### Verifying Detection

Check which environment is detected:

```bash
docker compose run --rm app python -c "from src.config import Config; print(Config().environment)"
```

**Expected output:**

```
docker
```

From your Mac terminal:

```bash
python3 -c "from src.config import Config; print(Config().environment)"
```

**Expected output:**

```
host
```

---

## 8. Quick Reference

### File Locations Summary

| File                    | Purpose                   | Committed to Git?  |
| ----------------------- | ------------------------- | ------------------ |
| `.env`                  | API keys, secrets         | ❌ No (gitignored) |
| `.env.example`          | Template for `.env`       | ✅ Yes             |
| `.cursorrules`          | Cursor AI config          | ✅ Yes             |
| `.gitignore`            | Files to exclude from Git | ✅ Yes             |
| `.vscode/settings.json` | Cursor IDE settings       | ❌ No (gitignored) |
| `docker-compose.yml`    | Service definitions       | ✅ Yes             |
| `Dockerfile`            | Python app container      | ✅ Yes             |
| `Makefile`              | Development commands      | ✅ Yes             |
| `requirements.txt`      | Python dependencies       | ✅ Yes             |
| `pytest.ini`            | Test configuration        | ✅ Yes             |

### Essential Commands

```bash
# View hidden files in Finder
Cmd+Shift+.

# View hidden files in terminal
ls -la

# Edit .env file
cursor .env

# Test .env is working
grep OPENAI_API_KEY .env

# Check gitignore is working
git status | grep .env  # Should show nothing

# Reload Cursor AI rules
Cmd+Shift+P → "Developer: Reload Window"

# Check environment detection
docker compose run --rm app python -c "from src.config import Config; print(Config().environment)"
```

### Troubleshooting

**Problem: Cursor AI suggestions are generic/wrong**

1. Verify `.cursorrules` exists:

   ```bash
   ls -la .cursorrules
   ```

2. Reload Cursor:

   ```
   Cmd+Shift+P → "Developer: Reload Window"
   ```

3. Try AI chat:

   ```
   Cmd+L → Ask "What architecture pattern does this project use?"
   ```

   Should mention "event sourcing" and "Neo4j projections".

**Problem: API key errors even though `.env` is correct**

1. Check for extra whitespace:

   ```bash
   cat .env | grep OPENAI_API_KEY | cat -v
   ```

   Should show NO `^M` (carriage return) or extra spaces.

2. Verify key format:

   ```bash
   grep OPENAI_API_KEY .env
   ```

   Should start with `sk-proj-` or `sk-` (format: `sk-proj-[long-string]`).

3. Test key directly:

   ```bash
   export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d= -f2)
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY" | head -20
   ```

   Should show list of models, not error.

**Problem: Docker can't connect to databases**

1. Check environment detection:

   ```bash
   docker compose run --rm app python -c "from src.config import Config; c=Config(); print(f'{c.environment=}, {c.neo4j.uri=}')"
   ```

   Should show `environment='docker', uri='neo4j://neo4j:7687'`.

2. Force Docker environment:

   ```bash
   cursor .env
   # Add: ENVIRONMENT=docker
   ```

3. Restart:
   ```bash
   make db-down && make db-up
   ```

---

## Next Steps

✅ You've now configured all hidden files and understand how they work.

**Next:** Return to **05-development-workflow.md** to start coding, or see **06-troubleshooting.md** if you encounter issues.

**For Security:** Read **SECURITY-WARNING.md** to understand API key safety.

---

**Have questions?** Check **00-README.md** for links to all onboarding guides.
