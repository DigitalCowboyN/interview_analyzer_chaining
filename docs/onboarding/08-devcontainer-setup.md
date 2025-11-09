# DevContainer Setup Guide

This guide explains the DevContainer configuration for this project, which enables development in a fully configured Docker-based environment using Cursor IDE or VS Code.

**Time Required:** 10-15 minutes (first-time setup)

---

## What is a DevContainer?

A **DevContainer** (Development Container) is a Docker container specifically configured for development. It provides:

- ✅ Consistent development environment across all machines
- ✅ Pre-installed tools and extensions
- ✅ Automatic service startup (databases, APIs, etc.)
- ✅ Isolated environment (no conflicts with your Mac)

**Think of it as:** Your entire development environment packaged in a container.

---

## DevContainer Files

### 1. `.devcontainer/devcontainer.json` (Main Configuration)

**Purpose:** Defines how Cursor/VS Code sets up your development container.

**Location:**
```bash
/workspaces/interview_analyzer_chaining/.devcontainer/devcontainer.json
```

**Template Location:**
```bash
# Template is in docs/onboarding/ for reference
docs/onboarding/devcontainer.json.example
```

### 2. `.devcontainer/devcontainer.env` (Environment Variables)

**Purpose:** Stores API keys and secrets for the DevContainer.

**Location:**
```bash
/workspaces/interview_analyzer_chaining/.devcontainer/devcontainer.env
```

**Template Location:**
```bash
# Template for creating your own
docs/onboarding/devcontainer.env.example
```

**How to create:**
```bash
cd ~/Developer/interview_analyzer_chaining
cp docs/onboarding/devcontainer.env.example .devcontainer/devcontainer.env
cursor .devcontainer/devcontainer.env
# Replace [PLACEHOLDERS] with actual keys
```

---

## DevContainer Configuration Explained

### Basic Settings

```json
{
  "name": "Interview Analyzer Chaining Dev",
  "dockerComposeFile": ["../docker-compose.yml"],
  "service": "app",
  "workspaceFolder": "/workspaces/interview_analyzer_chaining"
}
```

**What this means:**
- **name:** Display name in Cursor/VS Code
- **dockerComposeFile:** Uses your project's docker-compose.yml
- **service:** Connects to the "app" service from docker-compose.yml
- **workspaceFolder:** Where your code appears inside the container

### Port Forwarding

```json
"forwardPorts": [
  8000,   // FastAPI (main API)
  6379,   // Redis
  7474,   // Neo4j Browser (main)
  7687,   // Neo4j Bolt (main)
  2113,   // EventStoreDB UI
  1113    // EventStoreDB gRPC
]
```

**What this means:**
- These ports are automatically forwarded from the container to your Mac
- Access services at `http://localhost:[port]` in your browser
- No manual port configuration needed

### Pre-installed Extensions

```json
"extensions": [
  "ms-python.python",              // Python language support
  "ms-python.black-formatter",     // Auto-formatting (black)
  "ms-python.flake8",              // Linting (flake8)
  "esbenp.prettier-vscode",        // JSON/YAML formatting
  "ms-python.debugpy",             // Python debugging
  "ms-azuretools.vscode-docker",   // Docker support
  "njpwerner.autodocstring",       // Auto-generate docstrings
  "charliermarsh.ruff"             // Fast Python linter
]
```

**What this means:**
- These extensions are automatically installed when you open the DevContainer
- No manual extension installation needed
- Consistent tooling across all developers

### Auto-Started Services

```json
"runServices": [
  "redis",
  "neo4j",
  "neo4j-test",
  "eventstore",
  "projection-service",
  "app"
]
```

**What this means:**
- These services start automatically when you open the DevContainer
- No need to run `docker compose up` manually
- Ready to code immediately

### Post-Start Command

```json
"postStartCommand": "sleep 20 && echo 'All services ready including EventStore and Projection Service'"
```

**What this means:**
- Waits 20 seconds for all services to fully start
- Displays confirmation message in terminal
- Ensures databases are ready before you start coding

---

## Opening the Project in a DevContainer

### Method 1: Using Cursor IDE (Recommended)

1. **Open Cursor**
   ```bash
   cd ~/Developer/interview_analyzer_chaining
   cursor .
   ```

2. **Cursor will detect DevContainer**
   - Look for popup: "Folder contains a Dev Container configuration file"
   - Click: **"Reopen in Container"**

3. **Wait for Container Build** (first time only)
   - Progress shown in terminal
   - Takes 5-10 minutes first time
   - Subsequent opens: 30 seconds

4. **Verify Setup**
   - Terminal prompt should show container: `root@[container-id]`
   - Extensions auto-installed (check Extensions panel)
   - Services running (check `docker ps`)

### Method 2: Using Command Palette

1. **Open Command Palette:** `Cmd+Shift+P`
2. **Type:** "Dev Containers: Reopen in Container"
3. **Select:** The option from the dropdown
4. **Wait for container to start**

### Method 3: From Outside Cursor

If you already have Cursor open and want to open in DevContainer:

1. **Close Cursor** (if currently open)
2. **Reopen with DevContainer flag:**
   ```bash
   cursor --folder-uri vscode-remote://dev-container+[encoded-path]
   ```
   
   Or simply:
   ```bash
   cd ~/Developer/interview_analyzer_chaining
   cursor .
   # Then click "Reopen in Container" when prompted
   ```

---

## Customizing the DevContainer

### Adding More Extensions

Edit `.devcontainer/devcontainer.json`:

```json
"extensions": [
  "ms-python.python",
  // Add your extension ID here
  "your-publisher.your-extension"
]
```

**Finding Extension IDs:**
1. Open Extensions panel in Cursor (`Cmd+Shift+X`)
2. Search for extension
3. Click on extension
4. Look for "Identifier" in details (e.g., `ms-python.python`)

### Changing Python Path

If you need to add custom Python import paths:

```json
"settings": {
  "python.analysis.extraPaths": [
    "/workspaces/interview_analyzer_chaining/src",
    "/workspaces/interview_analyzer_chaining/custom_modules"
  ]
}
```

### Adding More Port Forwards

```json
"forwardPorts": [
  8000,
  5432,  // Add PostgreSQL if needed
  9090   // Add any custom service
]
```

### Changing Auto-Started Services

If you don't want all services to auto-start:

```json
"runServices": [
  "redis",
  "neo4j",
  // Remove services you don't need on startup
]
```

Then manually start them later:
```bash
docker compose up eventstore -d
```

---

## DevContainer Environment Variables

### Creating `.devcontainer/devcontainer.env`

**IMPORTANT:** This file must exist for DevContainer to work properly.

```bash
# From project root
cp docs/onboarding/devcontainer.env.example .devcontainer/devcontainer.env
cursor .devcontainer/devcontainer.env
```

**Replace all placeholders:**

```bash
# Before
OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]

# After (with your actual key)
OPENAI_API_KEY=sk-proj-abc123...
```

**Why separate from `.env`?**
- `.env` is for running services outside DevContainer (docker compose up)
- `.devcontainer/devcontainer.env` is for running inside DevContainer
- Both should have same values but serve different contexts

### Verifying Environment Variables Work

Once in DevContainer:

```bash
# Check environment variable is loaded
echo $OPENAI_API_KEY

# Should show your actual key
# If blank, devcontainer.env wasn't loaded properly
```

---

## Troubleshooting DevContainers

### Problem: "Folder contains a Dev Container configuration file" doesn't appear

**Solution 1:** Manually trigger
```
Cmd+Shift+P → "Dev Containers: Reopen in Container"
```

**Solution 2:** Check Cursor has DevContainer support
```
Cmd+Shift+P → "Dev Containers: Install" (if option exists)
```

**Solution 3:** Verify Docker is running
```bash
docker ps
# Should show running containers or empty list (not error)
```

### Problem: Container build fails

**Solution 1:** Check Docker resources
- Docker Desktop → Settings → Resources
- Ensure: 4+ CPUs, 8GB+ Memory

**Solution 2:** Clean Docker cache
```bash
docker system prune -a
# Warning: Removes all unused images
```

**Solution 3:** Check devcontainer.json syntax
```bash
# Validate JSON
cat .devcontainer/devcontainer.json | python -m json.tool
# Should show formatted JSON, not error
```

### Problem: Extensions not installing

**Solution 1:** Manually install
```
Cmd+Shift+X → Search extension → Install in Container
```

**Solution 2:** Rebuild container
```
Cmd+Shift+P → "Dev Containers: Rebuild Container"
```

**Solution 3:** Check extension IDs are correct
```json
// RIGHT
"ms-python.python"

// WRONG
"Python"  // Must be publisher.extension format
```

### Problem: Services not accessible

**Solution 1:** Check port forwarding
```bash
# In DevContainer terminal
curl http://localhost:8000/health
# Should return JSON, not "connection refused"
```

**Solution 2:** Verify services are running
```bash
docker compose ps
# All services should show "Up"
```

**Solution 3:** Check forwardPorts in devcontainer.json
```json
"forwardPorts": [8000, 6379, 7474, 7687, 2113, 1113]
// Ensure your port is listed
```

### Problem: Environment variables not loaded

**Solution 1:** Verify file exists
```bash
ls -la .devcontainer/devcontainer.env
# Should show file, not "No such file"
```

**Solution 2:** Check file is referenced
```bash
# In devcontainer.json, ensure environment is configured
# (Our setup loads it automatically via docker-compose.yml)
```

**Solution 3:** Rebuild container
```
Cmd+Shift+P → "Dev Containers: Rebuild Container"
```

### Problem: "postStartCommand" fails

**Solution:** Check command in devcontainer.json
```json
// Current command
"postStartCommand": "sleep 20 && echo 'All services ready'"

// If services need more time:
"postStartCommand": "sleep 30 && echo 'All services ready'"
```

---

## DevContainer vs Local Development

### When to Use DevContainer

✅ **Use DevContainer when:**
- You want a fully isolated environment
- You're switching between projects frequently
- You want automatic service startup
- You're onboarding and want "one-click" setup
- You're using Cursor IDE (great DevContainer support)

❌ **Don't use DevContainer when:**
- You need maximum performance (native is faster)
- You're debugging Docker itself
- You prefer manual service control
- You have limited RAM (<8GB)

### Switching Between DevContainer and Local

**From DevContainer to Local:**
```
Cmd+Shift+P → "Dev Containers: Reopen Folder Locally"
```

**From Local to DevContainer:**
```
Cmd+Shift+P → "Dev Containers: Reopen in Container"
```

**Your code persists** - files are synced between container and Mac.

---

## Advanced Configuration

### Using devcontainer.json Features

**Add Docker-in-Docker:**
```json
"features": {
  "ghcr.io/devcontainers/features/docker-in-docker:2": {},
  "git": "latest"
}
```
*Already included - allows running Docker commands inside container*

**Add Node.js:**
```json
"features": {
  "ghcr.io/devcontainers/features/node:1": {
    "version": "18"
  }
}
```

**Add GitHub CLI:**
```json
"features": {
  "ghcr.io/devcontainers/features/github-cli:1": {}
}
```

### Custom Initialization Scripts

**Add to devcontainer.json:**
```json
"postCreateCommand": "pip install -r requirements.txt",
"postStartCommand": "sleep 20 && make db-up && echo 'Ready!'",
"postAttachCommand": "echo 'Welcome to Interview Analyzer Dev!'"
```

**Timing:**
- **postCreateCommand:** Runs once when container is first created
- **postStartCommand:** Runs every time container starts
- **postAttachCommand:** Runs when you attach to the container

### Remote User Configuration

```json
"remoteUser": "root"
```

**Why root?**
- Simplifies file permissions
- Allows installing packages globally
- Standard for development containers

**To change to non-root:**
```json
"remoteUser": "vscode",
"containerUser": "vscode"
```

---

## Quick Reference

### Essential Commands (in DevContainer Terminal)

```bash
# Check services status
docker compose ps

# Start specific service
docker compose up [service-name] -d

# View logs
docker compose logs [service-name] -f

# Run tests
make test

# Run linting
make lint

# Run pipeline
make run

# Check Python environment
which python  # Should show /usr/local/bin/python3
pip list      # Show installed packages
```

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| `devcontainer.json` | `.devcontainer/` | Main DevContainer config |
| `devcontainer.env` | `.devcontainer/` | Actual environment variables (NEVER commit) |
| `devcontainer.json.example` | `docs/onboarding/` | Template for reference (safe to commit) |
| `devcontainer.env.example` | `docs/onboarding/` | Template for creating devcontainer.env |

### Port Reference

| Port | Service | URL |
|------|---------|-----|
| 8000 | FastAPI | http://localhost:8000/docs |
| 6379 | Redis | (internal) |
| 7474 | Neo4j Browser (main) | http://localhost:7474 |
| 7687 | Neo4j Bolt (main) | bolt://localhost:7687 |
| 7475 | Neo4j Browser (test) | http://localhost:7475 |
| 7688 | Neo4j Bolt (test) | bolt://localhost:7688 |
| 2113 | EventStoreDB UI | http://localhost:2113 |
| 1113 | EventStoreDB gRPC | (internal) |

---

## Next Steps

1. ✅ Ensure `.devcontainer/devcontainer.env` exists and has your API keys
2. ✅ Open project in DevContainer: `cursor .` → "Reopen in Container"
3. ✅ Wait for services to start (check `docker compose ps`)
4. ✅ Start coding!

**For more details:**
- **Initial setup:** See [02-initial-setup.md](./02-initial-setup.md)
- **Running services:** See [03-running-the-system.md](./03-running-the-system.md)
- **Daily workflow:** See [05-development-workflow.md](./05-development-workflow.md)

---

**Have questions?** Check [06-troubleshooting.md](./06-troubleshooting.md) for common issues.

