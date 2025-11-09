# Troubleshooting

This guide provides solutions to common issues you may encounter while developing with the Interview Analyzer project.

---

## Table of Contents

- [Docker Issues](#docker-issues)
- [Service Startup Issues](#service-startup-issues)
- [Database Connection Issues](#database-connection-issues)
- [API Issues](#api-issues)
- [Test Failures](#test-failures)
- [Pipeline Issues](#pipeline-issues)
- [Environment Variable Issues](#environment-variable-issues)
- [Port Conflicts](#port-conflicts)
- [Performance Issues](#performance-issues)
- [Data Issues](#data-issues)

---

## Docker Issues

### Docker Desktop Not Running

**Symptom:** `Cannot connect to the Docker daemon`

**Solution:**
1. Open Docker Desktop from Applications
2. Wait for it to fully start (whale icon steady in menu bar)
3. Verify:
   ```bash
   docker ps
   ```

### Docker Out of Space

**Symptom:** `no space left on device`

**Solution 1 - Clean up Docker:**
```bash
# Remove unused containers, networks, images
docker system prune -a

# Remove volumes too (⚠️ deletes data)
docker system prune -a --volumes
```

**Solution 2 - Increase Docker Disk:**
1. Open Docker Desktop
2. Settings → Resources → Disk image size
3. Increase to 100 GB
4. Click "Apply & Restart"

### Docker Build Fails

**Symptom:** `error building image`

**Solution 1 - Rebuild without cache:**
```bash
docker compose build --no-cache app worker
```

**Solution 2 - Check internet connection:**
```bash
# Test pull
docker pull python:3.10.14-slim
```

**Solution 3 - Clear builder cache:**
```bash
docker builder prune -a
```

### Cannot Remove Container

**Symptom:** `container is in use`

**Solution:**
```bash
# Force remove
docker rm -f interview_analyzer_app

# Or stop first, then remove
docker stop interview_analyzer_app
docker rm interview_analyzer_app
```

---

## Service Startup Issues

### EventStoreDB Slow to Start

**Symptom:** `eventstore` stuck on `health: starting` for 2+ minutes

**Expected Behavior:** EventStoreDB takes 60-90 seconds to become healthy on first startup.

**Solution 1 - Wait patiently:**
```bash
# Watch until healthy
watch -n 5 'docker compose ps eventstore'
```

**Solution 2 - Check logs:**
```bash
docker compose logs eventstore | tail -50
```

Look for: `"[INF] HTTP API BOUND ON PORT 2113"`

**Solution 3 - Restart:**
```bash
docker compose restart eventstore
```

**Solution 4 - Fresh start:**
```bash
docker compose down eventstore
docker volume rm interview_analyzer_chaining_eventstore_data
docker compose up -d eventstore
```

### Neo4j Won't Start

**Symptom:** Neo4j container exits immediately

**Solution 1 - Check logs:**
```bash
docker compose logs neo4j
```

**Solution 2 - Reset data:**
```bash
docker compose down neo4j
docker volume rm interview_analyzer_chaining_neo4j_data
docker compose up -d neo4j
```

**Solution 3 - Verify authentication:**
Check that `NEO4J_AUTH` in `docker-compose.yml` matches your `.env`:
```yaml
  NEO4J_AUTH: "neo4j/[your-chosen-password]"
```

### All Services Fail to Start

**Symptom:** Multiple services show errors

**Solution:**
```bash
# Nuclear option: clean everything and rebuild
docker compose down -v
docker system prune -a --volumes
make build
docker compose up -d
```

### Celery Worker Not Processing Tasks

**Symptom:** Tasks queued but not executing

**Solution 1 - Check worker logs:**
```bash
docker compose logs worker | grep -E "ready|Received|Succeeded|Failed"
```

**Solution 2 - Restart worker:**
```bash
docker compose restart worker
```

**Solution 3 - Check Redis connection:**
```bash
docker compose exec worker python -c "from celery import Celery; app = Celery(); print(app.connection().ensure_connection(max_retries=3))"
```

### Projection Service Not Processing Events

**Symptom:** Events in EventStore but Neo4j not updating

**Solution 1 - Check projection logs:**
```bash
make projection-logs
```

Look for: `"Processing event"` or errors

**Solution 2 - Restart projection service:**
```bash
make projection-restart
```

**Solution 3 - Verify EventStore connection:**
```bash
docker compose logs projection-service | grep -E "Connected|Subscription"
```

---

## Database Connection Issues

### Neo4j "Connection Refused"

**Symptom:** `ServiceUnavailable: Failed to establish connection`

**Solution 1 - Wait for startup:**
```bash
# Neo4j takes ~15-30 seconds
docker compose logs neo4j | grep "Started"
```

**Solution 2 - Check Neo4j is running:**
```bash
docker compose ps neo4j
```

Should show: `Up X seconds`

**Solution 3 - Test connection:**
```bash
docker compose exec app python -c "from neo4j import AsyncGraphDatabase; import asyncio; asyncio.run(AsyncGraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', '[YOUR_NEO4J_PASSWORD]')).verify_connectivity())"
```

**Solution 4 - Check password:**
Verify `NEO4J_PASSWORD` in `.env` matches `docker-compose.yml`

### EventStoreDB "Connection Failed"

**Symptom:** `ConnectionError: Failed to connect to EventStore`

**Solution 1 - Verify EventStore is healthy:**
```bash
make eventstore-health
```

**Solution 2 - Check connection string:**
```bash
docker compose exec app python -c "from src.config import config; print(config['event_sourcing'])"
```

Should show: `esdb://eventstore:2113?tls=false`

**Solution 3 - Test connection manually:**
```bash
curl http://localhost:2113/health/live
```

**Solution 4 - Check network:**
```bash
docker network ls | grep interview
docker network inspect interview_analyzer_chaining_interview_net
```

### Redis "Connection Refused"

**Symptom:** `ConnectionError: Error connecting to Redis`

**Solution 1 - Check Redis health:**
```bash
docker compose ps redis
```

Should show: `(healthy)`

**Solution 2 - Test Redis:**
```bash
docker compose exec redis redis-cli ping
```

Should return: `PONG`

**Solution 3 - Restart Redis:**
```bash
docker compose restart redis
```

---

## API Issues

### API Returns 404 Not Found

**Symptom:** Endpoint exists but returns 404

**Solution 1 - Check API is running:**
```bash
curl http://localhost:8000/
```

**Solution 2 - View all routes:**
```bash
docker compose exec app python -c "from src.main import app; for route in app.routes: print(route.path)"
```

**Solution 3 - Check Swagger docs:**
Open http://localhost:8000/docs to see registered endpoints

**Solution 4 - Restart API:**
```bash
docker compose restart app
```

### API Returns 500 Internal Server Error

**Symptom:** Request succeeds but server errors

**Solution 1 - Check app logs:**
```bash
docker compose logs app | tail -50
```

**Solution 2 - Check for missing env vars:**
```bash
docker compose exec app env | grep -E "OPENAI|GEMINI|NEO4J"
```

**Solution 3 - Test database connections:**
```bash
# Neo4j
curl http://localhost:7474

# EventStore
curl http://localhost:2113/health/live
```

### API Slow to Respond

**Symptom:** Requests take 30+ seconds

**Solution 1 - Use background tasks:**
Use `POST /analysis/` which returns immediately and processes in background

**Solution 2 - Check worker is running:**
```bash
docker compose ps worker
```

**Solution 3 - Check OpenAI API rate limits:**
View response headers for rate limit info

**Solution 4 - Increase workers in config.yaml:**
```yaml
pipeline:
  num_analysis_workers: 20  # Increase from 10
```

---

## Test Failures

### Tests Fail with "Connection Refused"

**Symptom:** `ServiceUnavailable` or `ConnectionRefusedError` in tests

**Solution 1 - Start test database:**
```bash
make db-test-up
make eventstore-up
```

**Solution 2 - Wait for services:**
```bash
# Wait 30 seconds
sleep 30

# Run tests
make test
```

**Solution 3 - Check test database:**
```bash
docker compose ps neo4j-test eventstore
```

Both should be `Up`

### Many Tests Fail with Import Errors

**Symptom:** `ImportError: cannot import name 'X' from 'Y'`

**Solution 1 - Rebuild container:**
```bash
make build
```

**Solution 2 - Check requirements installed:**
```bash
docker compose run --rm app pip list | grep -E "fastapi|neo4j|pytest"
```

**Solution 3 - Clear Python cache:**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### Specific Test Always Fails

**Symptom:** One test consistently fails

**Solution 1 - Run test in isolation:**
```bash
docker compose run --rm app pytest tests/path/to/test.py::test_name -vv
```

**Solution 2 - Check test dependencies:**
Look for `@pytest.mark.depends` or fixtures

**Solution 3 - Clear test data:**
```bash
make db-test-clear
```

**Solution 4 - Read test output carefully:**
The error message usually indicates the issue

### Coverage Report Not Generated

**Symptom:** No `htmlcov/` directory

**Solution:**
```bash
# Run with coverage explicitly
docker compose run --rm app pytest tests/ --cov=src --cov-report=html

# Check it was created
ls -la htmlcov/
```

---

## Pipeline Issues

### Pipeline Fails Immediately

**Symptom:** Pipeline exits with error before processing

**Solution 1 - Check input file exists:**
```bash
ls -la data/input/
```

**Solution 2 - Verify file is readable:**
```bash
# For the sample file
cat data/input/GMT20231026-210203_Recording.txt | head -5

# Or for your own file
cat data/input/your_interview.txt | head -5
```

**Solution 3 - Check logs:**
```bash
docker compose logs app | grep ERROR
```

**Solution 4 - Run with debug output:**
```bash
docker compose run --rm app python src/main.py --run-pipeline --log-level DEBUG
```

### Pipeline Fails with OpenAI Error

**Symptom:** `OpenAI API error: Authentication failed`

**Solution 1 - Verify API key:**
```bash
docker compose exec app python -c "import os; print('Key exists:', bool(os.getenv('OPENAI_API_KEY')))"
```

**Solution 2 - Test API key directly:**
```bash
docker compose exec app python -c "import openai; openai.api_key = os.getenv('OPENAI_API_KEY'); print(openai.Model.list())"
```

**Solution 3 - Check billing:**
Visit https://platform.openai.com/account/billing/overview

**Solution 4 - Update .env and restart:**
```bash
# Edit .env with correct key
docker compose restart app worker
```

### Pipeline Produces No Output

**Symptom:** Pipeline completes but no files in `data/output/`

**Solution 1 - Check pipeline logs:**
```bash
docker compose logs app | grep "Results written"
```

**Solution 2 - Verify output directory:**
```bash
ls -la data/output/
docker compose exec app ls -la /workspaces/interview_analyzer_chaining/data/output/
```

**Solution 3 - Check permissions:**
```bash
docker compose exec app touch /workspaces/interview_analyzer_chaining/data/output/test.txt
rm data/output/test.txt
```

**Solution 4 - Check config.yaml paths:**
```bash
grep -A 5 "^paths:" config.yaml
```

---

## Environment Variable Issues

### Env Var Not Loading

**Symptom:** Config shows empty string or default value

**Solution 1 - Check .env file exists:**
```bash
ls -la .env
```

**Solution 2 - Check .env syntax:**
```bash
cat .env
```

Ensure:
- No spaces around `=`
- No quotes around values
- One variable per line

**Solution 3 - Verify .env is in project root:**
```bash
pwd  # Should show project root
ls .env  # Should exist here
```

**Solution 4 - Restart services:**
```bash
docker compose down
docker compose up -d
```

**Solution 5 - Check env vars in container:**
```bash
docker compose exec app env | grep OPENAI_API_KEY
```

### Wrong Env Var Value

**Symptom:** Env var has wrong value

**Solution 1 - Check .env vs docker-compose.yml:**
Variables in `docker-compose.yml` `environment:` section override `.env`

**Solution 2 - Check for duplicates:**
```bash
grep -n "OPENAI_API_KEY" .env
```

Should only appear once

**Solution 3 - Rebuild and restart:**
```bash
docker compose down
docker compose up -d
```

---

## Port Conflicts

### Port Already in Use

**Symptom:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution 1 - Find what's using the port:**
```bash
lsof -i :8000
```

**Solution 2 - Kill the process:**
```bash
# Get PID from lsof output (second column)
# Example: if lsof shows "Python    12345 user...", the PID is 12345
kill -9 12345  # Replace 12345 with your actual PID
```

**Example:**
```bash
# If lsof -i :8000 shows:
# COMMAND    PID   USER
# Python   45678  yourname
# Then run:
kill -9 45678
```

**Solution 3 - Use different port:**
Edit `docker-compose.yml`:
```yaml
app:
  ports:
    - "8001:8000"  # Change 8000 to 8001 (or any free port)
```

Then access at http://localhost:8001

**Common Ports:**
- 8000 - FastAPI
- 7474 - Neo4j HTTP
- 7687 - Neo4j Bolt
- 7475 - Neo4j Test HTTP
- 7688 - Neo4j Test Bolt
- 2113 - EventStoreDB
- 6379 - Redis

---

## Performance Issues

### Tests Take Too Long

**Symptom:** Test suite takes 10+ minutes

**Solution 1 - Run unit tests only:**
```bash
make test-unit  # ~30 seconds vs 3-4 minutes
```

**Solution 2 - Run in parallel:**
```bash
docker compose run --rm app pytest tests/ -n auto
```

**Solution 3 - Skip slow tests:**
```bash
docker compose run --rm app pytest tests/ -m "not slow"
```

**Solution 4 - Increase Docker resources:**
Docker Desktop → Settings → Resources → CPUs: 6, Memory: 12 GB

### Pipeline Processing Slow

**Symptom:** Takes 10+ minutes for small file

**Solution 1 - Check OpenAI rate limits:**
```bash
docker compose logs app | grep "rate_limit"
```

**Solution 2 - Increase workers:**
Edit `config.yaml`:
```yaml
pipeline:
  num_analysis_workers: 20  # Up from 10
```

**Solution 3 - Use faster model:**
```yaml
openai:
  model_name: "gpt-3.5-turbo"  # Instead of gpt-4
```

**Solution 4 - Check network:**
```bash
# Test latency to OpenAI
time curl -I https://api.openai.com
```

### Docker Desktop Using Too Much CPU/Memory

**Symptom:** Mac fan loud, Docker using lots of resources

**Solution 1 - Stop unused services:**
```bash
docker compose stop projection-service worker
```

**Solution 2 - Limit Docker resources:**
Docker Desktop → Settings → Resources:
- CPUs: 4 (from 6)
- Memory: 8 GB (from 12 GB)

**Solution 3 - Stop services when not in use:**
```bash
docker compose down
```

---

## Data Issues

### Neo4j Graph Empty After Pipeline

**Symptom:** Pipeline completes but no nodes in Neo4j

**Solution 1 - Check projection service:**
```bash
make projection-logs | grep -E "Processing event|Error"
```

**Solution 2 - Verify events in EventStore:**
Open http://localhost:2113 → Stream Browser

**Solution 3 - Restart projection service:**
```bash
make projection-restart
```

**Solution 4 - Check Neo4j connection from projection:**
```bash
docker compose logs projection-service | grep "Neo4j"
```

### Duplicate Data in Neo4j

**Symptom:** Same sentences appearing multiple times

**Solution 1 - Clear and reprocess:**
```bash
# Clear Neo4j
# In Neo4j Browser: MATCH (n) DETACH DELETE n

# Clear EventStore
make eventstore-clear

# Restart projection
make projection-restart

# Rerun pipeline
make run-pipeline
```

**Solution 2 - Check event versions:**
```cypher
MATCH (s:Sentence {sentence_id: 1})
RETURN s.event_version
```

Should have only one node per sentence_id

### Analysis Output Missing Fields

**Symptom:** `.jsonl` file has incomplete data

**Solution 1 - Check OpenAI response:**
```bash
docker compose logs app | grep "OpenAI response"
```

**Solution 2 - Verify prompts:**
```bash
cat prompts/task_prompts.yaml | head -50
```

**Solution 3 - Check Pydantic validation:**
```bash
docker compose logs app | grep "ValidationError"
```

---

## Clean Slate Recovery

If everything is broken and you want to start fresh:

### Complete Reset

```bash
# 1. Stop everything
docker compose down -v

# 2. Remove all project containers (if any exist)
docker ps -a | grep interview_analyzer | awk '{print $1}' | xargs -r docker rm -f

# 3. Remove all project volumes (if any exist)
docker volume ls | grep interview_analyzer | awk '{print $2}' | xargs -r docker volume rm

# 4. Remove images (will fail gracefully if they don't exist)
docker rmi interview_analyzer_chaining-app interview_analyzer_chaining-worker 2>/dev/null || true

# 5. Clean Docker system
docker system prune -a --volumes

# 6. Rebuild from scratch
cd ~/workspace/interview_analyzer_chaining
make build

# 7. Start services
docker compose up -d

# 8. Wait for EventStore (90 seconds)
sleep 90

# 9. Run tests
make test
```

### Reset Only Data (Keep Images)

```bash
# Stop services
docker compose down -v

# Start fresh
docker compose up -d

# Wait for services
sleep 30

# Verify
make test-unit
```

---

## Getting Help

### Collect Diagnostic Information

When asking for help, provide:

```bash
# System info
docker version
docker compose version
uname -a

# Service status
docker compose ps

# Recent logs
docker compose logs --tail=100 > logs.txt

# Environment (sanitized)
docker compose exec app env | grep -v -E "API_KEY|PASSWORD" > env.txt
```

### Enable Debug Logging

Edit `config.yaml`:
```yaml
logging:
  level: DEBUG  # Change from INFO
```

Restart services:
```bash
docker compose restart app worker projection-service
```

### Check Documentation

- Main README: `README.md`
- Architecture docs: `docs/onboarding/04-architecture-overview.md`
- Test documentation: Files in `docs/`

---

## Still Stuck?

1. **Re-read the guides:**
   - [Prerequisites](./01-prerequisites.md)
   - [Initial Setup](./02-initial-setup.md)
   - [Running the System](./03-running-the-system.md)

2. **Check test output carefully:**
   Error messages usually explain the issue

3. **Try the clean slate recovery**

4. **Ask your team:**
   - Contact your team lead for specific channel/contact info
   - Provide diagnostic info collected above
   - Include your OS version: `sw_vers` (macOS)
   - Include exact error messages, not paraphrased

---

[← Back to Development Workflow](./05-development-workflow.md) | [↑ Back to Onboarding Home](./00-README.md)

