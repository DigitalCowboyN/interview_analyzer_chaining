# Development Workflow

This guide covers daily development tasks, commands, and best practices for working with the Interview Analyzer codebase.

**Prerequisites:** Complete [04-architecture-overview.md](./04-architecture-overview.md) first.

---

## Opening the Project in Cursor

### Start Cursor with Project

```bash
cd ~/workspace/interview_analyzer_chaining
cursor .
```

**If you used a different workspace directory:**
```bash
# For ~/Developer
cd ~/Developer/interview_analyzer_chaining
cursor .

# For ~/Projects
cd ~/Projects/interview_analyzer_chaining
cursor .
```

This opens Cursor with the project root as the workspace.

### Recommended Cursor Settings

1. **Format on Save:**
   - Open Settings: `Cmd+,`
   - Search: "format on save"
   - ✅ Enable "Format On Save"

2. **Python Interpreter:**
   - Open Command Palette: `Cmd+Shift+P`
   - Type: "Python: Select Interpreter"
   - Choose: Python 3.10 (from Docker container if available)

3. **Line Length:**
   - The project uses 120 character line length (configured in `.flake8`)

### Cursor AI Configuration

This project includes a `.cursorrules` file that configures Cursor's AI assistant with project-specific context. This makes AI suggestions **much better** because it understands:

- Event sourcing architecture
- Neo4j transaction handling
- Type hint requirements
- Testing patterns
- Environment detection

**Using Cursor AI:**
- Inline edits: `Cmd+K` (AI suggests code changes)
- AI chat: `Cmd+L` (Ask questions about the codebase)
- Auto-complete: Just start typing (AI completes with project patterns)

**For more details:** See [07-hidden-files-and-cursor.md](./07-hidden-files-and-cursor.md)

---

## Running Tests

### Full Test Suite

Run all 673 tests with coverage:

```bash
make test
```

Or directly with Docker Compose:

```bash
docker compose run --rm app pytest tests/ --cov=src --cov-report=html --cov-report=term
```

**Expected Duration:** 3-4 minutes

**Expected Output:**
```
====== test session starts ======
...
668 passed, 5 failed, 10 skipped in 220.94s
Coverage: 73%
Coverage HTML written to dir htmlcov
```

### View Coverage Report

After running tests:

```bash
open htmlcov/index.html
```

This opens the interactive coverage report in your browser.

### Run Unit Tests Only

Skip slower integration tests:

```bash
make test-unit
```

Or:

```bash
docker compose run --rm app pytest tests/ -m "not integration" --cov=src --cov-report=term
```

**Expected Duration:** ~30 seconds

### Run Integration Tests Only

```bash
make test-integration
```

Or:

```bash
docker compose run --rm app pytest tests/ -m integration
```

**Expected Duration:** ~3 minutes

### Run Specific Test File

```bash
docker compose run --rm app pytest tests/api/test_analysis_api.py -v
```

**Flags:**
- `-v` - Verbose output (shows each test name)
- `-vv` - Very verbose (shows diff on failures)
- `-x` - Stop on first failure
- `-k "test_name"` - Run tests matching pattern

### Run Single Test

```bash
docker compose run --rm app pytest tests/api/test_analysis_api.py::TestAnalysisAPI::test_get_analysis_success -v
```

### Watch Mode (Run Tests on File Change)

**Note:** `pytest-watch` is not included in requirements.txt. You can add it for development:

```bash
# Add to requirements.txt, then rebuild
echo "pytest-watch==4.2.0" >> requirements.txt
docker compose build app

# Then use watch mode
docker compose run --rm app pytest-watch tests/
```

**Or use `ptw` (short form):**
```bash
docker compose run --rm app ptw tests/
```

**Alternative without pytest-watch:**
Use your IDE's built-in test runner or manually rerun tests after changes.

---

## Code Quality

### Run Linter (flake8)

Check code style and potential issues:

```bash
make lint
```

Or:

```bash
docker compose run --rm app flake8 src tests
```

**Configuration:** `.flake8` file
- Max line length: 120
- Ignores: E203, W503, F401 (conflicts with Black formatter)

**See:** [07-hidden-files-and-cursor.md](./07-hidden-files-and-cursor.md#flake8) for detailed Flake8 configuration.

**Expected Output (no issues):**
```
(no output means success)
```

**Example Issues:**
```
src/pipeline.py:123:1: E302 expected 2 blank lines, found 1
src/api/routers/files.py:45:121: E501 line too long (125 > 120 characters)
```

### Run Formatter (black)

Auto-format code to project standards:

```bash
make format
```

Or:

```bash
docker compose run --rm app black src tests
```

**Expected Output:**
```
reformatted src/api/routers/files.py
reformatted tests/api/test_files_api.py
All done! ✨ 🍰 ✨
2 files reformatted, 128 files left unchanged.
```

### Pre-Commit Workflow

Before committing code:

```bash
make lint       # Check for issues
make format     # Auto-fix formatting
make test-unit  # Quick test verification
```

---

## Running the Pipeline

### Process Sample File

```bash
make run-pipeline
```

Or:

```bash
docker compose run --rm app python src/main.py --run-pipeline
```

### Process Specific File

```bash
# Using the sample file
docker compose run --rm app python src/main.py --run-pipeline --input_dir data/input --file_name GMT20231026-210203_Recording.txt
```

**Or if you've added your own file:**
```bash
docker compose run --rm app python src/main.py --run-pipeline --input_dir data/input --file_name your_interview.txt
```

### Process All Files in Directory

```bash
docker compose run --rm app python src/main.py --run-pipeline --input_dir data/input
```

### Pipeline with Custom Output

```bash
docker compose run --rm app python src/main.py --run-pipeline \
  --input_dir data/input \
  --output_dir data/custom_output \
  --map_dir data/custom_maps
```

---

## Using the API

### Start API Server

```bash
make run
```

Or:

```bash
docker compose up -d app
```

Access at: http://localhost:8000/docs

### Test Endpoints with curl

**Health Check:**
```bash
curl http://localhost:8000/
```

**List Analysis Files:**
```bash
curl http://localhost:8000/files/ | python -m json.tool
```

**Get File Content:**
```bash
curl "http://localhost:8000/files/GMT20231026-210203_Recording_analysis.jsonl" | python -m json.tool
```

**Get Specific Sentence:**
```bash
curl "http://localhost:8000/files/GMT20231026-210203_Recording_analysis.jsonl/sentences/1" | python -m json.tool
```

**Trigger Analysis (Background Task):**
```bash
curl -X POST "http://localhost:8000/analysis/" \
  -H "Content-Type: application/json" \
  -d '{"input_filename": "GMT20231026-210203_Recording.txt"}'
```

### Test with Swagger UI

1. Open: http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"
6. View response below

---

## Database Management

### Neo4j (Main Database)

**Access Browser:**
```bash
open http://localhost:7474
```

**Login:**
- Username: `neo4j`
- Password: [from your `.env` file's `NEO4J_PASSWORD`]

**Clear All Data:**
```cypher
MATCH (n)
DETACH DELETE n
```

**Count Sentences:**
```cypher
MATCH (s:Sentence)
RETURN count(s) AS total
```

**View Recent Sentences:**
```cypher
MATCH (s:Sentence)
RETURN s.text, s.sequence_order, s.filename
ORDER BY s.sequence_order DESC
LIMIT 10
```

### Neo4j Test Database

**Start Test Database:**
```bash
make db-test-up
```

**Stop Test Database:**
```bash
make db-test-down
```

**Clear Test Data:**
```bash
make db-test-clear
```

Or manually:
```bash
docker compose exec neo4j-test cypher-shell -u neo4j -p testpassword -d neo4j "MATCH (n) DETACH DELETE n;"
```

### EventStoreDB

**Start EventStore:**
```bash
make eventstore-up
```

**Stop EventStore:**
```bash
make eventstore-down
```

**Check Health:**
```bash
make eventstore-health
```

Or:
```bash
curl http://localhost:2113/health/live
```

**View Logs:**
```bash
make eventstore-logs
```

**Access UI:**
```bash
open http://localhost:2113
```

**Clear All Data (⚠️ Destructive):**
```bash
make eventstore-clear
```

### Projection Service

**Start Projection Service:**
```bash
make projection-up
```

**Stop Projection Service:**
```bash
make projection-down
```

**View Logs:**
```bash
make projection-logs
```

**Check Status:**
```bash
make projection-status
```

**Restart (to replay events):**
```bash
make projection-restart
```

### Event Sourcing System Management

**Start Everything (EventStore + Projections):**
```bash
make es-up
```

**Stop Everything:**
```bash
make es-down
```

**Check Status:**
```bash
make es-status
```

**View Logs:**
```bash
make es-logs
```

---

## Viewing Logs

### All Services

```bash
docker compose logs -f
```

Press `Ctrl+C` to stop (services keep running).

### Specific Service

```bash
docker compose logs -f app
docker compose logs -f worker
docker compose logs -f eventstore
docker compose logs -f projection-service
```

### Recent Logs

```bash
docker compose logs --tail=50 app
```

### Search Logs

```bash
docker compose logs app | grep ERROR
docker compose logs projection-service | grep "Processing event"
```

---

## Makefile Quick Reference

### Building

```bash
make build              # Build Docker images
```

### Running

```bash
make run                # Start API server
make run-pipeline       # Run pipeline processing
make run-api            # Same as 'make run'
```

### Testing

```bash
make test               # Full test suite with coverage
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-cov           # Tests with HTML coverage report
make test-eventstore    # EventStore-specific tests
make test-e2e           # End-to-end tests
make test-projections   # Projection tests
```

### Code Quality

```bash
make lint               # Run flake8
make format             # Run black formatter
make clean-coverage     # Remove coverage reports
```

### Database Management

```bash
make db-up              # Start neo4j and redis
make db-down            # Stop neo4j and redis
make db-test-up         # Start test neo4j
make db-test-down       # Stop test neo4j
make db-test-clear      # Clear test database
```

### EventStore Management

```bash
make eventstore-up      # Start EventStoreDB
make eventstore-down    # Stop EventStoreDB
make eventstore-health  # Check health
make eventstore-logs    # View logs
make eventstore-restart # Restart EventStoreDB
make eventstore-clear   # Delete all data (⚠️)
```

### Projection Management

```bash
make projection-up      # Start projection service
make projection-down    # Stop projection service
make projection-logs    # View logs
make projection-restart # Restart projection service
make projection-status  # Check status
```

### Event Sourcing System

```bash
make es-up              # Start eventstore + projections
make es-down            # Stop eventstore + projections
make es-status          # Check status of both
make es-logs            # View logs for both
```

### Cleanup

```bash
make clean              # Clean Python cache files
make clean-explicit     # Stop containers and remove volumes
```

---

## Hot Reload / Development Mode

### API Auto-Reload

The FastAPI app runs with `--reload` flag by default in docker-compose.yml.

**When you edit Python files in `src/`:**
- Changes are detected automatically
- Server reloads within 1-2 seconds
- No need to restart containers

**View reload logs:**
```bash
docker compose logs -f app | grep "Reloading"
```

### Worker Auto-Reload

The Celery worker **does not** auto-reload.

**After editing worker code:**
```bash
docker compose restart worker
```

### Projection Service Auto-Reload

The projection service **does not** auto-reload.

**After editing projection code:**
```bash
make projection-restart
```

---

## Adding Sample Data

### Create New Transcript File

1. Create a `.txt` file in `data/input/`:
   ```bash
   touch data/input/my_interview.txt
   ```

2. Open in Cursor:
   ```bash
   cursor data/input/my_interview.txt
   ```

3. Add interview transcript (plain text, no special formatting needed)

4. Save and close

5. Process the file:
   ```bash
   make run-pipeline
   ```

### Sample Data Format

**Input files should be plain text:**

```
Speaker: Hello, welcome to the interview.
Interviewer: Thank you for having me.
Speaker: Can you tell me about your experience with Python?
Interviewer: I've been using Python for 5 years, primarily for data science.
...
```

**No special structure required** - the pipeline will segment it into sentences.

---

## Debugging

### Debug API Endpoint

1. Add breakpoint in code using `import pdb; pdb.set_trace()`

2. Attach to running container:
   ```bash
   docker attach interview_analyzer_app
   ```

3. Make API request that hits your breakpoint

4. Debug interactively

5. Detach: `Ctrl+P`, then `Ctrl+Q`

### Debug Pipeline

```bash
docker compose run --rm app python -m pdb src/main.py --run-pipeline
```

### Debug Tests

```bash
docker compose run --rm app pytest tests/api/test_files_api.py --pdb
```

Drops into debugger on test failure.

### View Environment Variables

```bash
docker compose exec app env | grep -E "OPENAI|NEO4J|PYTHON"
```

### Check Python Version

```bash
docker compose run --rm app python --version
```

### List Installed Packages

```bash
docker compose run --rm app pip list
```

---

## Git Workflow

### Create Feature Branch

```bash
git checkout -b feature/my-new-feature
```

### Check Status

```bash
git status
```

### Stage Changes

```bash
git add src/api/routers/my_router.py
git add tests/api/test_my_router.py
```

### Commit

```bash
git commit -m "Add new router for X functionality"
```

### Push Branch

```bash
git push origin feature/my-new-feature
```

### Before Committing - Checklist

- [ ] Run `make lint` - no errors
- [ ] Run `make format` - code formatted
- [ ] Run `make test-unit` - tests pass
- [ ] Add new tests for new functionality
- [ ] Update docstrings
- [ ] Check `.env` not staged (`git status` should not show `.env`)

---

## Common Development Tasks

### Add New API Endpoint

1. Create route function in `src/api/routers/`
2. Add Pydantic schemas in `src/api/schemas.py` if needed
3. Register router in `src/main.py` if new router file
4. Write tests in `tests/api/`
5. Test manually via Swagger UI
6. Run `make lint` and `make format`

### Add New Event Type

1. Define event class in `src/events/sentence_events.py` or `interview_events.py`
2. Update aggregate in `src/events/aggregates.py` to handle event
3. Create projection handler in `src/projections/handlers/`
4. Register handler in projection service
5. Write tests in `tests/events/` and `tests/projections/`
6. Test with: `make test-projections`

### Modify LLM Prompts

1. Edit `prompts/task_prompts.yaml` or `prompts/domain_prompts.yaml`
2. No code changes needed (prompts loaded at runtime)
3. Restart services to pick up changes:
   ```bash
   docker compose restart app worker
   ```
4. Test pipeline with new prompts:
   ```bash
   make run-pipeline
   ```

### Update Dependencies

1. Edit `requirements.txt`
2. Rebuild images:
   ```bash
   make build
   ```
3. Run tests:
   ```bash
   make test
   ```

---

## Performance Tips

### Speed Up Tests

```bash
# Run in parallel (requires pytest-xdist)
docker compose run --rm app pytest tests/ -n auto

# Skip slow tests
docker compose run --rm app pytest tests/ -m "not slow"
```

### Speed Up Pipeline

1. Increase workers in `config.yaml`:
   ```yaml
   pipeline:
     num_analysis_workers: 20  # Increase from 10
   ```

2. Use faster model:
   ```yaml
   openai:
     model_name: "gpt-3.5-turbo"  # Faster than gpt-4
   ```

### Reduce Docker Build Time

```bash
# Use build cache
docker compose build

# Skip cache for fresh build
docker compose build --no-cache
```

---

## What's Next?

You're ready to develop!

Next: [Troubleshooting →](./06-troubleshooting.md)

Reference guide for common issues and their solutions.

