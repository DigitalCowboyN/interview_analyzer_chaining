# Running the System

This guide walks you through starting all services, verifying they're working, and running your first pipeline test.

**Prerequisites:** Complete [02-initial-setup.md](./02-initial-setup.md) first.

---

## 1. Start All Services

### Start Docker Desktop

1. Open **Docker Desktop** from Applications
2. Wait for Docker to fully start (whale icon in menu bar should be steady, not animated)
3. Verify it's running:
   ```bash
   docker ps
   ```
   **Expected output:** Headers showing container columns (no containers yet)

### Navigate to Project

```bash
cd ~/workspace/interview_analyzer_chaining
```

Replace `~/workspace/` with your actual workspace path if different.

### Start Services

```bash
docker compose up -d
```

**What `-d` means:** Detached mode (runs in background)

### Monitor Startup

Watch the services start:

```bash
docker compose ps
```

**Initial output (first 30 seconds):**
```
NAME                                          STATUS
interview_analyzer_app                        Up 8 seconds
interview_analyzer_eventstore                 Up 15 seconds (health: starting)
interview_analyzer_neo4j_test                 Up 12 seconds
interview_analyzer_projection_service         Up 6 seconds
interview_analyzer_redis                      Up 18 seconds (healthy)
interview_analyzer_worker                     Up 10 seconds
neo4j                                         Up 14 seconds
```

**Typical startup times:**
- Redis: 3-5 seconds to healthy
- Neo4j (main & test): 15-30 seconds
- FastAPI app: 5-10 seconds
- Worker: 8-12 seconds
- Projection service: 10-15 seconds
- EventStoreDB: 60-120 seconds to healthy (be patient!)

### Wait for EventStoreDB

**Important:** EventStoreDB takes 60-120 seconds to become healthy on first startup.

Check status repeatedly.

**Option 1 - Manual checking (recommended):**
```bash
# Run this command every 15 seconds until you see "(healthy)"
docker compose ps eventstore
```

**Option 2 - Auto-refresh (requires watch command):**
```bash
# Install watch first (if not already installed)
brew install watch

# Then monitor
watch -n 10 'docker compose ps eventstore'
```

Press `Ctrl+C` when you see:
```
interview_analyzer_eventstore    Up 2 minutes (healthy)
```

**Option 3 - Loop in terminal (Mac built-in):**
```bash
while true; do clear; docker compose ps eventstore; sleep 10; done
```

Press `Ctrl+C` to stop when you see `(healthy)`.

---

## 2. Verify Each Service

### 2.1 FastAPI Application

**Check Status:**
```bash
docker compose ps app
```

**Expected:** `Up X seconds`

**Access Swagger UI:**
1. Open browser: http://localhost:8000/docs
2. You should see **"Interview Analyzer API"** documentation

**Test Health Endpoint:**
```bash
curl http://localhost:8000/
```

**Expected output:**
```json
{"message":"Welcome to the Interview Analyzer API"}
```

### 2.2 Redis (Message Broker)

**Check Status:**
```bash
docker compose ps redis
```

**Expected:** `Up X seconds (healthy)`

**Test Connection:**
```bash
docker compose exec redis redis-cli ping
```

**Expected output:** `PONG`

### 2.3 Neo4j (Main Database)

**Check Status:**
```bash
docker compose ps neo4j
```

**Expected:** `Up X seconds`

**Access Browser:**
1. Open: http://localhost:7474
2. You'll see Neo4j Browser login screen

**Login:**
- **Connect URL:** `bolt://localhost:7687`
- **Username:** `neo4j`
- **Password:** [from your `.env` file's `NEO4J_PASSWORD`]
- Click **"Connect"**

**Run Test Query:**

In the Neo4j Browser command input (top of page), type:
```cypher
RETURN "Connection successful!" AS message
```

Press **Enter** or click the play button.

**Expected:** You should see a result showing "Connection successful!"

### 2.4 Neo4j Test Database

**Check Status:**
```bash
docker compose ps neo4j-test
```

**Expected:** `Up X seconds`

**Access Browser:**
1. Open: http://localhost:7475
2. Login with:
   - **Connect URL:** `bolt://localhost:7688`
   - **Username:** `neo4j`
   - **Password:** `testpassword`

**Note:** This database is used exclusively for running tests.

### 2.5 EventStoreDB

**Check Status:**
```bash
docker compose ps eventstore
```

**Expected:** `Up X minutes (healthy)`

**Access UI:**
1. Open: http://localhost:2113
2. You should see EventStoreDB web interface
3. No login required (insecure mode for local dev)

**View Streams:**
- Click "Stream Browser" in the top menu
- Should see an empty stream list (or system streams starting with `$`)

### 2.6 Celery Worker

**Check Status:**
```bash
docker compose ps worker
```

**Expected:** `Up X seconds`

**View Logs:**
```bash
docker compose logs worker | tail -20
```

**Expected output should include:**
```
celery@<hostname> ready.
```

### 2.7 Projection Service

**Check Status:**
```bash
docker compose ps projection-service
```

**Expected:** `Up X seconds`

**View Logs:**
```bash
docker compose logs projection-service | tail -20
```

**Expected output should include:**
```
Starting projection service...
Subscribed to $all stream
```

---

## 3. Quick Health Check Summary

Run this command to see all services at once:

```bash
docker compose ps
```

**Healthy system should show:**
```
NAME                                    STATUS
interview_analyzer_app                  Up X minutes
interview_analyzer_eventstore           Up X minutes (healthy)
interview_analyzer_neo4j_test           Up X minutes
interview_analyzer_projection_service   Up X minutes
interview_analyzer_redis                Up X minutes (healthy)
interview_analyzer_worker               Up X minutes
neo4j                                   Up X minutes
```

**All services should be "Up"**. Two services have health checks:
- `redis` → should show `(healthy)`
- `eventstore` → should show `(healthy)`

---

## 4. Run Your First Pipeline Test

Now let's verify the entire system works by processing a sample file.

### 4.1 Check Sample Data

Verify sample transcript exists:

```bash
ls -lh data/input/
```

**Expected:** You should see `GMT20231026-210203_Recording.txt`

View the first few lines:
```bash
head -5 data/input/GMT20231026-210203_Recording.txt
```

### 4.2 Run Pipeline

Process the sample file:

```bash
docker compose run --rm app python src/main.py --run-pipeline
```

**What this does:**
1. Reads the text file
2. Segments it into sentences
3. Analyzes each sentence using OpenAI
4. Saves results to Neo4j and EventStoreDB
5. Generates analysis output

**Expected output:**
```
INFO: Pipeline starting...
INFO: Processing file: GMT20231026-210203_Recording.txt
INFO: Segmented 147 sentences
INFO: Running analysis with 10 workers...
INFO: Analysis complete
INFO: Results written to data/output/GMT20231026-210203_Recording_analysis.jsonl
INFO: Pipeline complete in 186.43 seconds
```

**Duration:** ~2-5 minutes for the sample file (147 sentences), depending on API response times.

**Note:** OpenAI API calls are rate-limited, so actual time may vary. Typical processing: 1-2 seconds per sentence.

### 4.3 Verify Output Files

**Check analysis output:**
```bash
ls -lh data/output/
```

**Expected:** `GMT20231026-210203_Recording_analysis.jsonl` (newly created or updated, typically 50-200KB)

**View first result:**
```bash
head -1 data/output/GMT20231026-210203_Recording_analysis.jsonl | python -m json.tool
```

**Expected:** JSON object with fields like:
```json
{
  "sentence_id": 1,
  "sentence": "Hello, welcome to today's interview.",
  "sequence_order": 1,
  "filename": "GMT20231026-210203_Recording.txt",
  "function_type": "greeting",
  "structure_type": "simple",
  "purpose": "establish_rapport",
  "keywords": ["hello", "welcome", "interview"],
  "topics": ["introduction"],
  "domain_keywords": []
}
```

**Check map file:**
```bash
ls -lh data/maps/
```

**Expected:** `GMT20231026-210203_Recording_map.jsonl`

### 4.4 Verify Neo4j Data

1. Open Neo4j Browser: http://localhost:7474
2. Run this query:
   ```cypher
   MATCH (s:Sentence)
   RETURN count(s) AS total_sentences
   ```

**Expected:** `total_sentences` should be greater than 0

3. View a sample sentence:
   ```cypher
   MATCH (s:Sentence)
   RETURN s
   LIMIT 1
   ```

**Expected:** A node with properties like `text`, `sentence_id`, `filename`

### 4.5 Verify EventStoreDB Events

1. Open EventStoreDB UI: http://localhost:2113
2. Click "Stream Browser"
3. You should see streams like:
   - `Interview-<uuid>`
   - `Sentence-<uuid>`

4. Click on a stream to view events inside

**Expected:** Events like `InterviewCreated`, `SentenceCreated`, `AnalysisGenerated`

---

## 5. Test the API

### 5.1 List Analysis Files

```bash
curl http://localhost:8000/files/ | python -m json.tool
```

**Expected output:**
```json
{
  "files": [
    "GMT20231026-210203_Recording_analysis.jsonl"
  ]
}
```

### 5.2 Get Specific Analysis

```bash
curl "http://localhost:8000/files/GMT20231026-210203_Recording_analysis.jsonl/sentences/1" | python -m json.tool
```

**Expected:** JSON object with analysis for sentence ID 1

### 5.3 Access API Documentation

1. Open: http://localhost:8000/docs
2. Try the "GET /files/" endpoint:
   - Click on "GET /files/"
   - Click "Try it out"
   - Click "Execute"
   - See the response below

---

## 6. Stopping Services

### Stop All Services

```bash
docker compose down
```

**What this does:**
- Stops all containers
- Removes containers
- Preserves data volumes (databases keep their data)

### Stop and Remove All Data

**⚠️ Warning:** This deletes all database data, including Neo4j graphs and EventStoreDB events!

```bash
docker compose down -v
```

**When to use:**
- Fresh start needed
- Cleaning up after testing
- Troubleshooting persistent issues

---

## 7. Restarting Services

### Start Again

```bash
docker compose up -d
```

**Note:** Startup is faster after the first time (services don't need initialization)

### View Logs

Watch all logs in real-time:
```bash
docker compose logs -f
```

Press `Ctrl+C` to stop watching (services keep running).

View specific service logs:
```bash
docker compose logs -f app
docker compose logs -f worker
docker compose logs -f eventstore
```

---

## Startup Checklist

Verify your system is ready:

- [ ] All 7 services show "Up" status
- [ ] Redis shows `(healthy)`
- [ ] EventStoreDB shows `(healthy)` (took 60-90 seconds)
- [ ] FastAPI responds at http://localhost:8000
- [ ] Neo4j Browser accessible at http://localhost:7474 with login
- [ ] Neo4j Test Browser accessible at http://localhost:7475
- [ ] EventStoreDB UI accessible at http://localhost:2113
- [ ] Sample pipeline ran successfully
- [ ] Output files created in `data/output/`
- [ ] Neo4j contains sentence nodes
- [ ] EventStoreDB contains event streams
- [ ] API returns analysis results

---

## Troubleshooting

### Service won't start

```bash
# View logs for specific service
docker compose logs <service-name>

# Examples:
docker compose logs app
docker compose logs eventstore
```

### Port already in use

**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:**
```bash
# Find what's using the port
lsof -i :8000

# Kill the process (replace <PID> with actual process ID)
kill -9 <PID>

# Or change port in docker-compose.yml
# Change "8000:8000" to "8001:8000" for app service
```

### EventStoreDB stuck on "health: starting"

**This is normal for the first 60-90 seconds.**

Check logs:
```bash
docker compose logs eventstore
```

If stuck for 5+ minutes:
```bash
docker compose restart eventstore
```

### Pipeline fails with "API key not found"

Verify `.env` file:
```bash
cat .env | grep OPENAI_API_KEY
```

Should show your actual key, not a placeholder.

### "Connection refused" to Neo4j

Wait 30 seconds for Neo4j to fully start, then try again.

Or check status:
```bash
docker compose logs neo4j | grep "Started"
```

Should see: `Started.`

---

## What's Next?

Your system is running successfully!

Next: [Architecture Overview →](./04-architecture-overview.md)

Learn how the system works, what each service does, and how data flows through the application.

