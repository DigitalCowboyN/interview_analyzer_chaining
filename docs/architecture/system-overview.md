# System Overview

> **Last Updated:** 2026-01-18

## System Context

The Interview Analyzer is an event-sourced application that processes interview transcripts using AI and stores results in a graph database.

```mermaid
C4Context
    title System Context Diagram - Interview Analyzer

    Person(user, "User", "Uploads transcripts, queries analysis results, makes corrections")

    System(analyzer, "Interview Analyzer", "Processes interview transcripts with AI, stores structured analysis in graph database")

    System_Ext(openai, "OpenAI API", "GPT models for sentence classification")
    System_Ext(anthropic, "Anthropic API", "Claude models (optional)")
    System_Ext(gemini, "Google Gemini API", "Gemini models (optional)")

    Rel(user, analyzer, "Uploads files, queries results", "HTTP/REST")
    Rel(analyzer, openai, "Classifies sentences", "HTTPS")
    Rel(analyzer, anthropic, "Classifies sentences", "HTTPS")
    Rel(analyzer, gemini, "Classifies sentences", "HTTPS")
```

## Container Diagram

```mermaid
C4Container
    title Container Diagram - Interview Analyzer Services

    Person(user, "User", "Analyst or Developer")

    Container_Boundary(app_boundary, "Interview Analyzer System") {
        Container(api, "FastAPI Application", "Python, FastAPI", "REST API, pipeline orchestration, command handling")
        Container(worker, "Celery Worker", "Python, Celery", "Background task processing")
        Container(projection, "Projection Service", "Python", "Event consumer, Neo4j writer")

        ContainerDb(eventstore, "EventStoreDB", "EventStoreDB 23.10", "Event streams, source of truth")
        ContainerDb(neo4j, "Neo4j", "Neo4j 5.22", "Graph database, read model")
        ContainerDb(redis, "Redis", "Redis 7", "Message broker, task queue")
    }

    System_Ext(llm, "LLM APIs", "OpenAI, Anthropic, Gemini")

    Rel(user, api, "HTTP requests", "REST")
    Rel(api, worker, "Queue tasks", "Redis")
    Rel(api, eventstore, "Write events", "gRPC")
    Rel(api, neo4j, "Direct write (temp)", "Bolt")
    Rel(api, llm, "Classify sentences", "HTTPS")

    Rel(eventstore, projection, "Subscribe", "gRPC")
    Rel(projection, neo4j, "Project events", "Bolt")

    Rel(worker, api, "Execute pipeline", "Internal")
```

## Service Descriptions

### FastAPI Application (Port 8000)

**Responsibilities:**
- REST API endpoints for file operations and analysis
- Pipeline orchestration for transcript processing
- Command handling for user edits
- Event emission to EventStoreDB

**Key Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/files/` | GET | List analysis files |
| `/files/{filename}` | GET | Get analysis content |
| `/analysis/` | POST | Trigger background analysis |
| `/edits/sentences/{id}/{index}/edit` | POST | Edit sentence text |
| `/edits/sentences/{id}/{index}/analysis/override` | POST | Override AI analysis |

### Celery Worker

**Responsibilities:**
- Execute long-running pipeline tasks
- Process files asynchronously
- Prevent API timeouts

**Configuration:**
- Broker: Redis at `redis:6379`
- Entry point: `celery -A src.celery_app worker`

### Projection Service

**Responsibilities:**
- Subscribe to EventStoreDB `$all` stream
- Process events in order
- Update Neo4j graph (create nodes, relationships)
- Handle idempotency via event versioning

**Configuration:**
- 12 parallel processing lanes
- Checkpoint management for resume
- Entry point: `python -m src.run_projection_service`

### EventStoreDB (Ports 2113, 1113)

**Responsibilities:**
- Store all domain events (source of truth)
- Provide event streams for aggregates
- Enable event replay and audit trail
- Support persistent subscriptions

**Event Streams:**
- `Interview-{uuid}` - Interview aggregate events
- `Sentence-{uuid}` - Sentence aggregate events

### Neo4j (Ports 7474, 7687)

**Responsibilities:**
- Store sentence analysis as graph nodes
- Maintain relationships (topics, keywords, etc.)
- Enable complex graph queries
- Serve as CQRS read model

**Authentication:**
- Username: `neo4j`
- Password: From `.env` file

### Redis (Port 6379)

**Responsibilities:**
- Message broker for Celery
- Task result backend
- Pub/sub for worker communication

## Deployment View

```mermaid
flowchart TB
    subgraph Docker["Docker Compose Environment"]
        subgraph AppServices["Application Services"]
            api[fa:fa-server FastAPI<br/>Port 8000]
            worker[fa:fa-cogs Celery Worker]
            projection[fa:fa-stream Projection Service]
        end

        subgraph DataStores["Data Stores"]
            eventstore[fa:fa-database EventStoreDB<br/>Port 2113]
            neo4j[fa:fa-project-diagram Neo4j<br/>Ports 7474, 7687]
            neo4j_test[fa:fa-vial Neo4j Test<br/>Ports 7475, 7688]
            redis[fa:fa-memory Redis<br/>Port 6379]
        end

        subgraph Volumes["Persistent Volumes"]
            vol_neo4j[(neo4j_data)]
            vol_redis[(redis_data)]
            vol_es[(eventstore_data)]
        end
    end

    subgraph External["External Services"]
        openai[fa:fa-brain OpenAI API]
        anthropic[fa:fa-robot Anthropic API]
        gemini[fa:fa-google Google Gemini]
    end

    api --> redis
    api --> eventstore
    api --> neo4j
    api --> openai
    api --> anthropic
    api --> gemini

    worker --> redis
    worker --> api

    projection --> eventstore
    projection --> neo4j

    neo4j --> vol_neo4j
    redis --> vol_redis
    eventstore --> vol_es
```

## Environment Detection

The system auto-detects its runtime environment:

```mermaid
flowchart TD
    Start[Start] --> CheckDocker{/.dockerenv exists?}
    CheckDocker -->|Yes| Docker[Docker Environment]
    CheckDocker -->|No| CheckCI{CI env var set?}
    CheckCI -->|Yes| CI[CI Environment]
    CheckCI -->|No| Host[Host Environment]

    Docker --> DockerConfig["Use service names<br/>neo4j:7687<br/>eventstore:2113"]
    CI --> CIConfig["Use localhost<br/>with CI ports"]
    Host --> HostConfig["Use localhost<br/>localhost:7687<br/>localhost:2113"]
```
