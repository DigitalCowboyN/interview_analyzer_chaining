# Event Sourcing Architecture

> **Last Updated:** 2026-01-18

## Overview

The Interview Analyzer uses an event-sourced architecture with CQRS (Command Query Responsibility Segregation). EventStoreDB is the single source of truth, with Neo4j serving as a projected read model.

## CQRS Pattern

```mermaid
flowchart LR
    subgraph Commands["Write Side (Commands)"]
        cmd[User Command]
        handler[Command Handler]
        agg[Aggregate]
        event[Domain Event]
        esdb[(EventStoreDB)]

        cmd --> handler --> agg --> event --> esdb
    end

    subgraph Queries["Read Side (Queries)"]
        query[User Query]
        api[API Endpoint]
        neo4j[(Neo4j)]

        query --> api --> neo4j
    end

    subgraph Projection["Projection Layer"]
        sub[Subscription]
        proj[Projection Handler]

        esdb --> sub --> proj --> neo4j
    end
```

## Aggregate Model

### Interview Aggregate

```mermaid
classDiagram
    class Interview {
        +UUID id
        +str filename
        +str status
        +datetime created_at
        +int version
        +create(filename)
        +update_metadata(metadata)
        +change_status(status)
    }

    class InterviewCreated {
        +UUID interview_id
        +str filename
        +datetime created_at
        +Actor actor
    }

    class InterviewMetadataUpdated {
        +UUID interview_id
        +dict metadata
    }

    class InterviewStatusChanged {
        +UUID interview_id
        +str old_status
        +str new_status
    }

    Interview ..> InterviewCreated : emits
    Interview ..> InterviewMetadataUpdated : emits
    Interview ..> InterviewStatusChanged : emits
```

### Sentence Aggregate

```mermaid
classDiagram
    class Sentence {
        +UUID id
        +UUID interview_id
        +int sentence_index
        +str text
        +dict analysis
        +bool is_edited
        +int version
        +create(interview_id, index, text)
        +edit(new_text, actor)
        +generate_analysis(analysis)
        +override_analysis(field, value, actor)
    }

    class SentenceCreated {
        +UUID sentence_id
        +UUID interview_id
        +int sentence_index
        +str text
        +int sequence_order
    }

    class SentenceEdited {
        +UUID sentence_id
        +str old_text
        +str new_text
        +Actor actor
    }

    class AnalysisGenerated {
        +UUID sentence_id
        +dict analysis
        +str model_used
    }

    class AnalysisOverridden {
        +UUID sentence_id
        +str field
        +any old_value
        +any new_value
        +Actor actor
    }

    Sentence ..> SentenceCreated : emits
    Sentence ..> SentenceEdited : emits
    Sentence ..> AnalysisGenerated : emits
    Sentence ..> AnalysisOverridden : emits
```

## Event Envelope

All events are wrapped in an `EventEnvelope` for metadata:

```mermaid
classDiagram
    class EventEnvelope {
        +UUID event_id
        +UUID aggregate_id
        +str aggregate_type
        +str event_type
        +int event_version
        +datetime timestamp
        +UUID correlation_id
        +UUID causation_id
        +Actor actor
        +dict payload
    }

    class Actor {
        +str actor_type
        +str user_id
    }

    EventEnvelope *-- Actor
```

### Event Envelope JSON Example

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "aggregate_id": "660e8400-e29b-41d4-a716-446655440001",
  "aggregate_type": "Sentence",
  "event_type": "SentenceCreated",
  "event_version": 1,
  "timestamp": "2026-01-18T12:00:00Z",
  "correlation_id": "770e8400-e29b-41d4-a716-446655440002",
  "causation_id": "880e8400-e29b-41d4-a716-446655440003",
  "actor": {
    "actor_type": "system",
    "user_id": "pipeline"
  },
  "payload": {
    "sentence_id": "660e8400-e29b-41d4-a716-446655440001",
    "interview_id": "990e8400-e29b-41d4-a716-446655440004",
    "sentence_index": 1,
    "text": "The product launch was successful.",
    "sequence_order": 1
  }
}
```

## Event Streams

```mermaid
flowchart TD
    subgraph Streams["EventStoreDB Streams"]
        all["$all<br/>(system stream)"]

        interview1["Interview-{uuid1}"]
        interview2["Interview-{uuid2}"]

        sentence1["Sentence-{uuid1}"]
        sentence2["Sentence-{uuid2}"]
        sentence3["Sentence-{uuid3}"]
    end

    subgraph Events1["Interview 1 Events"]
        e1[InterviewCreated]
    end

    subgraph Events2["Sentence 1 Events"]
        e2[SentenceCreated]
        e3[AnalysisGenerated]
        e4[SentenceEdited]
    end

    e1 --> interview1
    e2 --> sentence1
    e3 --> sentence1
    e4 --> sentence1

    interview1 --> all
    interview2 --> all
    sentence1 --> all
    sentence2 --> all
    sentence3 --> all
```

## Event Flow: File Processing

```mermaid
sequenceDiagram
    participant U as User
    participant P as Pipeline
    participant E as EventEmitter
    participant ES as EventStoreDB
    participant N4 as Neo4j (Direct)
    participant PS as Projection Service
    participant N4P as Neo4j (Projected)

    U->>P: Upload file

    Note over P: Create Interview
    P->>E: emit InterviewCreated
    E->>ES: append to Interview-{id}
    P->>N4: create Interview node (temp)

    loop For each sentence
        Note over P: Create Sentence
        P->>E: emit SentenceCreated
        E->>ES: append to Sentence-{id}
        P->>N4: create Sentence node (temp)

        Note over P: Analyze Sentence
        P->>E: emit AnalysisGenerated
        E->>ES: append to Sentence-{id}
        P->>N4: update with analysis (temp)
    end

    Note over PS: Projection Service subscribes
    ES-->>PS: InterviewCreated
    PS->>N4P: MERGE Interview node

    ES-->>PS: SentenceCreated
    PS->>N4P: MERGE Sentence node

    ES-->>PS: AnalysisGenerated
    PS->>N4P: MERGE analysis relationships
```

## Event Flow: User Edit

```mermaid
sequenceDiagram
    participant U as User
    participant API as Edit API
    participant CH as Command Handler
    participant R as Repository
    participant ES as EventStoreDB
    participant PS as Projection Service
    participant N4 as Neo4j

    U->>API: PUT /sentences/{id}/edit
    API->>CH: EditSentenceCommand

    CH->>R: load Sentence aggregate
    R->>ES: read Sentence-{id} stream
    ES-->>R: event history
    R-->>CH: Sentence (reconstructed)

    CH->>CH: sentence.edit(new_text)
    Note over CH: Generates SentenceEdited event

    CH->>R: save(sentence)
    R->>ES: append SentenceEdited

    API-->>U: 202 Accepted

    Note over PS: Async projection
    ES-->>PS: SentenceEdited
    PS->>N4: UPDATE sentence text
```

## Projection Service Architecture

```mermaid
flowchart TD
    subgraph EventStore["EventStoreDB"]
        all["$all stream"]
    end

    subgraph SubscriptionMgr["Subscription Manager"]
        sub[Persistent Subscription<br/>projection-service-group]
        checkpoint[(Checkpoint Store)]
    end

    subgraph LaneMgr["Lane Manager"]
        hash[Consistent Hash<br/>by aggregate_id]

        lane1[Lane 1]
        lane2[Lane 2]
        lane3[Lane 3]
        lanen[Lane 12]
    end

    subgraph Handlers["Projection Handlers"]
        h1[InterviewCreatedHandler]
        h2[SentenceCreatedHandler]
        h3[SentenceEditedHandler]
        h4[AnalysisGeneratedHandler]
        h5[AnalysisOverriddenHandler]
    end

    subgraph Neo4j["Neo4j"]
        nodes[(Graph Nodes)]
    end

    all --> sub
    sub --> checkpoint
    sub --> hash

    hash --> lane1 & lane2 & lane3 & lanen

    lane1 & lane2 & lane3 & lanen --> h1 & h2 & h3 & h4 & h5

    h1 & h2 & h3 & h4 & h5 --> nodes
```

### Lane Processing

- **12 parallel lanes** for horizontal scaling
- **Consistent hashing** ensures same aggregate always goes to same lane
- **In-order processing** within each lane
- **Checkpoint management** for resume capability

### Idempotency

Projection handlers use `event_version` to prevent duplicate processing:

```cypher
MERGE (s:Sentence {sentence_id: $sentence_id})
ON CREATE SET s.event_version = $event_version, s.text = $text
ON MATCH SET s.text = CASE
    WHEN s.event_version IS NULL OR s.event_version < $event_version
    THEN $text ELSE s.text END,
    s.event_version = CASE
    WHEN s.event_version IS NULL OR s.event_version < $event_version
    THEN $event_version ELSE s.event_version END
```

## Command Handlers

```mermaid
flowchart TD
    subgraph Commands["Available Commands"]
        c1[CreateInterviewCommand]
        c2[CreateSentenceCommand]
        c3[EditSentenceCommand]
        c4[GenerateAnalysisCommand]
        c5[OverrideAnalysisCommand]
    end

    subgraph Handlers["Command Handlers"]
        h[CommandHandler]
    end

    subgraph Repository["Repository Layer"]
        r[AggregateRepository]
    end

    subgraph Aggregates["Aggregates"]
        i[Interview]
        s[Sentence]
    end

    c1 & c2 & c3 & c4 & c5 --> h
    h --> r
    r --> i & s
```

## Deterministic UUIDs

Sentence IDs are generated deterministically using `uuid5` for idempotency:

```python
sentence_uuid = uuid5(
    NAMESPACE_DNS,
    f"{interview_id}:{sentence_index}"
)
```

This ensures:
- Same sentence always gets same UUID
- Replay safety (events can be reprocessed)
- Idempotent operations across retries

## Optimistic Concurrency

EventStoreDB enforces event ordering via expected version:

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2
    participant ES as EventStoreDB

    C1->>ES: Read stream (version 5)
    C2->>ES: Read stream (version 5)

    C1->>ES: Append event (expected: 5)
    ES-->>C1: Success (now version 6)

    C2->>ES: Append event (expected: 5)
    ES-->>C2: CONFLICT (version is 6)

    Note over C2: Must reload and retry
```
