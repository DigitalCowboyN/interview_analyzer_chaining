# Data Flow

> **Last Updated:** 2026-01-18

## Pipeline Processing Flow

The core pipeline transforms raw text files into structured, multi-dimensional sentence analysis.

```mermaid
flowchart TD
    subgraph Input["1. Input"]
        file[fa:fa-file-text Text File<br/>data/input/*.txt]
    end

    subgraph Segmentation["2. Text Segmentation"]
        read[Read File<br/>TextDataSource]
        spacy[spaCy NLP<br/>en_core_web_sm]
        sentences[Sentence List]

        read --> spacy --> sentences
    end

    subgraph Mapping["3. Sentence Mapping"]
        map_create[Create Sentence Map]
        map_file[fa:fa-file-code Map File<br/>data/maps/*_map.jsonl]
        event_interview[fa:fa-bolt InterviewCreated<br/>Event]
        event_sentence[fa:fa-bolt SentenceCreated<br/>Events]

        sentences --> map_create
        map_create --> map_file
        map_create --> event_interview
        map_create --> event_sentence
    end

    subgraph Context["4. Context Building"]
        context[ContextBuilder]
        windows[Context Windows<br/>immediate, observer,<br/>broader, overall]

        sentences --> context --> windows
    end

    subgraph Analysis["5. LLM Analysis"]
        queue[Async Queue<br/>10 workers]
        analyzer[SentenceAnalyzer]
        llm[fa:fa-brain LLM API<br/>7 parallel calls]

        windows --> queue --> analyzer --> llm
    end

    subgraph Classification["Analysis Dimensions"]
        func[Function Type]
        struct[Structure Type]
        purpose[Purpose]
        topic1[Topic Level 1]
        topic3[Topic Level 3]
        kw_overall[Overall Keywords]
        kw_domain[Domain Keywords]

        llm --> func & struct & purpose & topic1 & topic3 & kw_overall & kw_domain
    end

    subgraph Persistence["6. Event-First Persistence"]
        consolidate[Consolidate Results]
        event_analysis[fa:fa-bolt AnalysisGenerated<br/>Event]
        jsonl[fa:fa-file-code Analysis File<br/>data/output/*_analysis.jsonl]

        func & struct & purpose & topic1 & topic3 & kw_overall & kw_domain --> consolidate
        consolidate --> event_analysis
        consolidate --> jsonl
    end

    subgraph Projection["7. Event Projection"]
        eventstore[(EventStoreDB)]
        proj_service[Projection Service]
        neo4j_proj[fa:fa-project-diagram Neo4j<br/>Projected View]

        event_interview & event_sentence & event_analysis --> eventstore
        eventstore --> proj_service --> neo4j_proj
    end

    file --> read
```

## Analysis Dimensions

Each sentence is classified across 7 dimensions via parallel LLM calls:

```mermaid
flowchart LR
    subgraph Input
        sentence[Sentence + Context]
    end

    subgraph LLM["Parallel LLM Calls"]
        call1[Function Type<br/>declarative, interrogative, etc.]
        call2[Structure Type<br/>simple, compound, complex]
        call3[Purpose<br/>statement, query, explanation]
        call4[Topic Level 1<br/>high-level category]
        call5[Topic Level 3<br/>specific subcategory]
        call6[Overall Keywords<br/>general terms]
        call7[Domain Keywords<br/>domain-specific terms]
    end

    subgraph Output
        result[Consolidated<br/>Analysis Result]
    end

    sentence --> call1 & call2 & call3 & call4 & call5 & call6 & call7
    call1 & call2 & call3 & call4 & call5 & call6 & call7 --> result
```

## Context Window Building

The `ContextBuilder` creates different context windows around each sentence:

```mermaid
flowchart TD
    subgraph Document["Full Document"]
        s1[Sentence 1]
        s2[Sentence 2]
        s3[Sentence 3]
        target[**TARGET SENTENCE**]
        s5[Sentence 5]
        s6[Sentence 6]
        s7[Sentence 7]
    end

    subgraph Windows["Context Windows"]
        immediate["Immediate Context<br/>±1 sentence"]
        observer["Observer Context<br/>±3 sentences"]
        broader["Broader Context<br/>±5 sentences"]
        overall["Overall Context<br/>Full document summary"]
    end

    s3 & target & s5 --> immediate
    s2 & s3 & target & s5 & s6 --> observer
    s1 & s2 & s3 & target & s5 & s6 & s7 --> broader
    Document --> overall
```

## Async Worker Architecture

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant Q as Async Queue
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant Wn as Worker N
    participant LLM as LLM API
    participant R as Result Queue

    P->>Q: Enqueue sentences (batch)

    par Concurrent Processing
        Q->>W1: Sentence 1
        W1->>LLM: Analyze (7 calls)
        LLM-->>W1: Results
        W1->>R: Analysis 1

        Q->>W2: Sentence 2
        W2->>LLM: Analyze (7 calls)
        LLM-->>W2: Results
        W2->>R: Analysis 2

        Q->>Wn: Sentence N
        Wn->>LLM: Analyze (7 calls)
        LLM-->>Wn: Results
        Wn->>R: Analysis N
    end

    R->>P: Consolidated results
```

## File Input/Output

### Input Files
- **Location:** `data/input/`
- **Format:** Plain text (`.txt`)
- **Content:** Interview transcripts, documents

### Output Files

| File Type | Location | Format | Content |
|-----------|----------|--------|---------|
| Map Files | `data/maps/*_map.jsonl` | JSON Lines | `{sentence_id, sequence_order, sentence}` |
| Analysis Files | `data/output/*_analysis.jsonl` | JSON Lines | Full analysis per sentence |
| Log Files | `logs/pipeline.log` | Text | Processing logs |

### JSONL Analysis Format

```json
{
  "sentence_id": 1,
  "sequence_order": 1,
  "sentence": "The product launch was successful.",
  "filename": "interview_001.txt",
  "function_type": "declarative",
  "structure_type": "simple",
  "purpose": "statement",
  "topic_level_1": "Product",
  "topic_level_3": "Product Launch",
  "overall_keywords": ["product", "launch", "successful"],
  "domain_keywords": ["product launch", "release"]
}
```

## Error Handling Flow

```mermaid
flowchart TD
    subgraph Pipeline["Pipeline Operation"]
        op[Operation Start]
        emit[Emit Event to ESDB]
        complete[Operation Complete]
    end

    subgraph Errors["Error Handling"]
        event_fail{Event Failed?}
        abort[fa:fa-times-circle ABORT<br/>Raise Exception]
    end

    subgraph Projection["Projection Service"]
        proj[Process Event]
        neo4j_write[Write to Neo4j]
        proj_fail{Write Failed?}
        park[Park Event<br/>for Retry]
    end

    op --> emit --> event_fail
    event_fail -->|Yes| abort
    event_fail -->|No| complete

    emit -.->|async| proj --> neo4j_write --> proj_fail
    proj_fail -->|Yes| park
    proj_fail -->|No| complete

    style abort fill:#f66,stroke:#333
    style park fill:#ff9,stroke:#333
    style complete fill:#6f6,stroke:#333
```

**Key Principle:** Events are the source of truth. Event failures abort the operation. Projection service is the sole writer to Neo4j and handles retries independently.
