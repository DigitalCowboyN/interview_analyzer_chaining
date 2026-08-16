# Graph

> Counts are live — regenerate with `make graph-index` after adding nodes or edges.

| edge | inverse | from → to | source | properties | count |
| --- | --- | --- | --- | --- | --- |
| implements | implemented_by | Capability → CodeUnit | authored | — | 104 |
| child_of | parent_of | Capability → Capability | authored | — | 40 |
| depends_on | depended_on_by | CodeUnit → CodeUnit | derived | — | 60 |
| governs | governed_by | ADR → CodeUnit | authored | — | 24 |
| supersedes | superseded_by | ADR → ADR | authored | — | 1 |
| fulfilled_by | fulfills | UseCase → Capability | authored | — | 52 |
| verifies | verified_by | Test → CodeUnit\|UseCase\|Capability | derived | test_type | 205 |
| defined_in | defines | GlossaryTerm → CodeUnit | authored | — | 110 |
| consumed_by | consumes | GraphQuery → CodeUnit | derived | — | 61 |
| consumed_by | consumes | Prompt → CodeUnit | derived | — | 61 |

## Nodes

- ADR: 25
- Capability: 55
- CodeUnit: 48
- GlossaryTerm: 111
- GraphQuery: 33
- Prompt: 28
- Test: 205
- UseCase: 20

## Meta-schema

```mermaid
graph LR
    Capability -->|implements| CodeUnit
    Capability -->|child_of| Capability
    CodeUnit -->|depends_on| CodeUnit
    ADR -->|governs| CodeUnit
    ADR -->|supersedes| ADR
    UseCase -->|fulfilled_by| Capability
    Test -->|verifies| CodeUnit
    Test -->|verifies| UseCase
    Test -->|verifies| Capability
    GlossaryTerm -->|defined_in| CodeUnit
    GraphQuery -->|consumed_by| CodeUnit
    Prompt -->|consumed_by| CodeUnit
```
