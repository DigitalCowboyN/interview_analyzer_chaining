# Graph

> Counts are live — regenerate with `make graph-index` after adding nodes or edges.

| edge | inverse | from → to | source | properties | count |
| --- | --- | --- | --- | --- | --- |
| implements | implemented_by | Capability → CodeUnit | authored | — | 106 |
| child_of | parent_of | Capability → Capability | authored | — | 41 |
| depends_on | depended_on_by | CodeUnit → CodeUnit | derived | — | 375 |
| contains | contained_by | CodeUnit → CodeUnit | derived | — | 166 |
| governs | governed_by | ADR → CodeUnit | authored | — | 80 |
| supersedes | superseded_by | ADR → ADR | authored | — | 1 |
| fulfilled_by | fulfills | UseCase → Capability | authored | — | 55 |
| verifies | verified_by | Test → CodeUnit\|UseCase\|Capability | derived | test_type | 211 |
| defined_in | defines | GlossaryTerm → CodeUnit | authored | — | 110 |
| consumed_by | consumes | GraphQuery → CodeUnit | derived | — | 61 |
| consumed_by | consumes | Prompt → CodeUnit | derived | — | 61 |

## Nodes

- ADR: 27
- Capability: 56
- CodeUnit: 199
- GlossaryTerm: 111
- GraphQuery: 33
- Prompt: 28
- Test: 212
- UseCase: 21

## Meta-schema

```mermaid
graph LR
    Capability -->|implements| CodeUnit
    Capability -->|child_of| Capability
    CodeUnit -->|depends_on| CodeUnit
    CodeUnit -->|contains| CodeUnit
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
