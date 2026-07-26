# M5.2 — Edit Observability (design)

**Status:** approved by owner 2026-07-26 (brainstorm dialogue)
**Milestone:** M5.2 — the last item of the M5.x UI arc, reframed by the owner
from "metrics visualized in the gallery" to a **standalone telemetry service**.

## Goal

Measure how much humans change what the system produced, to serve two decisions:

1. **Where are the gaps** — which extractors / lenses / dimensions get corrected
   most (and whether the model even knew it was weak there). Points model/prompt
   investment at real weaknesses.
2. **How to improve the correction workflow** — where humans spend correction
   effort, where corrections churn, and which corrections recur (rule/prompt
   candidates) — for the cases we can't yet automate away.

This is **telemetry for the builder, not a user feature.** It is deliberately
*not* mounted in the user-facing API or the gallery. It reads **correction
metadata** — the *type* of change, never the corrected content.

## What is already in place (why this is low-risk)

Every event carries an `Actor` with `actor_type ∈ {human, system, ai}`, so the
raw signal is in the event log today — no wire-format change, no backfill, no
aggregate changes. The correction events already record *what kind* of thing
changed:

- `AnalysisOverridden.fields_overridden` — the dict **keys** name the dimensions
  a human changed (`purpose`, `function_type`, `topics`, …).
- `LensExtractionOverridden` — `item_id` + `fields_overridden` keys + `note`.
- `SpeakerReattributed`, `SpeakerRenamed`, `SpeakerMerged`, `StitchRemoved`,
  `SegmentRemoved`, `EntityMergeConfirmed`, `EntitySplit`, `PersonLinkRemoved` —
  the **event type itself** is the correction taxonomy.

The machine-produced events also carry `confidence`, `dimension_confidences`,
and review `flags`, which is what makes the calibration analysis possible.

## Architecture

A standalone module `src/observability/` with an on-demand entry point:

```
python -m src.observability <scope>
```

- **Stateless full replay.** Each run replays the three category streams
  (`$ce-Interview`, `$ce-Sentence`, `$ce-Project`) from the start, computes
  metrics in memory, and emits an OKF-style telemetry domain. No checkpoint, no
  standing process (that is the v2 roadmap item, below).
- **Both sides of every rate come from the log.** Denominators (machine
  productions: `AnalysisGenerated`, `LensExtractionGenerated`,
  `SpeakerAttributed`, …) and numerators (human corrections) are both derived
  from replayed events. **Zero dependency on Neo4j or the user read model** —
  the telemetry service is fully decoupled from the user-facing app.
- **Not user-accessible.** No FastAPI router, no gallery view. It is a CLI/batch
  tool that produces files.

### The "no content" boundary (enforced structurally)

The reader projects each event to a meta-only record. It reads: event *type*,
`actor_type`, target ids, overridden-field *keys*, `confidence`/
`dimension_confidences`/`flags`, and *categorical* values (enum labels such as
`purpose: statement`). It **never** reads free-text field values (transcript
text, lens item prose, speaker display names as content). This boundary lives in
one function (`reader.py`), so "we don't look at content" is a property the code
can demonstrate, not merely a promise. A unit test asserts the projected records
contain no free-text values.

### Scope

Parameterized:

- **project** (default) — enough interviews for the rates to be meaningful.
- **corpus** — all projects.
- **interview** — allowed, but the domain flags it as small-sample (one
  interview's corrections don't generalize).

## What it measures

### Correction taxonomy

The **numerator** is human events that *change or undo* machine output. Pure
human *authorship* (e.g. a human identifying a brand-new person with no prior
machine link) is counted separately as "human authorship," not as a correction.

| Machine produces (denominator) | Human corrects (numerator) | Producer type |
|---|---|---|
| `SpeakerAttributed` | `SpeakerReattributed` | ai (inference) / system (parsed) |
| `SpeakerCreated` | `SpeakerRenamed`, `SpeakerMerged` | ai / system |
| `AnalysisGenerated` (per dimension) | `AnalysisOverridden` (per `fields_overridden` key) | ai |
| `LensExtractionGenerated` (per lens / node_type) | `LensExtractionOverridden` | ai |
| `UtteranceIdentified` / `InterruptionRecorded` | `StitchRemoved` | system |
| `SegmentIdentified` | `SegmentRemoved` | ai |
| `EntityCanonicalized` / `SpeakerLinkedToPerson` | `EntityMergeConfirmed`, `EntitySplit`, `PersonLinkRemoved` | ai |

**Producer type is reported separately.** The sharpest "AI gap" signal is the
rate against `ai` denominators; `system` (deterministic) corrections usually
mean bad input, not a weak model, and are not conflated.

### Goal (a) — where the gaps are

- **Correction rate** = corrections ÷ machine-produced, sliced by
  extractor/dimension, lens/node_type, speaker-method, and resolution.
- **Calibration** — for corrected items, the machine's `confidence` /
  `dimension_confidences` and whether a review `flag` fired. Report
  mean-confidence-of-corrected vs mean-confidence-of-uncorrected. Corrections
  clustered at **high confidence with no flag** are a blind spot the model
  cannot self-detect — surfaced prominently as the highest-value finding.

### Goal (b) — the correction workflow

- **Volume ranking** — the most-frequent correction types/paths (categorical
  X→Y where the field is enumerated; otherwise by field/type), so correction-UX
  investment goes where the human effort actually is.
- **Churn / rework** — targets corrected more than once, re-corrected, or
  reverted. Because lens/analysis re-runs respect `locked`, churn should not
  arise from re-runs; any observed churn is either the human going back and
  forth or a bug — flagged either way.
- **Recurring patterns** — categorical corrections that repeat (e.g.
  `purpose: statement→question`, N times). These are rule/prompt candidates and
  the first bridge toward the future automate-it loop. Limited to categorical
  fields by the content boundary; free-text overrides are counted but not
  content-mined.

## Output: the OKF telemetry domain

Written on demand to `data/observability/<scope>/` (git-versionable):

```
observability_<scope>/
├── index.md            # headline: totals, overall correction rate, top hotspots,
│                       #           top recurring patterns, worst calibration
├── extractors/
│   ├── purpose.md      # rate · calibration · volume · patterns · churn
│   ├── function-type.md
│   └── topics.md
├── lenses/
│   └── persona.md      # override rate per node_type
├── speakers/
│   └── attribution.md  # reattribution rate, by method
├── resolution/
│   └── entities.md     # merge / split / unlink rates
└── patterns.md         # recurring categorical X→Y across the corpus
```

Each file has YAML frontmatter (`scope`, `generated_at`, event-count and
commit-position range — provenance, **not** content) and the metrics as small
Markdown tables. "Grounded" here means grounded in **event provenance** (counts,
ids, positions), never transcript text.

The domain gets its **own lightweight renderer** rather than reusing the lens/OKF
renderer — the shape is telemetry, not lens items, and forcing reuse would
couple two things that should evolve independently.

## Module design

New module `src/observability/` (deliberately not named `metrics` —
`src/projections/metrics.py` and `src/utils/metrics.py` already exist and mean
other things). Mirrors the `src/export/` guard → read → compute → render → write
split:

- `reader.py` — replays the category streams; projects each event to a meta-only
  `ProductionRecord` / `CorrectionRecord`. **The content boundary lives here.**
- `metrics.py` — pure aggregation (records → rates, calibration, volume, churn,
  patterns). No I/O.
- `renderer.py` — metric structures → `(relative_path, markdown)` pairs. Pure.
- `domain.py` — orchestration: scope → read → compute → render → write.
- `__main__.py` — the `python -m src.observability <scope>` CLI.

The pure pieces (`reader`, `metrics`, `renderer`) are designed to be reused
wholesale by the v2 standing service — only the *driver* changes (replay-once →
subscribe-and-accumulate).

## Error handling & edge cases

- Legacy/malformed events are skipped and counted (mirroring the projection
  handlers' best-effort decode), never fatal.
- Empty scope → a valid "no data" domain rather than an error.
- A scope with zero corrections → rates of 0, domain still emitted (zero
  corrections is itself a signal — either nothing needs fixing or nobody is
  reviewing).

## Testing

- **Unit** — `reader` projects events to records and its records **contain no
  free-text values** (the content boundary as an executable guarantee);
  `metrics` aggregation over synthetic record sets (rates, calibration, churn,
  pattern detection); `renderer` output shape.
- **Integration smoke (env-gated)** — seed real events (ingest + a handful of
  corrections) into ESDB, run the generator, assert the OKF files materialize
  with the expected rates. Mirrors the layer smokes.

## Non-goals (v1)

- Any user-facing surface (no gallery view, no user API).
- A standing/live service, checkpoints, or incremental accumulation (v2 — see
  roadmap).
- Reading or reporting corrected **content** (only the type/shape of change).
- The automated-learning feedback loop (v2+); v1 only *shapes* its artifact to
  enable it later.
- Correction effort/latency metrics (weakly supported by current event data;
  deferred).

## v2 horizon (roadmapped, not built here)

A **standing telemetry service** (roadmap milestone M5.3): a checkpointed
consumer that subscribes to the category streams, accumulates metrics into a
store, and exposes an **internal (non-user) query API / dashboard**, live and
regression-alertable. Triggered when on-demand replay gets too slow at corpus
scale or continuous monitoring is wanted. This is also where the **feedback
loop** lands — the telemetry domain feeding prompt/model improvement and
eventually automated learning. It reuses v1's `reader` / `metrics` / `renderer`
unchanged; only the driver differs.
