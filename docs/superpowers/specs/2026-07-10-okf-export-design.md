# M4.4 — Layer 5: OKF Export + Richer Queries (design)

**Status:** approved 2026-07-10
**Parent spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md` (Layer 5 section)
**OKF reference:** Google Cloud OKF v0.1 — `github.com/GoogleCloudPlatform/knowledge-catalog/okf/SPEC.md`

## Goal

Make the mine consumable. Three deliverables:

1. **OKF exporter** — render one interview × one lens into an Open Knowledge Format
   bundle (directory of markdown files with YAML frontmatter), git-versionable and
   readable by humans and agents equally.
2. **Front-matter capture (ingest)** — transcripts may open with an OKF-compatible
   YAML block (title, project, date, participants); parse it, store it on the
   Interview, and seed speaker genesis with the participant names.
3. **Richer REST queries** — lens items by interview, low-confidence review
   worklist, by-speaker rollups.

Decisions locked during brainstorming: consumer is humans and agents **equally**;
surface is **CLI + API**; front-matter capture is **in scope** (full round-trip);
**all three** query families ship; bundles include **everything enriched** (lens
items + claims + entities + analysis dimensions).

## Architectural decision

**Read-side exporter over Neo4j** (chosen over aggregate/ESDB rendering and over a
hybrid). The exporter is a projection renderer: parameterized Cypher pulls the
interview's subgraph and pure functions write markdown. The REST queries read the
same Neo4j model through the same reader module — one data-access layer for both.
Rejected alternatives:

- *Aggregate-side rendering* — Interview state deliberately keeps only skeletal
  lens-item state (id/type/locked, no fields), so rendering from aggregates would
  force raw event re-reads or fattened state; by-speaker rollups are cross-interview
  and need Neo4j regardless.
- *Hybrid (bundles from events, queries from Neo4j)* — two data models for the same
  content, no consumer-visible gain.

Projection lag is handled by an explicit **consistency guard** (below), not ignored.

## OKF v0.1 rules that bind us (verified against the spec)

- Every non-reserved `.md` file: YAML frontmatter with a non-empty **`type`** —
  the only required field. Recommended: `title`, `description`, `resource`, `tags`,
  `timestamp`. Producers may add arbitrary keys; consumers preserve unknown keys.
- **`index.md` is reserved and contains NO frontmatter** — it is a plain listing
  with relative links and descriptions. (This corrects the parent spec's sketch,
  which put interview front matter in `index.md`; that content lives in
  `interview.md` instead.)
- **`log.md` is reserved** — change history, flat list grouped by ISO date,
  newest first.
- Links: bundle-absolute (`/`-prefixed, recommended) or relative; a link asserts a
  relationship (consumers treat links as edges). Broken links must be tolerated.
- A bundle may ship as a git repo, a subdirectory, or a zip/tarball.

## Bundle layout

One export = one interview × one lens → one directory:

```
<out>/<interview_id>-<lens>/
  index.md                          # reserved: no frontmatter; listing w/ links + descriptions
  log.md                            # reserved: export history, newest-first, prepended on re-export
  interview.md                      # type: Interview — title, source, project, date,
                                    #   participants, fragment/utterance counts, lens+version,
                                    #   raw ingested front matter (round-trip fidelity)
  transcript.md                     # type: Transcript — as-spoken utterances in order,
                                    #   speaker-labeled, one stable anchor per utterance
  analysis.md                       # type: AnalysisSummary — compact per-fragment table
                                    #   (function/structure/purpose/topics/keywords/confidence)
  speakers/<handle-slug>.md         # type: Speaker — display name, provisional flag,
                                    #   links to items they own/made
  <kebab(node_type)>s/<kebab(node_type)>-<id8>.md
                                    # e.g. decisions/decision-8435750e.md — type: <node_type>
  claims/claim-<id8>.md             # type: Claim — kind, confidence, speaker link, grounding
  entities/<entity-slug>.md         # type: Entity — ONE file per entity; mentions aggregated
                                    #   across fragments (not one file per mention)
```

Naming rules (all mechanical — zero per-lens code):

- Item directory = `kebab-case(node_type) + "s"` (naive plural; correct for all
  current node types and documented as the convention lens authors accept).
- Item filename stem = `kebab-case(node_type)-<first 8 hex of item_id>` —
  deterministic and stable across re-exports so git diffs stay clean.
- Speaker/entity slugs: lowercase, non-alphanumeric → `-`, collision-suffixed.

### Lens item file anatomy

```markdown
---
type: Decision                    # node_type verbatim (required OKF field)
title: Go with DataFlux           # item text, truncated ~80 chars
description: <one-sentence: the item text>
id: <full item_id>
lens: meeting_minutes
lens_version: 1
confidence: 0.9
model: haiku
provider: anthropic
locked: false                     # true when human-overridden
tags: [lens:meeting_minutes]
timestamp: <export time, ISO 8601>
due: Friday                       # extracted fields pass through as-is,
owner_unresolved: true            #   including unresolved markers
---

Go with DataFlux

## Relationships

- DECIDED_BY: [Alice](/speakers/alice.md)

## Grounding

> Let's go with DataFlux then.
> — [Alice](/speakers/alice.md), [utterance 3](/transcript.md#u-3)
```

Relationship links carry the graph edge name in the link text (OKF links-as-edges);
grounding quotes are verbatim fragment text with bundle-absolute links to the
transcript anchor. Claims files follow the same anatomy with `type: Claim` and
`kind` in frontmatter. Entity files list mentions as a table (fragment link, span,
surface form, confidence).

`analysis.md` renders one row per fragment: index, speaker, truncated text,
function/structure/purpose, topics, keywords, mean confidence, flags. Aggregated
topic/keyword tallies at the top.

## Exporter architecture

Package `src/export/`:

| Unit | Responsibility | Depends on |
|------|----------------|-----------|
| `reader.py` | ALL Cypher for Layer 5: `interview_header`, `transcript_utterances`, `speakers`, `lens_items`, `claims`, `entities`, `analysis_rows`, plus worklist/rollup queries. Returns plain dicts. | Neo4j driver |
| `renderer.py` | Pure functions: dicts in → `(relative_path, content)` tuples out. No I/O, no Neo4j. Lens-item rendering driven by `node_type` + node properties + the lens YAML's `projects_to` (relationship names). | `src/lens/models.py` |
| `bundler.py` | `OkfExporter.export(interview_id, lens_name, out_dir, zip=False) -> ExportResult` — guard, read, render fully in memory, then write (or zip); prepend `log.md` entry. | reader, renderer, repositories |

`ExportResult(BaseModel)`: `interview_id, lens, lens_version, bundle_path,
files_written, items, claims, entities`.

**Consistency guard:** before rendering, load the Interview aggregate (existing
repository) and compute the ids that SHOULD be projected for this lens: items whose
`lens_version == lens_runs[lens]` (the current run) plus locked items of any version
(supersession never deletes them). Compare against Neo4j's projected item ids.
Mismatch = projection lag → retryable error (CLI exit 1 with a clear message;
API 409). Note the aggregate's `lens_items` alone is NOT the expected set — it
retains superseded ids forever while `LensApplied` correctly deletes their graph
nodes. Zero items for an applied lens is NOT an error (an honest empty report:
bundle ships with interview/transcript/claims).

**Surfaces:**

- CLI: `python -m src.export <interview_id> <lens_name> [--out exports/] [--zip]`,
  prints `ExportResult` JSON. Mirrors `src.ingestion` / `src.lens`.
- API: `GET /exports/{interview_id}/{lens_name}` → synchronous zip download
  (export is LLM-free and fast; no background job). 404 unknown interview,
  409 projection lag, 422 unknown lens. Router `src/api/routers/exports.py`.

Re-export overwrites the bundle directory deterministically; `log.md` accumulates
one entry per run ("exported N items from <lens> v<version>").

## Front-matter capture (ingest)

Transcripts may open with a `---` YAML block:

```markdown
---
title: Q3 Vendor Selection
project: telemetry
date: 2026-07-01
participants: [Alice Johnson, Bob Reyes]
tags: [vendor, infra]
---
Alice: Thanks for joining...
```

- **Detection/parsing:** the format detector recognizes a leading front-matter
  block. Parsing is tolerant: known keys map to typed fields; unknown keys are
  preserved verbatim; malformed YAML degrades to "no front matter" with a warning —
  **ingest never fails on front matter**.
- **Storage:** through the existing `InterviewCreated` payload — `title`,
  `started_at` (from `date`), and everything (including the raw block under
  `metadata["front_matter"]`) into `metadata`. **No new event types.**
- **Offsets invariant:** segmentation runs over the original file with the
  front-matter region excluded; fragment `start_char`/`end_char` remain offsets
  into the UNMODIFIED source file, so `source[start:end] == fragment` still holds.
- **Speaker seeding:** participants are hints, never constraints.
  - Labeled transcripts: a label matching a participant name (case-insensitive
    full-name match, or first-name match when exactly ONE participant has that
    first name — ambiguous first names don't seed) creates that speaker confirmed
    (`provisional=False`, method `front_matter`) with the participant's full name
    as display name.
  - Unlabeled transcripts: participant names ride into the speaker-inference
    prompt as candidate speakers.
  - Unlisted speakers still emerge as provisional `S<n>`; listed-but-silent
    participants create no speaker.

## Richer REST queries

All read Neo4j through `reader.py`; all scoped; list endpoints paginated
(`limit`/`offset`, default 50). New router `src/api/routers/queries.py`.

1. **`GET /interviews/{interview_id}/lenses/{lens}/items`**
   (`?node_type=`, `?min_confidence=`) — the bundle's programmatic mirror: items
   with fields, speaker links (relationship + speaker id + display name), and
   grounding fragment ids.
2. **`GET /review/worklist`** (`?project_id=`, `?threshold=0.7`) — the human
   review queue: lens items with `confidence < threshold` OR any `*_unresolved`
   field marker, plus claims below threshold. Each row carries what the override
   endpoint needs (`interview_id`, `item_id`, reason it surfaced).
3. **`GET /speakers/rollup`** (`?project_id=`, `?name=`) — by-speaker across
   interviews: action items owned, decisions made, claims asserted, with per-item
   interview references. Cross-interview speaker identity is **display-name string
   match** in v1 — documented limitation; real identity resolution is Layer 4.

## Error handling summary

| Failure | Behavior |
|---------|----------|
| Unknown interview (CLI/API) | `ValueError` / 404 |
| Unknown lens | `ValueError("Unknown lens")` via `load_lens` / 422 |
| Projection lag (guard trips) | Retryable error / 409, no partial bundle |
| Zero lens items | Valid bundle (interview + transcript + claims + note) |
| Filesystem error | Render fully in memory first; abort before partial writes |
| Malformed front matter | Warning; treated as body text; ingest proceeds |
| Empty query results | 200 with empty list |

## Testing strategy

- **Unit** (no infra, no LLM):
  - `renderer`: dicts → files; **OKF conformance test** — every generated
    non-reserved file parses YAML frontmatter with non-empty `type`; `index.md`
    has no frontmatter; all internal links bundle-absolute and resolvable within
    the generated file set.
  - front-matter parser: extraction, tolerant degradation, offsets invariant
    (`source[start:end] == fragment` with front matter present).
  - speaker seeding: labeled match → confirmed speaker; prompt-string assertion
    for the inference hint.
  - `reader`: Cypher shape via mocked tx (existing handler-test pattern).
  - routers: patched reader (existing router-test pattern).
- **Integration** (live test ESDB + Neo4j, canned LLM):
  - extend the Layer 3 smoke: ingest a transcript WITH front matter → canned lens
    → replay through the real registry → `OkfExporter.export` from real Neo4j →
    assert bundle structure + OKF conformance + grounding links.
  - query endpoints against the same projected graph.

## Non-goals (M4.4)

- GraphRAG retrieval (later phase per parent spec).
- Corpus-level / multi-interview bundles.
- Incremental or diff exports (full re-render each run).
- Importing arbitrary OKF bundles (ingest reads front matter only).
- Cross-interview speaker identity resolution (Layer 4).
- Export audit events (export is pure egress; no new event types anywhere in M4.4).
