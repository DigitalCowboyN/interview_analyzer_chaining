# Interview Analyzer

![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: active development](https://img.shields.io/badge/status-active%20development-orange)
![Event-sourced](https://img.shields.io/badge/architecture-event--sourced-blueviolet)

Turn interview transcripts into a queryable knowledge graph — speakers,
utterances, entities, claims, topics, cross-interview personas, and
purpose-built "lens" views — with every fact grounded back to the exact words
someone said, and every AI guess correctable by a human.

**Who it's for:** anyone who works through interviews at volume and needs the
output to be *trustworthy* — user researchers synthesizing across sessions,
analysts pulling decisions and action items out of meetings, journalists or
investigators tracing claims back to who said what. If a spreadsheet of
AI-summarized quotes isn't good enough because you need to defend every line,
this is built for that: nothing is inferred without a confidence score, nothing
is stored without a link back to the source, and a human can correct anything
the AI got wrong.

<!-- SCREENSHOT: add a workbench/gallery screenshot here, e.g.
     ![Workbench](docs/images/workbench.png) — see note in README audit. -->
> 📸 *A UI screenshot goes here — the workbench (transcript + inline
> corrections) and gallery (persona/person cards, review worklist). Not yet
> captured; see [The UI](#the-ui) for what's on screen.*

The system is **event-sourced**: EventStoreDB holds the full history of what
happened, and a projection service is the only thing that writes to the Neo4j
read model. Nothing is ever silently overwritten; corrections are new events,
not edits in place.

## Status

**Working today:** ingestion + speaker attribution, the full enrichment
pipeline, two lenses (meeting minutes, persona), topic segments, cross-interview
identity resolution, OKF export, ask-the-corpus (GraphRAG), and a live
two-surface web UI (workbench + gallery) that updates without a refresh.

**Not yet:** real authentication (there's a dev identity switcher instead), and
a few UI affordances still on the roadmap.

Actively developed. See [docs/ROADMAP.md](docs/ROADMAP.md) for the milestone
history and current test/coverage numbers, and
[docs/architecture/](docs/architecture/) for the diagrams.

## What it does

You give it a transcript — a labeled interview, a raw wall of unlabeled prose,
or something messy in between — and it works through several passes:

- **Ingests and maps.** The transcript is split into offset-grounded fragments
  with spaCy. Every fragment records exactly where it came from in the source,
  so nothing downstream is untraceable.
- **Attributes speakers.** Labels are parsed when present; when they're absent,
  speakers are inferred with a confidence score. Every attribution can be
  overridden by a human, and a human correction locks against later re-runs.
- **Stitches interruptions.** Utterances split across an interruption are
  reconnected as a relationship overlay — the verbatim text is never rewritten,
  but "who interrupted whom" becomes queryable.
- **Enriches.** A registry of focused extractors runs one LLM call per
  dimension (function, structure, purpose, topics, keywords, entities, claims).
  Each call is schema-checked and carries a numeric confidence, behind a
  provider failover chain (Anthropic Haiku → Claude Code → OpenAI).
- **Embeds.** Fragments and utterances get vector embeddings in per-model Neo4j
  indexes for semantic search.
- **Applies lenses.** A lens is a purpose-built reading of an interview —
  *meeting minutes* (objectives, decisions, action items, follow-ups) or
  *persona* (traits, goals, pain points, notable quotes). One generic engine
  serves every lens; adding one is a YAML profile plus a prompts file, no code.
  A human override on a lens item locks it against future re-runs.
- **Segments by topic.** Utterances are grouped into topic episodes (Layer 4),
  a pure overlay over the fragment sequence — correctable, never a rewrite.
- **Resolves identity.** Speakers across interviews get linked to canonical
  Persons; entity surface forms get canonicalized (merge / split / alias).
  These are human-in-the-loop decisions surfaced in a review worklist.
- **Exports and answers.** Any lens can be exported as an **OKF bundle** (Open
  Knowledge Format — a folder of Markdown files with YAML front matter,
  git-versionable and grounded back to the transcript). And you can ask the
  corpus a question and get a cited answer (hybrid graph + vector retrieval,
  GraphRAG-style).

Everything above is stored as events first. Neo4j is a projection of those
events, rebuildable from scratch at any time.

## How it works

```
Ingestion / correction commands
        │  (produce events)
        ▼
   Aggregates ── Interview · Sentence(=Fragment) · Project
        │
        ▼
   EventStoreDB  ◀── source of truth (append-only history)
        │  (catch-up subscriptions)
        ▼
   Projection service  ── the ONLY writer to Neo4j; replays events in
        │                  commit-position order per lane (reorder buffer)
        ▼
      Neo4j  ── read model: fragments, speakers, utterances, claims,
                lens items, persons, entities, segments, topics
```

Two ideas do most of the work:

- **Event sourcing.** Commands validate against an aggregate and emit events;
  the events are the record. The Neo4j graph is derived, so a bad projection is
  a rebuild, not a data-loss incident.
- **CQRS.** The write side (aggregates, commands, corrections) and the read side
  (Neo4j queries, the UI's gallery) are separate. The UI mirrors this split: a
  **workbench** for making changes, a **gallery** for reading them back.

The projection service processes events across parallel lanes but releases them
to Neo4j in each stream's causal (commit-position) order, with a bounded
reorder buffer and a redrive path for events whose referents aren't ready yet
(see M4.9 in the roadmap). This is what makes the live UI trustworthy: what you
see projected matches the order things actually happened.

For the full picture — the layered "Mine" model (Layer 1 conversation
structure, Layer 2 enrichment, Layer 3 lenses, Layer 4 segments, Layer 5
export), the three aggregates and their events, and the Neo4j schema — see
[docs/architecture/](docs/architecture/).

## Technology

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10 |
| API | FastAPI + Uvicorn | 0.117+ |
| Event store | EventStoreDB | 23.10 |
| Read model | Neo4j | 5.26 |
| Background work | Celery + Redis | 5.5 / 7 |
| NLP | spaCy | 3.8 |
| LLM providers | Anthropic, OpenAI (+ Claude Code) | — |
| UI | Next.js 15 + React + TanStack Query | — |

## Quick start

**Heads up before you start:** this is a multi-service system, not a
single-binary demo. `docker compose up` brings up EventStoreDB, Neo4j, and Redis
alongside the app, and the enrichment pipeline calls out to **paid LLM APIs** —
you'll need both an OpenAI and an Anthropic key to run the full pipeline. It
takes a few minutes and a bit of RAM, not two seconds.

You need Docker, Docker Compose, and Git.

```bash
git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining
cd interview_analyzer_chaining

# API keys and DB password
cat > .env <<'EOF'
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
NEO4J_PASSWORD=your_password_here
EOF

docker compose up -d
```

The projection service creates its own Neo4j schema (indexes and constraints) at
startup and fails fast if Neo4j is unreachable. To apply the schema standalone,
run `python -m src.projections.ensure_schema`. `make deployed-smoke` proves the
whole dockerized ingest → projection path against real containers.

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Neo4j browser | http://localhost:7474 |
| EventStoreDB UI | http://localhost:2113 |

## Using it from the command line

The pipeline is a handful of `python -m` entry points. A typical run:

```bash
# Ingest + enrich a transcript (Layer 1 + Layer 2) in one shot
python -m src.ingestion data/input/interview.txt --enrich
#   ...or the whole input directory at once:
make ingest FILE=data/input/interview.txt

# Apply a lens (Layer 3) — meeting_minutes or persona, same engine
python -m src.lens <interview_id> meeting_minutes

# Export a lens as an OKF bundle (Layer 5)
python -m src.export <interview_id> meeting_minutes

# Ask the corpus a question (GraphRAG), CLI or API
python -m src.ask <project_id> "What did they decide about Acme Corp?"
#   ...or: POST /ask/{project_id}
```

Want something to try this on? `data/samples/` has transcripts covering the
range — clean labeled interviews, adversarial/mixed labeling, and raw unlabeled
prose. `data/samples/MANIFEST.md` maps each file to the capability it exercises.

## What the output looks like

Exporting the `meeting_minutes` lens writes a folder of grounded Markdown — one
file per item, each linking back to who said it and where:

```
meeting_minutes_bundle/
├── index.md              # overview + links to every item
├── transcript.md         # the verbatim transcript, anchored per utterance
├── speakers/
│   └── alice-johnson.md
├── decisions/
│   └── go-with-acme-corp.md
├── action-items/
│   └── draft-the-doc.md
└── objectives/
    └── choose-a-vendor.md
```

A single decision file (`decisions/go-with-acme-corp.md`):

```markdown
---
type: Decision
title: Go with Acme Corp
lens: meeting_minutes
lens_version: 1
confidence: 0.92
model: claude-3-haiku
provider: anthropic
locked: false
---

Go with Acme Corp for the vendor contract.

**DECIDED_BY:** [Alice Johnson](/speakers/alice-johnson.md)

Grounded in:
> We'll go with Acme Corp and I'll draft the doc by Friday. (/transcript.md#u-1)
```

The `confidence`, `model`, and `provider` tell you *how much to trust it*; the
blockquote and `/transcript.md#u-1` anchor let you jump straight to the words it
came from. `locked: true` would mean a human corrected this item, so re-running
the lens leaves it alone.

## The UI

A Next.js 15 app in `frontend/`, two surfaces mirroring the backend's CQRS
split:

- **Workbench** (write side): projects → interviews → transcript, with inline
  corrections — text edits, speaker rename/reattribute, segment removal,
  lens-item overrides, and manual speaker→person linking. Every change goes
  through a correction endpoint as a command (fire → pending → bounded
  confirm-poll → settled), never a direct state mutation.
- **Gallery** (read side): persona and person cards, their core views, and an
  actionable review worklist.

**Live updates.** Both surfaces update themselves as events project — a new
line, a correction, a linked speaker, a re-run lens — with no manual refresh. A
backend SSE endpoint (`GET /ui/streams/events`) watches the event store and
pushes thin, surface-tagged notifications; the browser reacts by invalidating
the matching queries, with a debounced trailing re-fetch to absorb projection
lag. A subtle header dot shows the connection state, and if SSE is unavailable
the UI quietly falls back to fetch-on-navigation. (M5.1 brought the workbench
live; M5.1b extended it to the gallery, including persona-lens content.)

```bash
make ui-dev   # cd frontend && npm run dev — http://localhost:3000
```

`next.config.ts` rewrites the frontend's same-origin `/api/*` calls to the
FastAPI backend at `:8000` (no CORS), so start the backend separately
(`make run` or `make run-api`) for the UI to have data. The SSE stream is the
one exception — it's served through a Next.js route handler, because the
rewrite buffers streaming responses.

**Identity.** There's no real auth yet. Every request carries an `X-User-ID`
header from a small dev identity switcher in the app header (localStorage,
defaults to `"dev"`), so corrections are attributed to whoever's selected.

**Types.** The API client is generated against the backend's OpenAPI schema.
After any backend contract change, run `make ui-typegen`; `npm run
typegen:check` (in `frontend/`) fails on drift.

**Gates.**

```bash
make ui-test    # lint + typecheck + vitest
make ui-build   # production build
make ui-smoke   # Playwright: corrections settle, and a server-side line append
                # appears LIVE via the SSE feed (env-gated, needs the dev stack)
```

## Project layout

```
src/
├── ingestion/     # transcript → fragments, speaker genesis, stitching (Layer 1)
├── enrichment/    # extractor registry, provider chain, claims/entities (Layer 2)
├── lens/          # generic lens engine + profiles (Layer 3)
├── resolution/    # persons, entity canonicalization (identity)
├── export/        # OKF bundle export (Layer 5)
├── ask/           # GraphRAG ask-the-corpus
├── events/        # aggregates, event store, repository (event sourcing core)
├── projections/   # event → Neo4j projection service (sole writer)
├── commands/      # CQRS command handlers
├── ui/            # SSE notification bridge for the live feed
├── api/routers/   # FastAPI endpoints
├── agents/        # LLM provider adapters
└── main.py        # FastAPI app + batch ingest/enrich CLI

frontend/          # Next.js UI (workbench + gallery)
docs/
├── architecture/  # system, data flow, event sourcing, schema
├── onboarding/    # setup and dev-workflow guides
├── archive/       # historical milestone notes (superseded)
└── ROADMAP.md     # milestones, status, test/coverage stats
tests/             # unit, integration, e2e (counts in ROADMAP.md)
```

## Selected API endpoints

Full, always-current docs live at http://localhost:8000/docs. A few worth
knowing:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ui/streams/events` | GET | SSE live feed (surface-tagged notifications) |
| `/interviews/{id}/lenses/{lens}/items` | GET | A lens's items for an interview |
| `/lenses/{id}/items/{item_id}/override` | POST | Correct a lens item (locks it) |
| `/resolution/{project_id}/persons/{person_id}/link` | POST | Link a speaker to a person |
| `/interviews/{id}/segments` | GET | Topic segments for an interview |
| `/exports/{interview_id}/{lens_name}` | GET | Download an OKF bundle |
| `/review/worklist` | GET | Low-confidence + unresolved-reference review queue |
| `/speakers/rollup` | GET | Speaker rollup by display name, across interviews |
| `/ask/{project_id}` | POST | Ask the corpus (cited GraphRAG synthesis) |
| `/edits/sentences/{id}/{index}/edit` | POST | Edit fragment text |

## Common commands

```bash
make run          # start all services (docker compose up -d)
make run-api      # run the FastAPI app locally
make ingest FILE=<path>   # ingest + enrich a transcript
make test         # backend test suite
make ui-dev       # frontend dev server
make clean        # stop and remove containers
make help         # everything else
```

## Documentation

| Document | What's in it |
|----------|--------------|
| [docs/index.md](docs/index.md) | **The guarded knowledge graph** — how this repo documents itself and stays honest |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Milestones, status, test/coverage stats |
| [docs/architecture/](docs/architecture/) | System diagrams, data flow, event sourcing, schema |
| [docs/onboarding/](docs/onboarding/) | Setup, configuration, troubleshooting, dev workflow |

**The guarded knowledge graph.** Beyond prose docs, the repo keeps a graph *over its own
codebase*: small Markdown domains that catalog the system's decisions, vocabulary, code
map, capabilities, use-cases, and tests — each reconciling itself against the real code via
a non-blocking `make <domain>-check`, so documentation drift is surfaced instead of
rotting. The domains form a traceability spine — **use-case → capability → code → test** —
with derived implementation *and* verification coverage. Start at
[docs/index.md](docs/index.md).

## Contributing

Issues, feature branches, tests, PRs — the usual. Style is enforced with `black`
and `flake8`; the frontend with ESLint. Run `make test` (and `make ui-test` for
UI changes) before opening a PR.

## License

MIT — see [LICENSE](LICENSE).
