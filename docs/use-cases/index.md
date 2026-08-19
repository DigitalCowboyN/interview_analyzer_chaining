# Use-Cases

The user-centered intents this system serves — the "why" above the capabilities (`../capabilities/`). Coverage is derived from `fulfilled_by`, never stored.

## product

### user-story

#### export-my-analysis-to-take-elsewhere — FULLY_COVERED
As a researcher who needs to hand findings to a stakeholder or archive them outside this system, I want my analysis exported as a portable, human-readable bundle, so my work doesn't live only inside a database I don't control.

- **actor:** researcher
- **fulfilled_by:** export-a-portable-bundle, assemble-bundle, render-bundle
- **acceptance_criteria:** 1

#### navigate-the-transcript-by-meaning — FULLY_COVERED
As an analyst facing a wall of undifferentiated transcript text, I want every fragment classified by what kind of thing it is — a claim, a topic, a purpose — so I can navigate by meaning instead of scrolling through raw dialogue.

- **actor:** analyst
- **fulfilled_by:** enrich-fragments, classify-dimensions, extract-claims, tag-topics-keywords
- **acceptance_criteria:** 1

#### onboard-my-transcripts — PARTIALLY_COVERED
As a researcher with a backlog of past interviews, I want to bring my existing transcripts into the system so my prior work is analyzable, not stranded outside it.

- **actor:** researcher
- **fulfilled_by:** import-transcripts
- **acceptance_criteria:** 1

### feature

#### correct-what-the-system-got-wrong — FULLY_COVERED
As an analyst who spots something the AI got wrong — a misattributed line, a bad resolution, an invented action item — I want to correct it directly and trust that correction is recorded as history, not quietly overwritten, so I can rely on the output without losing an audit trail.

- **actor:** analyst
- **fulfilled_by:** correct-the-analysis, correct-resolution, edit-text, override-lens-items, remove-segments, rename-reattribute-speakers
- **acceptance_criteria:** 1

#### keep-the-workbench-live — FULLY_COVERED
As an analyst working across many interviews at once alongside others touching the same corpus, I want a workbench and gallery that update themselves the moment something changes, so I'm never staring at stale analysis while I work.

- **actor:** analyst
- **fulfilled_by:** serve-workbench-and-gallery, gallery-read, live-notifications, run-read-queries, workbench-write
- **acceptance_criteria:** 1

#### know-who-said-what-across-interviews — FULLY_COVERED
As a researcher studying dozens of interviews, I want the same person and the same real-world entity recognized as one identity everywhere they appear, so I can trace someone's story across the whole corpus instead of re-learning who's who in every transcript.

- **actor:** researcher
- **fulfilled_by:** resolve-entities-and-people, canonicalize-entities, merge-split-link-alias, resolve-persons
- **acceptance_criteria:** 2

#### structure-the-raw-transcript — FULLY_COVERED
As an analyst handed a raw, unlabeled transcript, I want it turned into clean, speaker-attributed, stitched utterances automatically so I can start reading for meaning instead of first doing manual transcript cleanup.

- **actor:** analyst
- **fulfilled_by:** ingest-transcripts, infer-speakers, parse-fragments, segment-conversation, stitch-utterances
- **acceptance_criteria:** 2

### requirement

#### collaborate-with-my-team-on-a-corpus — NOT_COVERED
As an analyst collaborating with a colleague on the same corpus, I want to share a finding or leave a note directly on a fragment or lens item, so we can discuss it in context instead of screenshotting things into Slack.

- **actor:** analyst
- **fulfilled_by:** —
- **acceptance_criteria:** 1

#### revisit-a-past-extraction — NOT_COVERED
As an analyst who has learned more since an early pass, I want to revisit and correct a past extraction so my conclusions improve as my understanding does — without redoing everything by hand.

- **actor:** analyst
- **fulfilled_by:** —
- **acceptance_criteria:** 1

### use-case

#### get-a-grounded-answer-from-my-corpus — FULLY_COVERED
As an analyst who suspects the answer to a question is buried somewhere across hundreds of interviews, I want to ask it in plain language and get one grounded, cited answer, so I don't have to manually search transcript by transcript.

- **actor:** analyst
- **fulfilled_by:** ask-the-corpus, cited-synthesis, hybrid-retrieval
- **acceptance_criteria:** 1

#### surface-the-signal — FULLY_COVERED
As an analyst drowning in raw interview transcripts, I want the meaningful signal surfaced automatically so I stop missing what matters across hundreds of conversations.

- **actor:** analyst
- **fulfilled_by:** extract-insights-via-lenses
- **acceptance_criteria:** 1

#### tailor-the-reading-to-the-audience — FULLY_COVERED
As a team lead who needs a specific shape of output — meeting minutes for a standup, a persona profile for a research readout — I want to apply a purpose-built lens and get exactly that structure back, without waiting for engineering to hand-write a new extractor every time I need a new reading.

- **actor:** team lead
- **fulfilled_by:** run-lens-engine, per-lens-extractors
- **acceptance_criteria:** 1

## operations

### feature

#### catalog-every-live-surface — FULLY_COVERED
As a maintainer who can't trust documentation to describe reality on its own, I want every surface the system exposes — its endpoints, commands, queries, prompts, capabilities — cataloged and checked against what's actually running, so stale docs get caught before they mislead the next person.

- **actor:** maintainer
- **fulfilled_by:** catalog-the-api-surface, catalog-the-cli-surface, catalog-the-graph-queries, catalog-the-prompt-registry, map-the-code, map-capabilities, map-use-cases
- **acceptance_criteria:** 1

#### govern-decisions-and-hold-specs-honest — FULLY_COVERED
As a maintainer trying to keep a fast-moving system's decisions honest, I want architectural choices captured durably with explicit supersession, and every new spec held accountable to the domains it touches, so decisions don't get silently overridden or forgotten as the system evolves.

- **actor:** maintainer
- **fulfilled_by:** govern-architectural-decisions, disclose-knowledge-and-check-specs, link-the-domains, maintain-the-glossary
- **acceptance_criteria:** 2

### requirement

#### survive-a-provider-outage — FULLY_COVERED
As an operator running analysis against external LLM providers I don't control, I want chat calls to fail over automatically to another provider when one goes down, so a single vendor outage doesn't stall the whole pipeline — while embeddings stay pinned, since mixing vector spaces would corrupt retrieval.

- **actor:** operator
- **fulfilled_by:** provider-strategy-and-focused-calls, chat-failover, pinned-embeddings
- **acceptance_criteria:** 2

#### trust-the-event-record — FULLY_COVERED
As an operator responsible for the system's integrity, I want every change captured as an immutable, replayable event and the read model rebuilt deterministically from it, so nothing is ever silently lost, and the graph can always be reconstructed from the one source of truth.

- **actor:** operator
- **fulfilled_by:** maintain-event-source-of-truth, project-events-to-graph
- **acceptance_criteria:** 2

### use-case

#### gather-context-with-the-graph — FULLY_COVERED
As a maintainer (often working through an AI agent that carries no memory forward), I want to walk the codebase's own knowledge graph to gather the correct, minimal context at the right layer for whatever task I'm on — tracing code up to the intent that governs it and out to what it relates to — so I spend effort on the task, not on re-reading the whole system.

- **actor:** maintainer
- **fulfilled_by:** walk-the-graph-for-context, link-the-domains, map-the-code
- **acceptance_criteria:** 3

#### keep-the-codebase-legible — PARTIALLY_COVERED
As a maintainer inheriting a system built and extended across many sessions — including by AI agents that don't carry memory forward — I want the codebase's own knowledge to explain itself, so work compounds instead of every session rediscovering the same ground.

- **actor:** maintainer
- **fulfilled_by:** maintain-a-guarded-knowledge-graph
- **acceptance_criteria:** 1

## supporting

### user-story

#### read-analysis-output-without-the-ui — FULLY_COVERED
As someone building a downstream tool or script that needs to consume a finished interview analysis, I want to list and fetch the output files directly, so I can integrate without reimplementing the workbench.

- **actor:** downstream tool integrator
- **fulfilled_by:** access-analysis-output-files
- **acceptance_criteria:** 1

### requirement

#### diagnose-a-stalled-analysis — NOT_COVERED
As a support engineer fielding "my interview never finished" tickets, I want visibility into where a specific interview is in the pipeline and why it stalled, so I can diagnose and unblock it without digging through server logs.

- **actor:** support engineer
- **fulfilled_by:** —
- **acceptance_criteria:** 1

#### notify-me-when-my-analysis-is-ready — NOT_COVERED
As an analyst who kicked off a long-running ingestion or enrichment job and stepped away, I want to be notified out-of-band when it finishes or fails, so I don't have to keep polling the workbench to find out.

- **actor:** analyst
- **fulfilled_by:** —
- **acceptance_criteria:** 1
