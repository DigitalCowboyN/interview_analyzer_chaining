# Code map

See `pipeline.md` for the dependency graph.

## agent

| unit | io | depends_on |
| --- | --- | --- |
| agents | LLM | utils |
| agents.agent_factory | LLM | utils |

## infrastructure

| unit | io | depends_on |
| --- | --- | --- |
| commands | ESDB | events |
| events | ESDB | utils |
| io |  |  |
| persistence | Neo4j |  |
| utils | ESDB, HTTP, LLM, Neo4j, files |  |

## model

| unit | io | depends_on |
| --- | --- | --- |
| models | LLM |  |

## pipeline-layer

| unit | io | depends_on |
| --- | --- | --- |
| enrichment | ESDB, LLM | agents, events, utils |
| enrichment.executor | LLM | agents, utils |
| enrichment.orchestrator | ESDB, LLM | agents, events, utils |
| export | ESDB, Neo4j, files | events, lens, utils |
| export.bundler | ESDB, Neo4j, files | events, lens, utils |
| export.reader | Neo4j |  |
| export.renderer | Neo4j | lens |
| ingestion | ESDB, LLM, Neo4j, files | agents, enrichment, events, models, utils |
| ingestion.orchestrator | ESDB, Neo4j, files | events, utils |
| ingestion.speaker_inference | LLM | agents, models, utils |
| ingestion.stitcher | LLM | agents, models, utils |
| lens | ESDB, LLM, files | agents, enrichment, events, utils |
| lens.engine | ESDB, LLM | agents, enrichment, events, utils |
| projections | ESDB, Neo4j | enrichment, events, utils |
| resolution | ESDB, Neo4j | enrichment, events, utils |
| resolution.engine | ESDB, Neo4j | enrichment, events, utils |

## surface

| unit | io | depends_on |
| --- | --- | --- |
| api | ESDB, HTTP, Neo4j, files | ask, commands, enrichment, events, export, ingestion, resolution, ui, utils |
| ask | LLM, Neo4j | agents, enrichment, projections, utils |
| ask.engine | LLM, Neo4j | agents, enrichment, projections, utils |
| ask.reader | Neo4j |  |
| ui | ESDB, HTTP, Neo4j | events |
| ui.reader | ESDB, Neo4j |  |
