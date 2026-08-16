# Code map

Derived from `src/` and `tools/`. See `pipeline.md` for the dependency graph.

## Packages

| unit | category | determinism | depends_on |
| --- | --- | --- | --- |
| agents | product | deterministic |  |
| api | product | deterministic |  |
| api.routers |  | deterministic |  |
| ask | product | probabilistic |  |
| commands | product | deterministic |  |
| enrichment | product | probabilistic |  |
| events | product | deterministic |  |
| export | product | deterministic |  |
| ingestion | product | probabilistic |  |
| lens | product | probabilistic |  |
| models | product | deterministic |  |
| persistence |  | deterministic |  |
| projections | product | deterministic |  |
| projections.handlers |  | deterministic |  |
| resolution | product | deterministic |  |
| tools.adr |  | deterministic |  |
| tools.api |  | deterministic |  |
| tools.capability |  | deterministic |  |
| tools.cli |  | deterministic |  |
| tools.code |  | deterministic |  |
| tools.corpus |  | deterministic |  |
| tools.glossary |  | deterministic |  |
| tools.graph |  | deterministic |  |
| tools.graphq |  | deterministic |  |
| tools.knowledge |  | deterministic |  |
| tools.prompts |  | deterministic |  |
| tools.testmap |  | deterministic |  |
| tools.usecase |  | deterministic |  |
| ui | product | deterministic |  |
| utils |  | deterministic |  |

## Modules

| unit | category | determinism | depends_on |
| --- | --- | --- | --- |
| agents.agent_factory |  | deterministic | config, utils.logger |
| agents.anthropic_agent |  | probabilistic | agents.agent_factory, config, utils.logger, utils.metrics |
| agents.base_agent |  | deterministic |  |
| agents.claude_code_agent |  | deterministic | config, utils.logger |
| agents.failover_agent |  | probabilistic | agents.agent_factory, config, utils.logger |
| agents.openai_agent |  | probabilistic | agents.agent_factory, config, utils.logger, utils.metrics |
| api.routers.analysis |  | deterministic | api.schemas, config, enrichment.orchestrator, ingestion.orchestrator, utils.logger |
| api.routers.ask |  | deterministic | ask.engine |
| api.routers.edits |  | deterministic | commands.handlers, commands.sentence_commands, config, events.envelope, events.repository, events.store, utils.environment, utils.logger |
| api.routers.exports |  | deterministic | export.bundler, utils.logger |
| api.routers.files |  | deterministic | api.schemas, config, utils.logger |
| api.routers.lenses |  | deterministic | events.aggregates, events.envelope, events.repository, utils.logger |
| api.routers.queries |  | deterministic | config, enrichment.embedder, events.project_events, events.repository, export, export.renderer, resolution.suggestions, utils.neo4j_driver |
| api.routers.resolution |  | deterministic | events.envelope, events.project_events, events.repository, resolution.candidates |
| api.routers.segments |  | deterministic | events.aggregates, events.envelope, events.repository, export, utils.neo4j_driver |
| api.routers.speakers |  | deterministic | events.aggregates, events.envelope, events.repository, utils.logger |
| api.routers.ui |  | deterministic | events.project_events, resolution.candidates, ui, ui.notifications, utils.neo4j_driver |
| api.schemas |  | deterministic |  |
| ask.__main__ |  | deterministic | ask.engine |
| ask.context |  | deterministic |  |
| ask.engine | product | probabilistic | agents.failover_agent, config, enrichment.embedder, projections.handlers.embedding_handlers, utils.helpers, utils.logger, utils.neo4j_driver |
| ask.fusion |  | deterministic |  |
| ask.reader | product | deterministic |  |
| celery_app | product | deterministic |  |
| commands.handlers |  | deterministic | events.aggregates, events.envelope, events.interview_events, events.repository, events.sentence_events, events.store |
| commands.interview_commands |  | deterministic |  |
| commands.sentence_commands |  | deterministic |  |
| config | product | deterministic |  |
| enrichment.__main__ |  | deterministic | enrichment.orchestrator |
| enrichment.embedder |  | deterministic | config, utils.logger |
| enrichment.executor | product | probabilistic | agents.failover_agent, enrichment.graph_context, enrichment.models, enrichment.syntax_check, utils.logger |
| enrichment.graph_context |  | deterministic |  |
| enrichment.models |  | deterministic |  |
| enrichment.orchestrator | product | probabilistic | agents.failover_agent, config, enrichment.embedder, enrichment.executor, enrichment.graph_context, enrichment.registry, enrichment.segments, events.aggregates, events.envelope, events.interview_events, events.repository, utils.helpers, utils.logger |
| enrichment.registry |  | deterministic | utils.helpers |
| enrichment.segments |  | deterministic |  |
| enrichment.syntax_check |  | deterministic | utils.text_processing |
| events.aggregates |  | deterministic |  |
| events.envelope |  | deterministic |  |
| events.interview_events |  | deterministic |  |
| events.project_events |  | deterministic |  |
| events.repository |  | deterministic |  |
| events.sentence_events |  | deterministic |  |
| events.store |  | deterministic | utils.environment |
| export.__main__ |  | deterministic | export.bundler |
| export.bundler | product | deterministic | config, events.repository, export.renderer, lens.models, utils.logger, utils.neo4j_driver |
| export.reader | product | deterministic |  |
| export.renderer | product | deterministic | lens.models |
| ingestion.__main__ |  | deterministic | enrichment.orchestrator, ingestion.orchestrator |
| ingestion.format_detector |  | deterministic |  |
| ingestion.front_matter |  | deterministic | utils.logger |
| ingestion.models |  | deterministic |  |
| ingestion.normalizer |  | deterministic | utils.text_processing |
| ingestion.orchestrator | product | deterministic | events.aggregates, events.envelope, events.repository, ingestion.models, ingestion.normalizer, ingestion.speaker_inference, ingestion.stitcher, utils.helpers, utils.logger |
| ingestion.speaker_inference | product | probabilistic | agents.agent_factory, ingestion.models, models.ingestion_responses, utils.helpers, utils.logger |
| ingestion.stitcher | product | probabilistic | agents.agent_factory, ingestion.models, ingestion.speaker_inference, models.ingestion_responses, utils.helpers, utils.logger |
| lens.__main__ |  | deterministic | lens.engine |
| lens.engine | product | probabilistic | agents.failover_agent, config, enrichment.executor, events.aggregates, events.envelope, events.repository, lens.models, utils.helpers, utils.logger |
| lens.models |  | deterministic | enrichment.models, utils.helpers |
| main | product | deterministic | api.routers, config, enrichment.orchestrator, ingestion.orchestrator, utils.logger, utils.metrics |
| models.analysis_result |  | deterministic |  |
| models.extractor_responses |  | deterministic |  |
| models.ingestion_responses |  | deterministic |  |
| models.lens_responses |  | deterministic | models.extractor_responses |
| persistence.graph_persistence |  | deterministic |  |
| projections.bootstrap |  | deterministic | events.store, projections.handlers.claim_handlers, projections.handlers.embedding_handlers, projections.handlers.entity_handlers, projections.handlers.interview_handlers, projections.handlers.lens_handlers, projections.handlers.registry, projections.handlers.resolution_handlers, projections.handlers.segment_handlers, projections.handlers.sentence_handlers, projections.handlers.speaker_handlers, projections.handlers.utterance_handlers, projections.parked_events |
| projections.config |  | deterministic | utils.environment |
| projections.ensure_schema |  | deterministic | projections.schema, utils.neo4j_driver |
| projections.handlers.base_handler |  | deterministic | events.envelope, utils.neo4j_driver |
| projections.handlers.claim_handlers |  | deterministic | events.envelope |
| projections.handlers.embedding_handlers |  | deterministic | enrichment.embedder, events.envelope, utils.neo4j_driver |
| projections.handlers.entity_handlers |  | deterministic | events.envelope |
| projections.handlers.interview_handlers |  | deterministic | events.envelope, utils.metrics |
| projections.handlers.lens_handlers |  | deterministic | events.envelope |
| projections.handlers.registry |  | deterministic | events.envelope |
| projections.handlers.resolution_handlers |  | deterministic | events.envelope, projections.handlers.base_handler, utils.logger |
| projections.handlers.segment_handlers |  | deterministic | events.envelope, projections.handlers.base_handler, utils.logger |
| projections.handlers.sentence_handlers |  | deterministic | events.envelope, utils.metrics |
| projections.handlers.speaker_handlers |  | deterministic | events.envelope |
| projections.handlers.utterance_handlers |  | deterministic | events.envelope |
| projections.health |  | deterministic |  |
| projections.lane_manager |  | deterministic | events.envelope |
| projections.metrics |  | deterministic |  |
| projections.migrate_shim_drop |  | deterministic | utils.neo4j_driver |
| projections.parked_events |  | deterministic | events.envelope, events.store |
| projections.projection_service |  | deterministic | events.store |
| projections.redrive |  | deterministic | projections.bootstrap, projections.handlers.registry, projections.handlers.speaker_handlers, projections.parked_events |
| projections.reorder_buffer |  | deterministic |  |
| projections.schema |  | deterministic |  |
| projections.subscription_manager |  | deterministic | events.envelope, events.store |
| resolution.__main__ |  | deterministic | resolution.engine |
| resolution.candidates |  | deterministic |  |
| resolution.engine | product | deterministic | config, enrichment.embedder, events.aggregates, events.envelope, events.project_events, events.repository, resolution.candidates, resolution.reader, utils.logger, utils.neo4j_driver |
| resolution.reader |  | deterministic |  |
| resolution.suggestions |  | deterministic | events.aggregates, events.project_events, resolution.candidates, resolution.reader |
| run_projection_service | product | deterministic | events.store, projections.bootstrap, projections.config, projections.projection_service, projections.schema, utils.neo4j_driver |
| tasks | product | deterministic | celery_app, enrichment.orchestrator, ingestion.orchestrator, utils.logger |
| tools.adr.__main__ |  | deterministic | tools.adr.check, tools.adr.index, tools.adr.intent, tools.adr.scaffold |
| tools.adr.check |  | deterministic | ingestion.front_matter, tools.adr.code_links, tools.adr.index, tools.adr.model |
| tools.adr.code_links |  | deterministic |  |
| tools.adr.index |  | deterministic | tools.adr.model |
| tools.adr.intent |  | deterministic |  |
| tools.adr.model |  | deterministic | ingestion.front_matter |
| tools.adr.scaffold |  | deterministic | tools.adr.index |
| tools.api.__main__ |  | deterministic | tools.api.check, tools.api.reader, tools.api.render |
| tools.api.check |  | deterministic | tools.api.reader, tools.api.render |
| tools.api.reader |  | deterministic | main |
| tools.api.render |  | deterministic | tools.api.reader |
| tools.capability.__main__ |  | deterministic | tools.capability.check, tools.capability.reader, tools.capability.render |
| tools.capability.check |  | deterministic | tools.capability.reader, tools.capability.render, tools.code.reader |
| tools.capability.reader |  | deterministic | ingestion.front_matter, tools.code.reader |
| tools.capability.render |  | deterministic | tools.capability.reader |
| tools.cli.__main__ |  | deterministic | tools.cli.check, tools.cli.reader, tools.cli.render |
| tools.cli.check |  | deterministic | tools.cli.reader, tools.cli.render |
| tools.cli.reader |  | deterministic |  |
| tools.cli.render |  | deterministic | tools.cli.reader |
| tools.code.__main__ |  | deterministic | tools.code.check, tools.code.reader, tools.code.render, tools.graph.classify |
| tools.code.check |  | deterministic | tools.code.reader, tools.code.render, tools.graph.classify |
| tools.code.reader |  | deterministic |  |
| tools.code.render |  | deterministic | tools.code.reader |
| tools.corpus.__main__ |  | deterministic | tools.corpus.check, tools.corpus.reader |
| tools.corpus.check |  | deterministic | ingestion.front_matter, tools.corpus.model, tools.corpus.reader |
| tools.corpus.model |  | deterministic |  |
| tools.corpus.reader |  | deterministic | ingestion.front_matter, tools.corpus.model |
| tools.glossary.__main__ |  | deterministic | tools.glossary.check, tools.glossary.model, tools.glossary.render, tools.glossary.scaffold |
| tools.glossary.check |  | deterministic | tools.glossary.model, tools.glossary.reader, tools.glossary.render |
| tools.glossary.model |  | deterministic | ingestion.front_matter |
| tools.glossary.reader |  | deterministic |  |
| tools.glossary.render |  | deterministic | tools.glossary.model |
| tools.glossary.scaffold |  | deterministic | tools.glossary.reader |
| tools.graph.__main__ |  | deterministic | tools.graph.check, tools.graph.reader, tools.graph.render, tools.graph.traverse |
| tools.graph.check |  | deterministic | tools.graph.reader, tools.graph.registry, tools.graph.render, tools.graph.traverse |
| tools.graph.classify |  | deterministic | tools.capability.reader, tools.code.reader, tools.graph.reader |
| tools.graph.reader |  | deterministic | tools.adr.index, tools.capability.reader, tools.code.reader, tools.glossary.model, tools.graph.registry, tools.graphq.reader, tools.prompts.reader, tools.testmap.reader, tools.usecase.reader |
| tools.graph.registry |  | deterministic |  |
| tools.graph.render |  | deterministic | tools.graph.reader, tools.graph.registry |
| tools.graph.traverse |  | deterministic | tools.adr.index, tools.capability.reader, tools.code.reader, tools.glossary.model, tools.graph.reader, tools.graph.registry, tools.graphq.reader, tools.prompts.reader, tools.testmap.reader, tools.usecase.reader |
| tools.graphq.__main__ |  | deterministic | tools.graphq.check, tools.graphq.reader, tools.graphq.render |
| tools.graphq.check |  | deterministic | tools.glossary.reader, tools.graphq.reader, tools.graphq.render |
| tools.graphq.reader |  | deterministic |  |
| tools.graphq.render |  | deterministic | tools.graphq.reader |
| tools.knowledge.__main__ |  | deterministic | tools.knowledge.check, tools.knowledge.surfaces |
| tools.knowledge.check |  | deterministic | tools.capability.reader, tools.usecase.reader |
| tools.knowledge.surfaces |  | deterministic | tools.knowledge.check |
| tools.prompts.__main__ |  | deterministic | tools.prompts.check, tools.prompts.reader, tools.prompts.render |
| tools.prompts.check |  | deterministic | tools.glossary.model, tools.prompts.reader, tools.prompts.render |
| tools.prompts.reader |  | deterministic |  |
| tools.prompts.render |  | deterministic | tools.prompts.reader |
| tools.testmap.__main__ |  | deterministic | tools.capability.reader, tools.testmap.check, tools.testmap.reader, tools.testmap.render, tools.testmap.verification, tools.usecase.reader |
| tools.testmap.check |  | deterministic | tools.capability.reader, tools.testmap.reader, tools.testmap.render, tools.testmap.verification, tools.usecase.reader |
| tools.testmap.reader |  | deterministic | tools.capability.reader |
| tools.testmap.render |  | deterministic | tools.testmap.reader |
| tools.testmap.verification |  | deterministic | tools.capability.reader, tools.testmap.reader, tools.usecase.reader |
| tools.usecase.__main__ |  | deterministic | tools.capability.reader, tools.usecase.check, tools.usecase.coverage, tools.usecase.reader, tools.usecase.render |
| tools.usecase.check |  | deterministic | tools.capability.reader, tools.usecase.coverage, tools.usecase.reader, tools.usecase.render |
| tools.usecase.coverage |  | deterministic | tools.capability.reader, tools.usecase.reader |
| tools.usecase.reader |  | deterministic | ingestion.front_matter |
| tools.usecase.render |  | deterministic | tools.capability.reader, tools.usecase.coverage, tools.usecase.reader |
| ui.notifications |  | deterministic | events.store |
| ui.reader | product | deterministic |  |
| utils.environment |  | deterministic |  |
| utils.helpers |  | deterministic |  |
| utils.logger |  | deterministic | config |
| utils.metrics |  | deterministic |  |
| utils.neo4j_driver |  | deterministic | config, utils.environment, utils.logger |
| utils.path_helpers |  | deterministic |  |
| utils.text_processing |  | deterministic | utils.logger |
| utils.visualize |  | deterministic |  |
