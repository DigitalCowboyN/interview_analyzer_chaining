# Code map

Derived from `src/` and `tools/`. See `pipeline.md` for the dependency graph.

## Packages

| unit | depends_on |
| --- | --- |
| agents |  |
| api |  |
| api.routers |  |
| ask |  |
| commands |  |
| enrichment |  |
| events |  |
| export |  |
| ingestion |  |
| lens |  |
| models |  |
| persistence |  |
| projections |  |
| projections.handlers |  |
| resolution |  |
| tools.adr |  |
| tools.api |  |
| tools.capability |  |
| tools.cli |  |
| tools.code |  |
| tools.corpus |  |
| tools.glossary |  |
| tools.graph |  |
| tools.graphq |  |
| tools.knowledge |  |
| tools.prompts |  |
| tools.testmap |  |
| tools.usecase |  |
| ui |  |
| utils |  |

## Modules

| unit | depends_on |
| --- | --- |
| agents.agent_factory | config, utils.logger |
| agents.anthropic_agent | agents.agent_factory, config, utils.logger, utils.metrics |
| agents.base_agent |  |
| agents.claude_code_agent | config, utils.logger |
| agents.failover_agent | agents.agent_factory, config, utils.logger |
| agents.openai_agent | agents.agent_factory, config, utils.logger, utils.metrics |
| api.routers.analysis | api.schemas, config, enrichment.orchestrator, ingestion.orchestrator, utils.logger |
| api.routers.ask | ask.engine |
| api.routers.edits | commands.handlers, commands.sentence_commands, config, events.envelope, events.repository, events.store, utils.environment, utils.logger |
| api.routers.exports | export.bundler, utils.logger |
| api.routers.files | api.schemas, config, utils.logger |
| api.routers.lenses | events.aggregates, events.envelope, events.repository, utils.logger |
| api.routers.queries | config, enrichment.embedder, events.project_events, events.repository, export, export.renderer, resolution.suggestions, utils.neo4j_driver |
| api.routers.resolution | events.envelope, events.project_events, events.repository, resolution.candidates |
| api.routers.segments | events.aggregates, events.envelope, events.repository, export, utils.neo4j_driver |
| api.routers.speakers | events.aggregates, events.envelope, events.repository, utils.logger |
| api.routers.ui | events.project_events, resolution.candidates, ui, ui.notifications, utils.neo4j_driver |
| api.schemas |  |
| ask.__main__ | ask.engine |
| ask.context |  |
| ask.engine | agents.failover_agent, config, enrichment.embedder, projections.handlers.embedding_handlers, utils.helpers, utils.logger, utils.neo4j_driver |
| ask.fusion |  |
| ask.reader |  |
| celery_app |  |
| commands.handlers | events.aggregates, events.envelope, events.interview_events, events.repository, events.sentence_events, events.store |
| commands.interview_commands |  |
| commands.sentence_commands |  |
| config |  |
| enrichment.__main__ | enrichment.orchestrator |
| enrichment.embedder | config, utils.logger |
| enrichment.executor | agents.failover_agent, enrichment.graph_context, enrichment.models, enrichment.syntax_check, utils.logger |
| enrichment.graph_context |  |
| enrichment.models |  |
| enrichment.orchestrator | agents.failover_agent, config, enrichment.embedder, enrichment.executor, enrichment.graph_context, enrichment.registry, enrichment.segments, events.aggregates, events.envelope, events.interview_events, events.repository, utils.helpers, utils.logger |
| enrichment.registry | utils.helpers |
| enrichment.segments |  |
| enrichment.syntax_check | utils.text_processing |
| events.aggregates |  |
| events.envelope |  |
| events.interview_events |  |
| events.project_events |  |
| events.repository |  |
| events.sentence_events |  |
| events.store | utils.environment |
| export.__main__ | export.bundler |
| export.bundler | config, events.repository, export.renderer, lens.models, utils.logger, utils.neo4j_driver |
| export.reader |  |
| export.renderer | lens.models |
| ingestion.__main__ | enrichment.orchestrator, ingestion.orchestrator |
| ingestion.format_detector |  |
| ingestion.front_matter | utils.logger |
| ingestion.models |  |
| ingestion.normalizer | utils.text_processing |
| ingestion.orchestrator | events.aggregates, events.envelope, events.repository, ingestion.models, ingestion.normalizer, ingestion.speaker_inference, ingestion.stitcher, utils.helpers, utils.logger |
| ingestion.speaker_inference | agents.agent_factory, ingestion.models, models.ingestion_responses, utils.helpers, utils.logger |
| ingestion.stitcher | agents.agent_factory, ingestion.models, ingestion.speaker_inference, models.ingestion_responses, utils.helpers, utils.logger |
| lens.__main__ | lens.engine |
| lens.engine | agents.failover_agent, config, enrichment.executor, events.aggregates, events.envelope, events.repository, lens.models, utils.helpers, utils.logger |
| lens.models | enrichment.models, utils.helpers |
| main | api.routers, config, enrichment.orchestrator, ingestion.orchestrator, utils.logger, utils.metrics |
| models.analysis_result |  |
| models.extractor_responses |  |
| models.ingestion_responses |  |
| models.lens_responses | models.extractor_responses |
| persistence.graph_persistence |  |
| projections.bootstrap | events.store, projections.handlers.claim_handlers, projections.handlers.embedding_handlers, projections.handlers.entity_handlers, projections.handlers.interview_handlers, projections.handlers.lens_handlers, projections.handlers.registry, projections.handlers.resolution_handlers, projections.handlers.segment_handlers, projections.handlers.sentence_handlers, projections.handlers.speaker_handlers, projections.handlers.utterance_handlers, projections.parked_events |
| projections.config | utils.environment |
| projections.ensure_schema | projections.schema, utils.neo4j_driver |
| projections.handlers.base_handler | events.envelope, utils.neo4j_driver |
| projections.handlers.claim_handlers | events.envelope |
| projections.handlers.embedding_handlers | enrichment.embedder, events.envelope, utils.neo4j_driver |
| projections.handlers.entity_handlers | events.envelope |
| projections.handlers.interview_handlers | events.envelope, utils.metrics |
| projections.handlers.lens_handlers | events.envelope |
| projections.handlers.registry | events.envelope |
| projections.handlers.resolution_handlers | events.envelope, projections.handlers.base_handler, utils.logger |
| projections.handlers.segment_handlers | events.envelope, projections.handlers.base_handler, utils.logger |
| projections.handlers.sentence_handlers | events.envelope, utils.metrics |
| projections.handlers.speaker_handlers | events.envelope |
| projections.handlers.utterance_handlers | events.envelope |
| projections.health |  |
| projections.lane_manager | events.envelope |
| projections.metrics |  |
| projections.migrate_shim_drop | utils.neo4j_driver |
| projections.parked_events | events.envelope, events.store |
| projections.projection_service | events.store |
| projections.redrive | projections.bootstrap, projections.handlers.registry, projections.handlers.speaker_handlers, projections.parked_events |
| projections.reorder_buffer |  |
| projections.schema |  |
| projections.subscription_manager | events.envelope, events.store |
| resolution.__main__ | resolution.engine |
| resolution.candidates |  |
| resolution.engine | config, enrichment.embedder, events.aggregates, events.envelope, events.project_events, events.repository, resolution.candidates, resolution.reader, utils.logger, utils.neo4j_driver |
| resolution.reader |  |
| resolution.suggestions | events.aggregates, events.project_events, resolution.candidates, resolution.reader |
| run_projection_service | events.store, projections.bootstrap, projections.config, projections.projection_service, projections.schema, utils.neo4j_driver |
| tasks | celery_app, enrichment.orchestrator, ingestion.orchestrator, utils.logger |
| tools.adr.__main__ | tools.adr.check, tools.adr.index, tools.adr.intent, tools.adr.scaffold |
| tools.adr.check | ingestion.front_matter, tools.adr.code_links, tools.adr.index, tools.adr.model |
| tools.adr.code_links |  |
| tools.adr.index | tools.adr.model |
| tools.adr.intent |  |
| tools.adr.model | ingestion.front_matter |
| tools.adr.scaffold | tools.adr.index |
| tools.api.__main__ | tools.api.check, tools.api.reader, tools.api.render |
| tools.api.check | tools.api.reader, tools.api.render |
| tools.api.reader | main |
| tools.api.render | tools.api.reader |
| tools.capability.__main__ | tools.capability.check, tools.capability.reader, tools.capability.render |
| tools.capability.check | tools.capability.reader, tools.capability.render, tools.code.reader |
| tools.capability.reader | ingestion.front_matter, tools.code.reader |
| tools.capability.render | tools.capability.reader |
| tools.cli.__main__ | tools.cli.check, tools.cli.reader, tools.cli.render |
| tools.cli.check | tools.cli.reader, tools.cli.render |
| tools.cli.reader |  |
| tools.cli.render | tools.cli.reader |
| tools.code.__main__ | tools.code.check, tools.code.reader, tools.code.render |
| tools.code.check | tools.code.reader, tools.code.render |
| tools.code.reader |  |
| tools.code.render | tools.code.reader |
| tools.corpus.__main__ | tools.corpus.check, tools.corpus.reader |
| tools.corpus.check | ingestion.front_matter, tools.corpus.model, tools.corpus.reader |
| tools.corpus.model |  |
| tools.corpus.reader | ingestion.front_matter, tools.corpus.model |
| tools.glossary.__main__ | tools.glossary.check, tools.glossary.model, tools.glossary.render, tools.glossary.scaffold |
| tools.glossary.check | tools.glossary.model, tools.glossary.reader, tools.glossary.render |
| tools.glossary.model | ingestion.front_matter |
| tools.glossary.reader |  |
| tools.glossary.render | tools.glossary.model |
| tools.glossary.scaffold | tools.glossary.reader |
| tools.graph.__main__ | tools.graph.check, tools.graph.reader, tools.graph.render, tools.graph.traverse |
| tools.graph.check | tools.graph.reader, tools.graph.registry, tools.graph.render, tools.graph.traverse |
| tools.graph.reader | tools.adr.index, tools.capability.reader, tools.code.reader, tools.glossary.model, tools.graph.registry, tools.graphq.reader, tools.prompts.reader, tools.testmap.reader, tools.usecase.reader |
| tools.graph.registry |  |
| tools.graph.render | tools.graph.reader, tools.graph.registry |
| tools.graph.traverse | tools.adr.index, tools.capability.reader, tools.code.reader, tools.glossary.model, tools.graph.reader, tools.graph.registry, tools.graphq.reader, tools.prompts.reader, tools.testmap.reader, tools.usecase.reader |
| tools.graphq.__main__ | tools.graphq.check, tools.graphq.reader, tools.graphq.render |
| tools.graphq.check | tools.glossary.reader, tools.graphq.reader, tools.graphq.render |
| tools.graphq.reader |  |
| tools.graphq.render | tools.graphq.reader |
| tools.knowledge.__main__ | tools.knowledge.check, tools.knowledge.surfaces |
| tools.knowledge.check | tools.capability.reader, tools.usecase.reader |
| tools.knowledge.surfaces | tools.knowledge.check |
| tools.prompts.__main__ | tools.prompts.check, tools.prompts.reader, tools.prompts.render |
| tools.prompts.check | tools.glossary.model, tools.prompts.reader, tools.prompts.render |
| tools.prompts.reader |  |
| tools.prompts.render | tools.prompts.reader |
| tools.testmap.__main__ | tools.capability.reader, tools.testmap.check, tools.testmap.reader, tools.testmap.render, tools.testmap.verification, tools.usecase.reader |
| tools.testmap.check | tools.capability.reader, tools.testmap.reader, tools.testmap.render, tools.testmap.verification, tools.usecase.reader |
| tools.testmap.reader | tools.capability.reader |
| tools.testmap.render | tools.testmap.reader |
| tools.testmap.verification | tools.capability.reader, tools.testmap.reader, tools.usecase.reader |
| tools.usecase.__main__ | tools.capability.reader, tools.usecase.check, tools.usecase.coverage, tools.usecase.reader, tools.usecase.render |
| tools.usecase.check | tools.capability.reader, tools.usecase.coverage, tools.usecase.reader, tools.usecase.render |
| tools.usecase.coverage | tools.capability.reader, tools.usecase.reader |
| tools.usecase.reader | ingestion.front_matter |
| tools.usecase.render | tools.capability.reader, tools.usecase.coverage, tools.usecase.reader |
| ui.notifications | events.store |
| ui.reader |  |
| utils.environment |  |
| utils.helpers |  |
| utils.logger | config |
| utils.metrics |  |
| utils.neo4j_driver | config, utils.environment, utils.logger |
| utils.path_helpers |  |
| utils.text_processing | utils.logger |
| utils.visualize |  |
