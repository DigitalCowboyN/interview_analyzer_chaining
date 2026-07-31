# Prompt registry (probabilistic components)

## ask_prompts.yaml

| key | classification | used_for | audience | consumers | values |
| --- | --- | --- | --- | --- | --- |
| ask_synthesis | probabilistic | synthesis | ask | ask |  |

## core_extractors.yaml

| key | classification | used_for | audience | consumers | values |
| --- | --- | --- | --- | --- | --- |
| claims | probabilistic | extraction | enrichment | enrichment | assertion, commitment, request |
| domain_keywords | probabilistic | extraction | enrichment | enrichment |  |
| entity_mentions | probabilistic | extraction | enrichment | enrichment | person, organization, product, tool, other |
| function_type | probabilistic | classification | enrichment | enrichment | declarative, interrogative, imperative, exclamatory |
| overall_keywords | probabilistic | extraction | enrichment | enrichment |  |
| purpose | probabilistic | classification | enrichment | enrichment | Statement, Query, Exclamation, Answer, Commentary, Observation, Retraction, Mockery, Objection, Clarification, Conclusion, Confession, Speculation, Recitation, Correction, Explanation, Qualification, Threat, Warning, Advisory, Request, Addendum, Musing, Amendment |
| structure_type | probabilistic | classification | enrichment | enrichment | simple, compound, complex, compound-complex |
| topic_level_1 | probabilistic | classification | enrichment | enrichment | goals, tools, processes, experiences, observations, pain points, responsibilities, collaborations, reporting, managing, mentoring, strategy, operations, small talk, niceties |
| topic_level_3 | probabilistic | classification | enrichment | enrichment | goals, tools, processes, experiences, observations, pain points, responsibilities, collaborations, reporting, managing, mentoring, strategy, operations, small talk, niceties |
| topic_segments | probabilistic | segmentation | enrichment | enrichment |  |

## ingestion_prompts.yaml

| key | classification | used_for | audience | consumers | values |
| --- | --- | --- | --- | --- | --- |
| speaker_window | probabilistic | ingestion | ingestion | ingestion |  |
| stitch_window | probabilistic | ingestion | ingestion | ingestion |  |

## lens_meeting_minutes.yaml

| key | classification | used_for | audience | consumers | values |
| --- | --- | --- | --- | --- | --- |
| action_items | probabilistic | lens | lens | lens |  |
| decisions | probabilistic | lens | lens | lens |  |
| followups | probabilistic | lens | lens | lens |  |
| objectives | probabilistic | lens | lens | lens |  |

## lens_persona.yaml

| key | classification | used_for | audience | consumers | values |
| --- | --- | --- | --- | --- | --- |
| goals | probabilistic | lens | lens | lens |  |
| notable_quotes | probabilistic | lens | lens | lens |  |
| pain_points | probabilistic | lens | lens | lens |  |
| traits | probabilistic | lens | lens | lens |  |

## task_prompts.yaml

| key | classification | used_for | audience | consumers | values |
| --- | --- | --- | --- | --- | --- |
| domain_specific_keywords | probabilistic |  |  |  |  |
| sentence_function_type | probabilistic |  |  |  | declarative, interrogative, imperative, exclamatory |
| sentence_purpose | probabilistic |  |  |  | Statement, Query, Exclamation, Answer, Commentary, Observation, Retraction, Mockery, Objection, Clarification, Conclusion, Confession, Speculation, Recitation, Correction, Explanation, Qualification, Threat, Warning, Advisory, Request, Addendum, Musing, Amendment |
| sentence_structure_type | probabilistic |  |  |  | simple, compound, complex, compound-complex |
| topic_level_1 | probabilistic |  |  |  | goals, tools, processes, experiences, observations, pain points, responsibilities, collaborations, reporting, managing, mentoring, strategy, operations, small talk, niceties |
| topic_level_3 | probabilistic |  |  |  | goals, tools, processes, experiences, observations, pain points, responsibilities, collaborations, reporting, managing, mentoring, strategy, operations, small talk, niceties |
| topic_overall_keywords | probabilistic |  |  |  |  |
