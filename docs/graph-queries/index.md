# Graph-query registry

## src/api/routers/segments.py

| query | purpose | scope | audience | consumers | labels | rels | returns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| list_segments | ui | task | api | api | Interview |  | found |

## src/ask/reader.py

| query | purpose | scope | audience | consumers | labels | rels | returns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| context_rows | ask | domain-broad | ask, agents | ask | Entity, Fragment, Interview, Person, Segment, Speaker, Utterance | CONTAINS, HAS_SENTENCE, IDENTIFIED_AS, MENTIONS, PART_OF_UTTERANCE, SPOKEN_BY | fragment_id, text, sequence_order, interview_id, title, speaker, person, segment_topics, entities |
| ensure_fulltext_index | ask | task | ask | ask |  |  |  |
| fulltext_rows | ask | domain-broad | ask, agents | ask | Interview, Project | CONTAINS_INTERVIEW, HAS_SENTENCE | fragment_id |
| graph_anchor_rows | ask | domain-broad | ask, agents | ask | CanonicalEntity, Entity, Fragment, Interview, Person, Project, Speaker | ALIAS_OF, CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, MENTIONS, SPOKEN_BY | fragment_id, fragment_id |
| name_rows | ask | domain-broad | ask, agents | ask | CanonicalEntity, Entity, Person | ALIAS_OF | kind, id, name, kind, id, name, surfaces |
| project_exists | ask | task | api, ask, agents | api, ask | Project |  | found |
| vector_fragment_rows | ask | domain-broad | ask, agents | ask | Interview, Project | CONTAINS_INTERVIEW, HAS_SENTENCE | fragment_id |
| vector_utterance_rows | ask | domain-broad | ask, agents | ask | Fragment, Interview, Project | CONTAINS_INTERVIEW, HAS_SENTENCE, PART_OF_UTTERANCE | fragment_id |

## src/export/reader.py

| query | purpose | scope | audience | consumers | labels | rels | returns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_rows | export | domain-broad | export | export | Analysis, Fragment, FunctionType, Interview, Keyword, Purpose, Speaker, StructureType, Topic | HAS_ANALYSIS, HAS_FUNCTION, HAS_PURPOSE, HAS_SENTENCE, HAS_STRUCTURE, MENTIONS_OVERALL_KEYWORD, MENTIONS_TOPIC, SPOKEN_BY | sequence_order, text, speaker, function, structure, purpose, topics, keywords, confidence, flags |
| claim_rows | export | domain-broad | export | export | Claim, Fragment, Speaker | MADE_BY, SUPPORTED_BY | claim_id, text, kind, confidence, model, provider, speaker_id, speaker, supporting_fragment_ids |
| entity_rows | export | domain-broad | export | export | CanonicalEntity, Entity, Fragment, Interview, Project | ALIAS_OF, CONTAINS_INTERVIEW, HAS_SENTENCE, MENTIONS | surface, entity_type, canonical_id, canonical_name |
| lens_item_rows | export | domain-broad | api, export | api, export | Fragment, LensItem, Speaker | SUPPORTED_BY | item_id, node_type, lens_version, confidence, model, provider, locked, props, supporting_fragment_ids |
| person_rows | export | domain-broad | export | export | Interview, Person, Speaker | HAS_PARTICIPANT, IDENTIFIED_AS | speaker_id, person_id, display_name |
| segment_rows | export | domain-broad | api, export | api, export | Fragment, Segment | CONTAINS | segment_id, topic, confidence, start_index, end_index |
| speaker_rollup_rows | export | domain-broad | api | api | Claim, Interview, LensItem, Person, Project, Speaker | CONTAINS_INTERVIEW, IDENTIFIED_AS, MADE_BY | display_name, node_type, relationship, text, interview_id, item_id, person_id, person_name |
| speaker_rows | export | domain-broad | export, resolution | export, resolution | Interview, Speaker | HAS_PARTICIPANT | speaker_id, handle, display_name, provisional |
| transcript_rows | export | domain-broad | export | export | Fragment, Interview, Speaker, Utterance | HAS_SENTENCE, PART_OF_UTTERANCE, SPOKEN_BY | sentence_id, sequence_order, text, speaker_id, speaker, utterance_id |
| worklist_rows | export | domain-broad | api | api | Claim, Interview, LensItem, Project | CONTAINS_INTERVIEW | interview_id, item_id, node_type, lens, confidence, reason |

## src/resolution/reader.py

| query | purpose | scope | audience | consumers | labels | rels | returns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| entity_surface_rows | resolution | domain-broad | resolution | resolution | Entity, Fragment, Interview, Project | CONTAINS_INTERVIEW, HAS_SENTENCE, MENTIONS | surface, entity_type, mentions |
| speaker_rows | resolution | domain-broad | export, resolution | export, resolution | Interview, Project, Speaker | CONTAINS_INTERVIEW, HAS_PARTICIPANT | interview_id, speaker_id, display_name, handle, provisional |

## src/ui/reader.py

| query | purpose | scope | audience | consumers | labels | rels | returns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interview_exists | ui | task | ui |  | Interview |  | found |
| interview_header_row | ui | task | api | api | Interview |  | interview_id, title, metadata_json |
| interview_rows | ui | domain-broad | api | api | Fragment, Interview, Project | CONTAINS_INTERVIEW, HAS_SENTENCE | interview_id, title, created_at, fragment_count |
| person_card_rows | ui | domain-broad | api | api | Fragment, Interview, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SPOKEN_BY | person_id, display_name, speaker_count, interview_count |
| person_contributes_to_persona | ui | task | api | api | Fragment, Interview, LensItem, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SUPPORTED_BY | found |
| person_detail_rows | ui | domain-broad | api | api | Fragment, Interview, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SPOKEN_BY | interview_id, interview_title, speaker_id, speaker_display_name |
| person_display_name_row | ui | task | api | api | Fragment, Interview, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SPOKEN_BY | person_id, display_name |
| person_exists | ui | task | api | api | Fragment, Interview, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SPOKEN_BY | found |
| persona_card_rows | ui | domain-broad | api | api | Fragment, Interview, LensItem, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SUPPORTED_BY | person_id, display_name |
| persona_detail_rows | ui | domain-broad | api | api | Fragment, Interview, LensItem, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SUPPORTED_BY | item_id, node_type, text, confidence, interview_id, interview_title |
| persona_exists | ui | task | api | api | Fragment, Interview, LensItem, Person, Project, Speaker | CONTAINS_INTERVIEW, HAS_SENTENCE, IDENTIFIED_AS, SUPPORTED_BY | found |
| project_exists | ui | task | api, ask | api, ask | Project |  | found |
| project_rows | ui | domain-broad | api | api | Interview, Project | CONTAINS_INTERVIEW | project_id, interview_count |
| transcript_line_rows | ui | domain-broad | api | api | Entity, Fragment, Interview, LensItem, Person, Segment, Speaker, Utterance | CONTAINS, HAS_SENTENCE, IDENTIFIED_AS, MENTIONS, PART_OF_UTTERANCE, SPOKEN_BY, SUPPORTED_BY | fragment_id, sequence_order, text, edited, speaker_id, speaker_display_name, person_id, person_display_name, utterance_id, segment_id, segment_topic |
