# API surface

## src.api.routers.analysis

- `POST /analysis/` — Trigger Analysis Pipeline

## src.api.routers.ask

- `POST /ask/{project_id}` — ask_project

## src.api.routers.edits

- `POST /edits/sentences/{interview_id}/{sentence_index}/analysis/override` — Override Analysis Results
- `POST /edits/sentences/{interview_id}/{sentence_index}/edit` — Edit Sentence Text
- `GET /edits/sentences/{interview_id}/{sentence_index}/history` — Get Sentence Edit History

## src.api.routers.exports

- `GET /exports/{interview_id}/{lens_name}` — download_bundle

## src.api.routers.files

- `GET /files/` — List Available Analysis Files
- `GET /files/{filename}` — Get Analysis File Content
- `GET /files/{filename}/sentences/{sentence_id}` — Get Specific Sentence Analysis

## src.api.routers.lenses

- `POST /lenses/{interview_id}/items/{item_id}/override` — override_lens_item

## src.api.routers.queries

- `GET /interviews/{interview_id}/lenses/{lens}/items` — lens_items
- `GET /review/worklist` — review_worklist
- `GET /speakers/rollup` — speakers_rollup

## src.api.routers.resolution

- `POST /resolution/{project_id}/entities/merge` — merge_entities
- `POST /resolution/{project_id}/entities/{canonical_id}/aliases` — add_alias
- `POST /resolution/{project_id}/entities/{canonical_id}/split` — split_entity
- `POST /resolution/{project_id}/persons/{person_id}/link` — link_speaker
- `POST /resolution/{project_id}/persons/{person_id}/unlink` — unlink_speaker

## src.api.routers.segments

- `GET /interviews/{interview_id}/segments` — list_segments
- `DELETE /segments/{interview_id}/{segment_id}` — remove_segment

## src.api.routers.speakers

- `POST /speakers/{interview_id}/fragments/{index}/reattribute` — reattribute_fragment
- `POST /speakers/{interview_id}/merge` — merge_speakers
- `POST /speakers/{interview_id}/split` — split_speaker
- `POST /speakers/{interview_id}/{speaker_id}/rename` — rename_speaker
- `DELETE /stitches/{interview_id}/{utterance_id}` — remove_stitch

## src.api.routers.ui

- `GET /ui/interviews/{interview_id}/transcript` — get_transcript
- `GET /ui/personas/{project_id}/{person_id}` — get_persona
- `GET /ui/persons/{project_id}/{person_id}` — get_person
- `GET /ui/projects` — list_projects
- `GET /ui/projects/{project_id}/interviews` — list_interviews
- `GET /ui/projects/{project_id}/person-id` — derive_person_id
- `GET /ui/projects/{project_id}/personas` — list_personas
- `GET /ui/projects/{project_id}/persons` — list_persons
- `GET /ui/streams/events` — stream_events

## src.main

- `GET /` — read_root
