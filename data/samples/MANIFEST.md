# Sample Corpus Manifest

Authored transcripts for exercising the pipeline against realistic,
coherent content rather than synthetic filler. Grows across milestones —
this manifest covers the corpus files that exist today and is extended
(not replaced) as new categories are added.

## Capability → content-property map

| Capability | Content property that exercises it |
| --- | --- |
| Speaker genesis / stitching | Continuous, unlabeled text (a category-2/3 "flat" sample; the category-1 files here are pre-labeled and skip inference) |
| Front-matter participants | YAML front matter block (`participants: [...]`) at the top of the file |
| Entity extraction + resolution | The same organization/product mentioned across multiple files with surface variants — "Acme Corp" vs. bare "Acme"; "Ledgerline" repeated verbatim |
| Person linking | The same human (`Maya Chen`) appearing in the front-matter `participants` list of ≥2 files, in different roles (interviewer vs. meeting participant) |
| Segments | Distinct topic phases within one conversation (e.g. intro → current workflow → pain points → wishlist) |
| Meeting_minutes lens | Decisions, action items, and objectives voiced explicitly in dialogue ("we've decided to...", "action item...", "our two objectives...") |
| Persona lens | A speaker expressing goals, frustrations, and quotable lines about their experience |
| Ask retrieval | Facts stated once, in one file, findable by a targeted question (e.g. Jordan's electric-bill detail, the specific action-item due dates) |

## File registry

| File | Category | Scenario type | Participants | Ground truth (where labels lie) | Size | Designed to stress |
| --- | --- | --- | --- | --- | --- | --- |
| `user_interview_mature.txt` | 1 (mature, clean) | UX research user interview | Interviewer: Maya Chen. Participant: Jordan Alvarez. | Labels are accurate; no adversarial mislabeling (category-1 files are clean by definition). | 64 speaker-prefixed lines, front matter with 2 participants | Persona lens (3 pain points explicitly enumerated by the speaker, 3+ goals, 2 flagged quotable lines); segments (5 topic phases: rapport/intro, current workflow, pain points, wishlist, closing); entity resolution ("Acme Corp" + bare "Acme", "Ledgerline"); person linking (Maya Chen also appears in `team_meeting_mature.txt`) |
| `team_meeting_mature.txt` | 1 (mature, clean) | Product team standup / planning meeting | Maya Chen, Priya Nandan, Ravi Okafor, Sam Lindqvist. | Labels are accurate; no adversarial mislabeling. | 47 speaker-prefixed lines, front matter with 4 participants | Meeting_minutes lens (2 explicit decisions, 4 action items with owners + due dates, 2 explicit objectives plus one "emerging" one); cross-file entity bait ("Ledgerline", "Acme Corp" + bare "Acme"); cross-file person linking (Maya Chen recurs from the user interview, in a different role) |

Category 2 (adversarial/mislabeled) and category 3 (edge-case/malformed)
samples are introduced in a later task and will extend this table in
place, not replace it.
