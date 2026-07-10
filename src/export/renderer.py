"""Pure OKF rendering: reader dicts in, (relative_path, content) pairs out.

No I/O, no Neo4j. Lens items render generically from node_type + properties +
the lens YAML's projects_to — zero per-lens code.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.lens.models import LensSpec

# Public: the queries router (Task 7) uses this same split.
RESERVED_PROPS = {
    "item_id", "lens", "lens_version", "node_type", "confidence", "model",
    "provider", "interview_id", "locked", "overridden_at", "override_note",
}

# OKF frontmatter keys _render_lens_item always sets itself. An extracted lens
# field sharing one of these names must not silently overwrite it; such keys
# get a field_ prefix instead (mirrors src/projections/handlers/lens_handlers.py).
_OKF_FRONTMATTER_KEYS = {
    "type", "title", "description", "item_id", "lens", "lens_version",
    "confidence", "model", "provider", "locked", "tags", "timestamp",
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class _SlugRegistry:
    """Bundle-wide unique slugs: collisions suffixed, empty slugs hashed."""

    def __init__(self) -> None:
        self._taken: set = set()

    def slug_for(self, value: str) -> str:
        base = slugify(value)
        if not base:
            base = "x-" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        candidate, n = base, 1
        while candidate in self._taken:
            n += 1
            candidate = f"{base}-{n}"
        self._taken.add(candidate)
        return candidate


def _kebab(node_type: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", node_type).lower()


def item_dir(node_type: str) -> str:
    return _kebab(node_type) + "s"


def item_filename(node_type: str, item_id: str) -> str:
    return f"{_kebab(node_type)}-{item_id[:8]}.md"


_ID_FIELDS = {"id", "item_id"}


def _coerce_scalar(key: str, value: Any) -> Any:
    """The id field's all-digit strings become ints so YAML doesn't quote them.

    Scoped to the id field only (`id` or `item_id`, whichever a given file's
    frontmatter uses): coercing every all-digit string would silently change
    the type of arbitrary extracted LLM props (e.g. a zip code or ref number).
    """
    if key in _ID_FIELDS and isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _frontmatter(fields: Dict[str, Any]) -> str:
    clean = {k: _coerce_scalar(k, v) for k, v in fields.items() if v is not None}
    dumped = yaml.safe_dump(clean, sort_keys=False, allow_unicode=True)
    if "---" in dumped:
        # Line-anchored detection of a bare "---" delimiter line has provable
        # bypasses: a scalar's folded/quoted rendering can put "---" flush
        # against an opening or closing quote on the same physical line (e.g.
        # text starting or ending with "---"), so no single-line regex catches
        # every case. A plain substring check over-triggers instead -- that's
        # safe, since the fallback quoted re-dump only makes the output
        # uglier (every scalar becomes one line, escaped), and forcing that
        # style whenever "---" appears anywhere guarantees no physical line
        # can ever equal the delimiter.
        dumped = yaml.safe_dump(
            clean, sort_keys=False, allow_unicode=True, default_style='"', width=float("inf")
        )
    return "---\n" + dumped + "---\n"


def _anchors(transcript: List[Dict[str, Any]]) -> Dict[str, str]:
    """sentence_id -> anchor. Utterances u-<n> by first appearance; loose fragments f-<seq>."""
    anchors: Dict[str, str] = {}
    utterance_anchor: Dict[str, str] = {}
    n = 0
    for row in transcript:
        uid = row.get("utterance_id")
        if uid:
            if uid not in utterance_anchor:
                n += 1
                utterance_anchor[uid] = f"u-{n}"
            anchors[row["sentence_id"]] = utterance_anchor[uid]
        else:
            anchors[row["sentence_id"]] = f"f-{row['sequence_order']}"
    return anchors


def _speaker_slug(
    display_name: Optional[str],
    handle: Optional[str] = None,
    *,
    registry: "_SlugRegistry",
    speaker_slugs: Dict[str, str],
    speaker_id: Optional[str] = None,
) -> str:
    """Slug for a speaker identity, resolved through the bundle-wide registry.

    Cached by speaker_id when the caller has one -- that's the only thing
    that reliably distinguishes two speakers, since their display_name/handle
    may be identical or both absent (e.g. two anonymous speakers both reduce
    to "unknown"). Falling back to the identity string only applies when no
    speaker_id is available at all. Every call site that resolves a speaker
    link (transcript participants, lens items, claims, grounding
    attributions, the speaker file itself) passes the speaker_id it knows so
    they all agree on the one slug that speaker's file actually got.
    """
    identity = display_name or handle or "unknown"
    cache_key = speaker_id if speaker_id is not None else identity
    slug = speaker_slugs.get(cache_key)
    if slug is None:
        slug = registry.slug_for(identity)
        speaker_slugs[cache_key] = slug
    return slug


def _link_text(value: str, limit: int = 80) -> str:
    """LLM/graph text used as a markdown link label: one line, brackets escaped.

    Truncates the raw text first, then escapes -- escaping first and
    truncating second can sever a `\\[`/`\\]` pair at the boundary, leaving a
    lone trailing backslash that escapes the link's closing `]` and breaks
    the markdown link. The escaped result may run slightly past `limit`;
    well-formedness beats exact length.
    """
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    return collapsed[:limit].replace("[", "\\[").replace("]", "\\]")


def _cell(value: Any) -> str:
    """LLM/graph text used as a markdown table cell: one line, pipes escaped."""
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    return collapsed.replace("|", "\\|")


def _render_interview(
    header: Dict[str, Any], registry: "_SlugRegistry", speaker_slugs: Dict[str, str]
) -> Tuple[str, str]:
    fm = {
        "type": "Interview",
        "title": header.get("title"),
        "id": header.get("interview_id"),
        "lens": header.get("lens"),
        "lens_version": header.get("lens_version"),
        "timestamp": header.get("started_at"),
        "front_matter": (header.get("metadata") or {}).get("front_matter"),
    }
    lines = [_frontmatter(fm), f"# {header.get('title') or header.get('interview_id')}", ""]
    lines.append(f"- Source: {header.get('source')}")
    lines.append(f"- Project: {header.get('project_id')}")
    lines.append(f"- Fragments: {header.get('fragment_count')}")
    lines.append(f"- Utterances: {header.get('utterance_count')}")
    lines.append("")
    lines.append("## Participants")
    for p in header.get("participants") or []:
        marker = " (provisional)" if p.get("provisional") else ""
        slug = _speaker_slug(
            p.get("display_name"), p.get("handle"),
            registry=registry, speaker_slugs=speaker_slugs, speaker_id=p.get("speaker_id"),
        )
        lines.append(f"- [{_link_text(p.get('display_name'))}](/speakers/{slug}.md){marker}")
    return "interview.md", "\n".join(lines) + "\n"


def _render_transcript(transcript: List[Dict[str, Any]], anchors: Dict[str, str]) -> Tuple[str, str]:
    fm = {"type": "Transcript"}
    lines = [_frontmatter(fm), "# Transcript", ""]
    last_anchor = None
    for row in transcript:
        anchor = anchors[row["sentence_id"]]
        if anchor != last_anchor:
            if last_anchor is not None:
                lines.append("")
            lines.append(f'<a id="{anchor}"></a>')
            lines.append(f"**Speaker:** {row.get('speaker')}")
            lines.append("")
            last_anchor = anchor
        lines.append(row.get("text", ""))
    return "transcript.md", "\n".join(lines) + "\n"


def _render_analysis(analysis: List[Dict[str, Any]]) -> Tuple[str, str]:
    fm = {"type": "AnalysisSummary"}
    topic_tally: Dict[str, int] = {}
    keyword_tally: Dict[str, int] = {}
    for row in analysis:
        for t in row.get("topics") or []:
            topic_tally[t] = topic_tally.get(t, 0) + 1
        for k in row.get("keywords") or []:
            keyword_tally[k] = keyword_tally.get(k, 0) + 1

    lines = [_frontmatter(fm), "# Analysis Summary", ""]
    lines.append("## Topics")
    for topic, count in sorted(topic_tally.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {topic}: {count}")
    lines.append("")
    lines.append("## Keywords")
    for keyword, count in sorted(keyword_tally.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {keyword}: {count}")
    lines.append("")
    lines.append("## Fragments")
    lines.append("| # | Speaker | Function | Structure | Purpose | Topics | Keywords | Confidence | Text |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in analysis:
        topics = ", ".join(row.get("topics") or [])
        keywords = ", ".join(row.get("keywords") or [])
        cells = [
            row.get("sequence_order"), row.get("speaker"), row.get("function"),
            row.get("structure"), row.get("purpose"), topics, keywords,
            row.get("confidence"), row.get("text"),
        ]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return "analysis.md", "\n".join(lines) + "\n"


def _render_speaker(
    speaker: Dict[str, Any],
    references: List[Tuple[str, str]],
    registry: "_SlugRegistry",
    speaker_slugs: Dict[str, str],
) -> Tuple[str, str]:
    slug = _speaker_slug(
        speaker.get("display_name"), speaker.get("handle"),
        registry=registry, speaker_slugs=speaker_slugs, speaker_id=speaker.get("speaker_id"),
    )
    fm = {
        "type": "Speaker",
        "title": speaker.get("display_name") or speaker.get("handle"),
        "id": speaker.get("speaker_id"),
        "provisional": speaker.get("provisional"),
    }
    lines = [_frontmatter(fm), f"# {speaker.get('display_name') or speaker.get('handle')}", ""]
    lines.append("## Referenced By")
    for rel_path, display in references:
        lines.append(f"- [{_link_text(display)}](/{rel_path})")
    return f"speakers/{slug}.md", "\n".join(lines) + "\n"


def _grounding_lines(
    supporting_fragment_ids: List[str],
    sentence_text: Dict[str, str],
    sentence_speaker: Dict[str, str],
    anchors: Dict[str, str],
    fallback_speaker: str,
    registry: "_SlugRegistry",
    speaker_slugs: Dict[str, str],
    sentence_speaker_id: Optional[Dict[str, str]] = None,
    fallback_speaker_id: Optional[str] = None,
) -> List[str]:
    lines = ["## Grounding"]
    for sid in supporting_fragment_ids or []:
        anchor = anchors.get(sid)
        if anchor is None:
            continue
        text = sentence_text.get(sid)
        speaker_display = sentence_speaker.get(sid) or fallback_speaker
        sp_id = (sentence_speaker_id or {}).get(sid) or fallback_speaker_id
        speaker_slug = _speaker_slug(speaker_display, registry=registry, speaker_slugs=speaker_slugs, speaker_id=sp_id)
        link = (
            f"[{_link_text(speaker_display)}](/speakers/{speaker_slug}.md), "
            f"[{anchor}](/transcript.md#{anchor})"
        )
        if text is not None:
            lines.append(f"> {text}")
            lines.append(f"> — {link}")
        else:
            lines.append(f"— {link}")
        lines.append("")
    return lines


def _render_lens_item(
    item: Dict[str, Any],
    lens: LensSpec,
    exported_at: str,
    sentence_text: Dict[str, str],
    sentence_speaker: Dict[str, str],
    anchors: Dict[str, str],
    registry: "_SlugRegistry",
    speaker_slugs: Dict[str, str],
    sentence_speaker_id: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    node_type = item["node_type"]
    item_id = item["item_id"]
    props = item.get("props") or {}
    extracted = {
        (f"field_{k}" if k in _OKF_FRONTMATTER_KEYS else k): v
        for k, v in props.items()
        if k not in RESERVED_PROPS
    }
    text = props.get("text")
    title = (_link_text(text) if text else None) or item_id

    fm = {
        "type": node_type,
        "title": title,
        "description": text,
        "item_id": item_id,
        "lens": item.get("lens", props.get("lens")),
        "lens_version": item.get("lens_version"),
        "confidence": item.get("confidence"),
        "model": item.get("model"),
        "provider": item.get("provider"),
        "locked": item.get("locked"),
        "tags": [f"lens:{item.get('lens', props.get('lens'))}"],
        "timestamp": exported_at,
        **extracted,
    }

    lines = [_frontmatter(fm)]
    if text:
        lines.append(text)
        lines.append("")

    mapping = lens.projects_to.get(node_type)
    speaker_link = mapping.speaker_link if mapping else None
    lines.append("## Relationships")
    for link in item.get("speaker_links") or []:
        rel = link.get("relationship") or (speaker_link.get("relationship") if speaker_link else None)
        display = link.get("display_name")
        if not rel or not display:
            continue
        sp_slug = _speaker_slug(
            display, registry=registry, speaker_slugs=speaker_slugs, speaker_id=link.get("speaker_id")
        )
        lines.append(f"{rel}: [{_link_text(display)}](/speakers/{sp_slug}.md)")
    lines.append("")

    speaker_links = item.get("speaker_links") or []
    fallback_speaker = speaker_links[0].get("display_name") if speaker_links else "Unknown"
    fallback_speaker_id = speaker_links[0].get("speaker_id") if speaker_links else None

    lines.extend(
        _grounding_lines(
            item.get("supporting_fragment_ids") or [],
            sentence_text,
            sentence_speaker,
            anchors,
            fallback_speaker,
            registry,
            speaker_slugs,
            sentence_speaker_id,
            fallback_speaker_id,
        )
    )

    path = f"{item_dir(node_type)}/{item_filename(node_type, item_id)}"
    return path, "\n".join(lines) + "\n"


def _render_claim(
    claim: Dict[str, Any],
    sentence_text: Dict[str, str],
    sentence_speaker: Dict[str, str],
    anchors: Dict[str, str],
    registry: "_SlugRegistry",
    speaker_slugs: Dict[str, str],
    sentence_speaker_id: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    claim_id = claim["claim_id"]
    text = claim.get("text")
    title = (_link_text(text) if text else None) or claim_id

    fm = {
        "type": "Claim",
        "title": title,
        "description": text,
        "item_id": claim_id,
        "kind": claim.get("kind"),
        "confidence": claim.get("confidence"),
        "model": claim.get("model"),
        "provider": claim.get("provider"),
        "timestamp": None,
    }

    lines = [_frontmatter(fm)]
    if text:
        lines.append(text)
        lines.append("")

    speaker_display = claim.get("speaker") or "Unknown"
    claim_speaker_id = claim.get("speaker_id")
    speaker_slug = _speaker_slug(
        speaker_display, registry=registry, speaker_slugs=speaker_slugs, speaker_id=claim_speaker_id
    )

    lines.append("## Relationships")
    if claim_speaker_id:
        lines.append(f"MADE_BY: [{_link_text(speaker_display)}](/speakers/{speaker_slug}.md)")
    lines.append("")

    lines.extend(
        _grounding_lines(
            claim.get("supporting_fragment_ids") or [],
            sentence_text,
            sentence_speaker,
            anchors,
            speaker_display,
            registry,
            speaker_slugs,
            sentence_speaker_id,
            claim_speaker_id,
        )
    )

    path = f"claims/claim-{claim_id[:8]}.md"
    return path, "\n".join(lines) + "\n"


def _render_entity(entity: Dict[str, Any], anchors: Dict[str, str], registry: "_SlugRegistry") -> Tuple[str, str]:
    surface = entity["surface"]
    slug = registry.slug_for(surface)
    fm = {
        "type": "Entity",
        "title": surface,
        "description": entity.get("entity_type"),
        "entity_type": entity.get("entity_type"),
    }
    lines = [_frontmatter(fm), f"# {surface}", ""]
    lines.append("## Mentions")
    lines.append("| Text | Confidence | Location |")
    lines.append("|---|---|---|")
    for mention in entity.get("mentions") or []:
        anchor = anchors.get(mention.get("sentence_id"))
        location = f"[{anchor}](/transcript.md#{anchor})" if anchor else ""
        lines.append(f"| {_cell(mention.get('text'))} | {mention.get('confidence')} | {location} |")
    return f"entities/{slug}.md", "\n".join(lines) + "\n"


_OVERVIEW_ENTRIES = [
    ("interview.md", "Interview header, participants, and counts."),
    ("transcript.md", "Full transcript, anchored by utterance/fragment."),
    ("analysis.md", "Aggregated topic/keyword tallies and per-fragment analysis."),
]


def _render_index(sections: Dict[str, List[Tuple[str, str]]]) -> Tuple[str, str]:
    lines = ["# Index", "", "## Overview"]
    for path, description in _OVERVIEW_ENTRIES:
        lines.append(f"- [{path}]({path}): {description}")
    lines.append("")
    for section, entries in sections.items():
        if not entries:
            continue
        lines.append(f"## {section.replace('-', ' ').title()}")
        for path, description in entries:
            lines.append(f"- [{path}]({path}): {_link_text(description)}")
        lines.append("")
    return "index.md", "\n".join(lines) + "\n"


def render_bundle(
    header: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    speakers: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    analysis: List[Dict[str, Any]],
    lens: LensSpec,
    exported_at: str,
) -> List[Tuple[str, str]]:
    anchors = _anchors(transcript)
    sentence_text = {row["sentence_id"]: row.get("text", "") for row in transcript}
    sentence_speaker = {row["sentence_id"]: row.get("speaker") for row in transcript}
    sentence_speaker_id = {row["sentence_id"]: row.get("speaker_id") for row in transcript}
    speaker_by_id = {sp["speaker_id"]: sp for sp in speakers}

    # One registry for the whole bundle: speakers and entities share a slug
    # namespace, so e.g. speaker "ECU" and entity "ecu" cannot collide. The
    # speaker_slugs cache resolves every /speakers/<slug>.md link (interview
    # participants, lens items, claims, grounding attributions, transcript-
    # adjacent references) to the same slug the speaker file itself gets.
    registry = _SlugRegistry()
    speaker_slugs: Dict[str, str] = {}

    files: List[Tuple[str, str]] = []
    index_sections: Dict[str, List[Tuple[str, str]]] = {"speakers": []}

    files.append(_render_interview(header, registry, speaker_slugs))
    files.append(_render_transcript(transcript, anchors))
    files.append(_render_analysis(analysis))

    # Single derivation of each item's (rel_path, title), reused by the item-file
    # loop, the index sections, and the speaker back-link references below.
    item_refs = [
        (
            item,
            f"{item_dir(item['node_type'])}/{item_filename(item['node_type'], item['item_id'])}",
            (item.get("props") or {}).get("text") or item["item_id"],
        )
        for item in items
    ]

    # Which items/claims reference each speaker, for the speaker file's back-links.
    references_by_speaker: Dict[str, List[Tuple[str, str]]] = {sp_id: [] for sp_id in speaker_by_id}
    for item, item_path, title in item_refs:
        for link in item.get("speaker_links") or []:
            sp_id = link.get("speaker_id")
            if sp_id in references_by_speaker:
                references_by_speaker[sp_id].append((item_path, title))
    for claim in claims:
        sp_id = claim.get("speaker_id")
        if sp_id in references_by_speaker:
            claim_path = f"claims/claim-{claim['claim_id'][:8]}.md"
            references_by_speaker[sp_id].append((claim_path, claim.get("text") or claim["claim_id"]))

    for speaker in speakers:
        path, content = _render_speaker(
            speaker, references_by_speaker.get(speaker["speaker_id"], []), registry, speaker_slugs
        )
        files.append((path, content))
        index_sections["speakers"].append((path, speaker.get("display_name") or speaker.get("handle")))

    for item, path, title in item_refs:
        _, content = _render_lens_item(
            item, lens, exported_at, sentence_text, sentence_speaker, anchors, registry, speaker_slugs,
            sentence_speaker_id,
        )
        files.append((path, content))
        index_sections.setdefault(item_dir(item["node_type"]), []).append((path, title))

    for claim in claims:
        path, content = _render_claim(
            claim, sentence_text, sentence_speaker, anchors, registry, speaker_slugs, sentence_speaker_id
        )
        files.append((path, content))
        index_sections.setdefault("claims", []).append((path, claim.get("text") or claim["claim_id"]))

    for entity in entities:
        path, content = _render_entity(entity, anchors, registry)
        files.append((path, content))
        index_sections.setdefault("entities", []).append((path, entity.get("entity_type") or entity["surface"]))

    files.append(_render_index(index_sections))
    return files
