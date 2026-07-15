"""Bundle orchestration: guard -> read -> render (fully in memory) -> write."""

import asyncio
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from src.events.repository import get_interview_repository
from src.export import reader
from src.export.renderer import render_bundle
from src.lens.models import load_lens
from src.utils.logger import get_logger
from src.utils.neo4j_driver import Neo4jConnectionManager

logger = get_logger()


class ProjectionLagError(RuntimeError):
    """The aggregate expects lens items the graph projection hasn't caught up to yet."""


class InterviewNotFoundError(ValueError):
    """No interview aggregate exists for the given id."""


class LensNeverAppliedError(ValueError):
    """The lens is valid but has never been applied to this interview."""


class ExportResult(BaseModel):
    interview_id: str
    lens: str
    lens_version: int
    bundle_path: str
    files_written: int
    items: int
    claims: int
    entities: int


class OkfExporter:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        from src.config import config as global_config

        self.config = config_dict if config_dict is not None else global_config

    async def export(
        self, interview_id: str, lens_name: str,
        out_dir: str = "exports", zip_bundle: bool = False,
    ) -> ExportResult:
        lens = load_lens(lens_name)
        interview = await get_interview_repository().load(interview_id)
        if interview is None:
            raise InterviewNotFoundError(f"Interview {interview_id} not found")

        async with await Neo4jConnectionManager.get_session() as session:
            items = await reader.lens_item_rows(session, interview_id, lens.name)
            self._guard(interview, lens.name, items)
            transcript = await reader.transcript_rows(session, interview_id)
            speakers = await reader.speaker_rows(session, interview_id)
            claims = await reader.claim_rows(session, interview_id)
            entities = await reader.entity_rows(session, interview_id)
            analysis = await reader.analysis_rows(session, interview_id)
            persons = await reader.person_rows(session, interview_id)

        header = self._header(interview, lens)
        exported_at = datetime.now(timezone.utc).isoformat()
        files = render_bundle(
            header, transcript, speakers, items, claims, entities, analysis, lens, exported_at,
            persons=persons,
        )

        bundle_dir = Path(out_dir) / f"{interview_id}-{lens.name}"
        log_content = self._log_entry(bundle_dir, lens, len(items), exported_at)
        bundle_path = await asyncio.to_thread(
            self._write_bundle, bundle_dir, files, log_content, zip_bundle
        )

        return ExportResult(
            interview_id=interview_id, lens=lens.name, lens_version=lens.version,
            bundle_path=bundle_path, files_written=len(files) + 1,
            items=len(items), claims=len(claims), entities=len(entities),
        )

    def _write_bundle(
        self,
        bundle_dir: Path,
        files: List[Tuple[str, str]],
        log_content: str,
        zip_bundle: bool,
    ) -> str:
        """Write all bundle files into a staging dir, then atomically swap it in.

        A failure mid-write leaves the OLD bundle intact: the write phase happens
        entirely in a sibling `.staging` directory, and only the rename at the end
        can destroy the previous bundle. Zipping happens after the swap.
        """
        staging = bundle_dir.with_name(bundle_dir.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)

        for rel_path, content in files:
            target = staging / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (staging / "log.md").write_text(log_content, encoding="utf-8")

        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        staging.rename(bundle_dir)

        bundle_path = str(bundle_dir)
        if zip_bundle:
            zip_path = bundle_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(bundle_dir.rglob("*")):
                    if f.is_file():
                        zf.write(f, f.relative_to(bundle_dir.parent))
            bundle_path = str(zip_path)

        return bundle_path

    def _guard(self, interview, lens_name: str, projected_rows) -> None:
        """Expected = current-version items + locked items of any version."""
        current = interview.lens_runs.get(lens_name)
        if current is None:
            raise LensNeverAppliedError(f"lens {lens_name!r} never applied to this interview")
        expected = {
            iid for iid, v in interview.lens_items.items()
            if v["lens"] == lens_name and (v["lens_version"] == current or v["locked"])
        }
        projected = {r["item_id"] for r in projected_rows}
        if expected != projected:
            raise ProjectionLagError(
                f"projection lag: aggregate expects {len(expected)} items for lens "
                f"{lens_name!r}, graph has {len(projected)}; retry shortly"
            )

    def _header(self, interview, lens) -> Dict[str, Any]:
        metadata = interview.metadata or {}
        participants = [
            {"speaker_id": sid, **sp}
            for sid, sp in interview.speakers.items()
            if sp.get("merged_into") is None
        ]
        utterance_count = sum(
            1 for u in interview.utterances.values() if not u.get("removed")
        )
        started_at = interview.started_at
        return {
            "interview_id": interview.aggregate_id,
            "title": interview.title,
            "source": interview.source,
            "started_at": started_at.isoformat() if started_at else None,
            "project_id": metadata.get("front_matter", {}).get("project"),
            "metadata": metadata,
            "participants": participants,
            "fragment_count": metadata.get("fragment_count"),
            "utterance_count": utterance_count,
            "lens": lens.name,
            "lens_version": lens.version,
        }

    def _log_entry(self, bundle_dir: Path, lens, item_count: int, exported_at: str) -> str:
        """Add a new dated entry to the existing log.md, newest-first.

        Entries are a flat list grouped by ISO date. If the log already starts
        with a heading for today's date, the new bullet joins that group
        (newest bullet first); otherwise a new date block is prepended.
        """
        log_path = bundle_dir / "log.md"
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        date = exported_at.split("T")[0]
        heading = f"## {date}\n"
        bullet = f"- {exported_at}: exported {item_count} items from {lens.name} v{lens.version}\n"

        if existing.startswith(heading):
            rest = existing[len(heading):].lstrip("\n")
            return f"{heading}\n{bullet}{rest}"

        entry = f"{heading}\n{bullet}"
        if existing:
            return entry + "\n" + existing
        return entry
