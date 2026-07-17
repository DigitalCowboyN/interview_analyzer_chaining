from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.aggregates import Interview
from src.export.bundler import OkfExporter

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
ITEM = "88888888-8888-8888-8888-888888888801"


def make_interview(with_item=True):
    i = Interview(IID)
    i.create(title="t", source="s", metadata={"fragment_count": 1})
    i.add_speaker(SP1, "S1", "Alice", False, 1.0, "parsed")
    i.apply_lens("meeting_minutes", 1)
    if with_item:
        i.record_lens_extraction(
            lens="meeting_minutes", lens_version=1, node_type="Decision", item_id=ITEM,
            fields={"text": "Go with X"}, supporting_fragment_ids=[], speaker_links=[],
            confidence=0.9, model="haiku", provider="anthropic",
        )
    i.mark_events_as_committed()
    return i


def patch_world(interview, projected_items):
    repo = MagicMock(load=AsyncMock(return_value=interview))
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    reader_rows = {
        "transcript_rows": [], "speaker_rows": [], "claim_rows": [],
        "entity_rows": [], "analysis_rows": [], "person_rows": [],
    }
    patches = [
        patch("src.export.bundler.get_interview_repository", return_value=repo),
        patch("src.export.bundler.Neo4jConnectionManager.get_session",
              new=AsyncMock(return_value=session)),
        patch("src.export.bundler.reader.lens_item_rows",
              new=AsyncMock(return_value=projected_items)),
    ]
    for name, rows in reader_rows.items():
        patches.append(patch(f"src.export.bundler.reader.{name}", new=AsyncMock(return_value=rows)))
    return patches


PROJECTED = [{
    "item_id": ITEM, "node_type": "Decision", "lens_version": 1, "confidence": 0.9,
    "model": "haiku", "provider": "anthropic", "locked": False,
    "props": {"item_id": ITEM, "text": "Go with X"},
    "speaker_links": [], "supporting_fragment_ids": [],
}]


@pytest.mark.asyncio
async def test_export_writes_conformant_bundle(tmp_path):
    import yaml
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        result = await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
    bundle = tmp_path / f"{IID}-meeting_minutes"
    assert (bundle / "index.md").exists() and (bundle / "interview.md").exists()
    assert (bundle / "log.md").exists()
    assert result.items == 1 and result.files_written >= 4
    content = (bundle / "decisions" / f"decision-{ITEM[:8]}.md").read_text()
    assert yaml.safe_load(content.split("---\n")[1])["type"] == "Decision"


@pytest.mark.asyncio
async def test_projection_lag_raises(tmp_path):
    patches = patch_world(make_interview(with_item=True), projected_items=[])  # graph empty
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        with pytest.raises(RuntimeError, match="projection lag"):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_unknown_interview_raises(tmp_path):
    repo = MagicMock(load=AsyncMock(return_value=None))
    with patch("src.export.bundler.get_interview_repository", return_value=repo):
        with pytest.raises(ValueError, match="not found"):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_reexport_prepends_log_entry(tmp_path):
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
    log = (tmp_path / f"{IID}-meeting_minutes" / "log.md").read_text()
    assert log.count("exported") == 2


@pytest.mark.asyncio
async def test_never_applied_lens_raises_422_error(tmp_path):
    from src.export.bundler import LensNeverAppliedError

    interview = Interview(IID)
    interview.create(title="t", source="s", metadata={"fragment_count": 1})
    interview.mark_events_as_committed()
    patches = patch_world(interview, projected_items=[])
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        with pytest.raises(LensNeverAppliedError, match="never applied"):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_reexport_same_day_groups_under_one_heading(tmp_path):
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
    log = (tmp_path / f"{IID}-meeting_minutes" / "log.md").read_text()
    date = datetime.now(timezone.utc).date().isoformat()

    assert log.count(f"## {date}") == 1

    bullets = [line for line in log.splitlines() if line.startswith("- ")]
    assert len(bullets) == 2
    first_ts = bullets[0][2:].split(": exported")[0]
    second_ts = bullets[1][2:].split(": exported")[0]
    assert first_ts > second_ts


@pytest.mark.asyncio
async def test_failed_write_preserves_previous_bundle(tmp_path, monkeypatch):
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        bundle = tmp_path / f"{IID}-meeting_minutes"
        original_index = (bundle / "index.md").read_text()

        real_write = Path.write_text
        calls = {"n": 0}

        def flaky_write(self, *a, **k):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise OSError("disk full")
            return real_write(self, *a, **k)

        monkeypatch.setattr(Path, "write_text", flaky_write)
        with pytest.raises(OSError):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        monkeypatch.setattr(Path, "write_text", real_write)

    assert (bundle / "index.md").read_text() == original_index  # old bundle intact


def _staging_dirs(tmp_path):
    return [p for p in tmp_path.iterdir() if ".staging-" in p.name]


@pytest.mark.asyncio
async def test_consecutive_writes_leave_no_staging_residue(tmp_path):
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
    assert _staging_dirs(tmp_path) == []


@pytest.mark.asyncio
async def test_staging_dir_name_is_unique_per_write(tmp_path, monkeypatch):
    """Two writes must not stage under the same fixed sibling path -- a fixed
    name lets concurrent exports collide (one export's in-progress staging
    dir gets rmtree'd by the other). Capture the staging path each write
    actually uses and assert they differ."""
    patches = patch_world(make_interview(), PROJECTED)
    seen_staging_paths = []
    real_rename = Path.rename

    def spy_rename(self, target):
        seen_staging_paths.append(self)
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", spy_rename)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))

    assert len(seen_staging_paths) == 2
    assert seen_staging_paths[0].name != seen_staging_paths[1].name


@pytest.mark.asyncio
async def test_failed_write_removes_its_own_staging_dir(tmp_path, monkeypatch):
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))

        real_write = Path.write_text
        calls = {"n": 0}

        def flaky_write(self, *a, **k):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise OSError("disk full")
            return real_write(self, *a, **k)

        monkeypatch.setattr(Path, "write_text", flaky_write)
        with pytest.raises(OSError):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        monkeypatch.setattr(Path, "write_text", real_write)

    assert _staging_dirs(tmp_path) == []
