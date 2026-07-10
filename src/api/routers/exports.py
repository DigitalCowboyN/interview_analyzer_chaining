"""OKF bundle download (Layer 5 egress)."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.export.bundler import InterviewNotFoundError, OkfExporter, ProjectionLagError
from src.utils.logger import get_logger

router = APIRouter(tags=["exports"])
logger = get_logger()


@router.get("/exports/{interview_id}/{lens_name}")
async def download_bundle(interview_id: str, lens_name: str):
    """Export the interview x lens as an OKF bundle and return it zipped."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = await OkfExporter().export(
                interview_id, lens_name, out_dir=tmp, zip_bundle=True
            )
        except InterviewNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ProjectionLagError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        payload = Path(result.bundle_path).read_bytes()
    filename = f"{interview_id}-{lens_name}.zip"
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
