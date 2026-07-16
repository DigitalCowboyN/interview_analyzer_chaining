"""
src/api/routers/ask.py

Ask-the-corpus endpoint (M4.6). Read-side only: retrieval + one synthesis
call; nothing persisted.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.ask.engine import AskEngine, SynthesisUnavailable

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(12, ge=1, le=50)


@router.post("/ask/{project_id}")
async def ask_project(project_id: str, body: AskRequest):
    """Answer a question from one project's graph, with verbatim citations."""
    try:
        result = await AskEngine().ask(project_id, body.question, top_k=body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SynthesisUnavailable as e:
        return JSONResponse(status_code=502, content=e.result.model_dump())
    return result.model_dump()
