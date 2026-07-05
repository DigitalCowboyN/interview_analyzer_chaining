"""Claude Code harness as an LLM provider.

Shells out to the local `claude` CLI in headless print mode. Behavior
replication of an API provider; the accepted caveat is that raw API lifecycle
(HTTP errors, token metering) is not exercised through this interface.
"""

import asyncio
import json
import re
from typing import Any, Dict, Optional

from src.config import config as global_config
from src.utils.logger import get_logger

from .base_agent import BaseLLMAgent

logger = get_logger()

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ClaudeCodeAgent(BaseLLMAgent):
    """LLM provider backed by the local Claude Code CLI (`claude -p`)."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = (config_dict if config_dict is not None else global_config).get(
            "claude_code", {}
        )
        self.command = cfg.get("command", "claude")
        self.model = cfg.get("model", "haiku")
        self.timeout = cfg.get("timeout_seconds", 120)

    async def call_model(
        self, function_prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        prompt = function_prompt
        if schema is not None:
            # No API-level enforcement through the CLI; the schema rides in the
            # prompt and Pydantic validation downstream remains the contract.
            prompt += "\n\nRespond ONLY with JSON matching this schema: " + json.dumps(schema)

        argv = [self.command, "-p", prompt, "--output-format", "json"]
        if self.model:
            argv += ["--model", self.model]

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s")

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed (exit {proc.returncode}): {stderr.decode()[:500]}"
            )

        try:
            envelope = json.loads(stdout.decode())
            result_text = envelope.get("result", "")
            cleaned = _FENCE_RE.sub("", result_text).strip()
            return json.loads(cleaned) if cleaned else {}
        except json.JSONDecodeError as e:
            logger.warning(f"ClaudeCodeAgent: unparseable response ({e}); returning {{}}")
            return {}

    def get_provider_name(self) -> str:
        return "claude_code"

    def get_model_name(self) -> str:
        return f"claude-code/{self.model}"
