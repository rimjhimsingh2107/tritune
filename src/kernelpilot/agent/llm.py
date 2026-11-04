"""LLM-backed planner. Optional: nothing else in KernelPilot needs an API key.

Providers keep the conversation in their own native message format and hand
back normalized (text, tool calls) turns. Configure with:

    KP_LLM_PROVIDER=anthropic   (needs `pip install kernelpilot[anthropic]` + ANTHROPIC_API_KEY)
    KP_LLM_PROVIDER=openai      (needs `pip install kernelpilot[openai]` + OPENAI_API_KEY, KP_LLM_MODEL)
    KP_LLM_MODEL=...            override the model id
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from kernelpilot.agent.planner import ToolCall, Turn
from kernelpilot.agent.tools import ToolResult

SYSTEM_PROMPT = """\
You are a GPU performance engineer optimizing a Triton kernel through a set of tools.

How to work:
- Ground every change in something you observed: compiler metadata (registers,
  spills, shared memory), profiler counters, PTX, validation errors or timings.
- One hypothesis per candidate. Say what you expect to happen and why in the
  generate_candidate `hypothesis` field, and cite the evidence in `rationale`.
- Order matters: compile, then validate_correctness, then benchmark_candidate.
  Never treat an unvalidated candidate as an improvement.
- If a candidate fails to compile or produces wrong numbers, deal with that
  before trying further optimizations (repair it or abandon that direction).
- benchmark_candidate tells you whether a candidate beat the incumbent beyond
  noise. Small wins inside the noise threshold don't count.
- recommend_optimization is a rule-based advisor. Use it as a second opinion,
  not an oracle.
- Don't resubmit configs you've already tried; generate_candidate reports
  duplicates.
- Stay within the candidate budget. Finish with save_experiment and a short,
  specific summary of what worked and what didn't.

Keep text between tool calls brief: what you saw, what you'll try next.
"""


@dataclass
class Reply:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""


class Provider(Protocol):
    name: str

    def reset(self, system: str, tools: list[dict[str, Any]], first_message: str) -> None: ...

    def step(self, results: list[ToolResult]) -> Reply: ...


def _result_text(r: ToolResult) -> str:
    body = r.content if r.ok else {"error": r.error}
    text = json.dumps(body, default=str)
    return text if len(text) < 12000 else text[:12000] + "...(truncated)"


class AnthropicProvider:
    """Claude via the official SDK, manual tool-use loop."""

    def __init__(self, model: str | None = None, max_tokens: int = 16000, fallbacks: bool | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("pip install 'kernelpilot[anthropic]' to use the Anthropic provider") from exc
        self.client = anthropic.Anthropic()
        self.model = model or "claude-opus-5-5"
        self.max_tokens = max_tokens
        if fallbacks is None:
            fallbacks = os.environ.get("KP_ANTHROPIC_FALLBACKS", "1") not in ("0", "false")
        self.fallbacks = fallbacks
        self.name = f"anthropic:{self.model}"
        self.system = ""
        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def reset(self, system: str, tools: list[dict[str, Any]], first_message: str) -> None:
        self.system = system
        self.tools = tools
        self.messages = [{"role": "user", "content": first_message}]

    def _create(self):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
        )
        if self.fallbacks:
            # server-side fallback if the model declines a request (refusal stop reason)
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        return self.client.messages.create(**kwargs)

    def step(self, results: list[ToolResult]) -> Reply:
        if results:
            self.messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": _result_text(r),
                     **({"is_error": True} if not r.ok else {})}
                    for r in results
                ],
            })
        resp = self._create()
        # keep the full content (incl. thinking blocks) so the next turn is valid
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")
        calls = [ToolCall(b.name, dict(b.input), b.id) for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "refusal":
            return Reply(text or "model declined to continue", [], "refusal")
        return Reply(text, calls, resp.stop_reason or "")


class OpenAIProvider:
    """Any OpenAI-compatible chat completions endpoint with function calling."""

    def __init__(self, model: str | None = None):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("pip install 'kernelpilot[openai]' to use the OpenAI provider") from exc
        self.model = model or os.environ.get("KP_LLM_MODEL") or ""
        if not self.model:
            raise RuntimeError("set KP_LLM_MODEL for the openai provider")
        self.client = openai.OpenAI()
        self.name = f"openai:{self.model}"
        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def reset(self, system: str, tools: list[dict[str, Any]], first_message: str) -> None:
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in tools]
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": first_message}]

    def step(self, results: list[ToolResult]) -> Reply:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": _result_text(r)})
        resp = self.client.chat.completions.create(model=self.model, messages=self.messages, tools=self.tools)
        msg = resp.choices[0].message
        self.messages.append(msg.model_dump(exclude_none=True))
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.function.arguments}
            calls.append(ToolCall(tc.function.name, args, tc.id))
        return Reply(msg.content or "", calls, resp.choices[0].finish_reason or "")


class LLMPlanner:
    def __init__(self, provider: Provider):
        self.provider = provider
        self.name = provider.name

    def start(self, briefing: str, tools: list[dict[str, Any]], context: dict[str, Any]) -> None:
        self.provider.reset(SYSTEM_PROMPT, tools, briefing)

    def next_turn(self, results: list[ToolResult]) -> Turn:
        reply = self.provider.step(results)
        done = not reply.calls and reply.stop_reason in ("end_turn", "stop", "refusal")
        return Turn(reply.text, reply.calls, done)


def make_provider(name: str, model: str | None = None) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(model or None)
    if name == "openai":
        return OpenAIProvider(model or None)
    raise ValueError(f"unknown LLM provider {name!r} (use anthropic or openai)")


def llm_available(name: str) -> tuple[bool, str]:
    if name in ("", "none"):
        return False, "KP_LLM_PROVIDER not set"
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(name)
    if key is None:
        return False, f"unknown provider {name!r}"
    module = name
    try:
        __import__(module)
    except ImportError:
        return False, f"python package '{module}' not installed"
    if not os.environ.get(key) and name == "openai":
        return False, f"{key} not set"
    return True, "ok"
