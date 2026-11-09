"""
Claude, through Anthropic's own Python SDK.

The pipeline engine keeps its conversation in the OpenAI chat shape (system /
user / assistant-with-tool_calls / tool). This provider translates that history
into the Messages API shape on every request and streams the answer back as the
same {"text"} / {"tool_calls"} events llm_openai yields, so nothing upstream
changes.

THINKING BLOCKS
    Current Claude models think on every turn and return signed `thinking`
    blocks that are bound to the conversation that produced them. The pipeline
    stores only text and tool calls, so each response's original content is
    kept here (keyed by its tool-call ids, or its text) and replayed verbatim.
    The pipeline also trims old turns once history grows, which is an "edit" as
    far as the API is concerned, so every request sets
    `thinking.block_binding.prefix_mismatch_behavior = "drop_block"`: a block
    that no longer matches is dropped by the API instead of failing the turn.

REFUSALS
    Opus 5.5 / Sonnet 5.5 / Fable 5.1 requests opt into server-side refusal
    fallbacks (`fallbacks: "default"`), so a safety decline is re-run on a
    fallback model inside the same call instead of ending the turn.
"""
from __future__ import annotations

import base64
import json
import re
from collections import OrderedDict
from importlib.util import find_spec
from typing import Any, AsyncIterator

from .base import DEFAULT_CAPS, Field, FunctionCall

MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
          "claude-fable-5-1")
DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Models that take adaptive thinking with a block_binding (all current ones).
_ADAPTIVE = set(MODELS)
# Server-side refusal fallback ("default" form) is offered on these.
_FALLBACK = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}

_BETA_BINDING = "thinking-binding-controls-2026-08-01"
_BETA_FALLBACK = "server-side-fallback-2026-07-01"

_STREAM_MAX_TOKENS = 64000
_COMPLETE_MAX_TOKENS = 16000
_CACHE_LIMIT = 256          # remembered assistant turns (for thinking replay)


def available() -> bool:
    return find_spec("anthropic") is not None


def _terse(text: str, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _image_block(url: str) -> dict | None:
    m = re.match(r"^data:(image/[\w.+-]+);base64,(.*)$", url or "", re.S)
    if m:
        return {"type": "image", "source": {"type": "base64",
                                            "media_type": m.group(1),
                                            "data": m.group(2)}}
    if (url or "").startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _user_content(content: Any) -> list:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    out = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text" and str(part.get("text") or "").strip():
            out.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            iu = part.get("image_url")
            img = _image_block(iu.get("url") if isinstance(iu, dict) else iu)
            if img:
                out.append(img)
    return out


def _text_key(text: Any) -> str:
    return " ".join(str(text or "").split())


class AnthropicLLM:
    """Claude via the Messages API (anthropic SDK)."""

    def __init__(self, cfg: dict):
        import anthropic  # inside the factory path — never at module scope

        self._anthropic = anthropic
        self.model = (cfg.get("model") or "").strip() or DEFAULT_MODEL
        self.key = (cfg.get("api_key") or "").strip() or None
        self.effort = (cfg.get("effort") or "low").strip().lower()
        if self.effort not in EFFORTS:
            self.effort = "low"
        self.timeout = float(cfg.get("timeout") or 120.0)
        self._client = anthropic.Anthropic(api_key=self.key, timeout=self.timeout)
        self._aclient = anthropic.AsyncAnthropic(api_key=self.key,
                                                 timeout=self.timeout)
        # Original response content per assistant turn, for verbatim replay.
        self._turns: "OrderedDict[Any, list]" = OrderedDict()

        self.caps = dict(DEFAULT_CAPS)
        self.caps.update({"tools": True, "tool_format": "anthropic",
                          "vision": True, "max_tools": None,
                          "max_tool_chars": None})

    # -- history translation --------------------------------------------------
    def _remember(self, content: list) -> None:
        ids = tuple(b.id for b in content if getattr(b, "type", "") == "tool_use")
        text = "".join(getattr(b, "text", "") for b in content
                       if getattr(b, "type", "") == "text")
        key = ("tools", ids) if ids else ("text", _text_key(text))
        self._turns[key] = list(content)
        self._turns.move_to_end(key)
        while len(self._turns) > _CACHE_LIMIT:
            self._turns.popitem(last=False)

    def _assistant_content(self, msg: dict) -> list:
        calls = msg.get("tool_calls") or []
        key = (("tools", tuple(c.get("id") for c in calls)) if calls
               else ("text", _text_key(msg.get("content"))))
        cached = self._turns.get(key)
        if cached is not None:
            return list(cached)
        out: list = []
        if str(msg.get("content") or "").strip():
            out.append({"type": "text", "text": msg["content"]})
        for c in calls:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {"value": args}
            out.append({"type": "tool_use", "id": c.get("id") or fn.get("name"),
                        "name": fn.get("name") or "", "input": args})
        return out

    def _translate(self, messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        out: list[dict] = []

        def push(role: str, blocks: list) -> None:
            if not blocks:
                return
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "system":
                text = m.get("content") if isinstance(m.get("content"), str) else ""
                if not out:
                    system_parts.append(text)
                elif text.strip():
                    push("user", [{"type": "text", "text": f"[Note] {text}"}])
            elif role == "user":
                push("user", _user_content(m.get("content")))
            elif role == "assistant":
                push("assistant", self._assistant_content(m))
            elif role == "tool":
                push("user", [{"type": "tool_result",
                               "tool_use_id": m.get("tool_call_id") or m.get("name"),
                               "content": str(m.get("content") or "")}])

        if out and out[0]["role"] != "user":
            out.insert(0, {"role": "user", "content": [
                {"type": "text", "text": "(continuing the conversation)"}]})
        return "\n\n".join(p for p in system_parts if p), out

    # -- request shape ----------------------------------------------------------
    def _kwargs(self, messages: list[dict], tools: list[dict] | None,
                max_tokens: int) -> dict:
        system, msgs = self._translate(messages)
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msgs,
            "cache_control": {"type": "ephemeral"},
            "output_config": {"effort": self.effort},
        }
        betas: list[str] = []
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = tools
        if self.model in _ADAPTIVE:
            kw["thinking"] = {"type": "adaptive",
                              "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
            betas.append(_BETA_BINDING)
        if self.model in _FALLBACK:
            kw["fallbacks"] = "default"
            betas.append(_BETA_FALLBACK)
        if betas:
            kw["betas"] = betas
        return kw

    @staticmethod
    def _calls(final) -> list[FunctionCall]:
        if final.stop_reason == "max_tokens":
            return []           # a truncated tool input must not run
        calls = []
        for b in final.content:
            if getattr(b, "type", "") == "tool_use":
                args = b.input if isinstance(b.input, dict) else {"value": b.input}
                calls.append(FunctionCall(id=b.id, name=b.name, args=args))
        return calls

    # -- streaming, async: the pipeline engine ----------------------------------
    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   *, stream: bool = True,
                   temperature: float | None = None) -> AsyncIterator[dict]:
        # temperature is ignored: current Claude models reject sampling params.
        # eager_input_streaming is deliberately not set on the tools: this
        # provider never reads partial tool input, and leaving it off keeps the
        # server validating every tool input against its schema.
        kw = self._kwargs(messages, tools, _STREAM_MAX_TOKENS)
        said = False
        async with self._aclient.beta.messages.stream(**kw) as s:
            async for event in s:
                if (event.type == "content_block_delta"
                        and getattr(event.delta, "type", "") == "text_delta"
                        and event.delta.text):
                    said = True
                    yield {"text": event.delta.text}
            final = await s.get_final_message()

        self._remember(final.content)
        if final.stop_reason == "refusal" and not said:
            yield {"text": "Sorry, I can't help with that one."}
            return
        calls = self._calls(final)
        if calls:
            yield {"tool_calls": calls}

    # -- one-shot, sync: core/gemini.py's ladder ---------------------------------
    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 *, timeout: float | None = None,
                 temperature: float | None = None) -> tuple[str, list[FunctionCall]]:
        kw = self._kwargs(messages, tools, _COMPLETE_MAX_TOKENS)
        try:
            client = (self._client.with_options(timeout=timeout)
                      if timeout else self._client)
            final = client.beta.messages.create(**kw)
        except self._anthropic.APIStatusError as e:
            raise RuntimeError(f"{e.status_code} {_terse(e.message)}") from None
        except self._anthropic.APIConnectionError as e:
            raise RuntimeError(f"could not reach Anthropic: {_terse(e)}") from None
        text = "".join(getattr(b, "text", "") for b in final.content
                       if getattr(b, "type", "") == "text")
        return text, self._calls(final)

    # -- for the settings panel ---------------------------------------------------
    def models(self, timeout: float = 8.0) -> list[str]:
        try:
            page = self._client.with_options(timeout=timeout).models.list()
            found = sorted({m.id for m in page if str(m.id).startswith("claude-")})
        except Exception:
            found = []
        return list(MODELS) + [m for m in found if m not in MODELS]

    def ensure_vision_known(self, save=None) -> bool:
        return True             # every current Claude model reads images

    def probe(self, check_vision: bool = True) -> tuple[bool, str]:
        import time
        t0 = time.monotonic()
        try:
            text, _ = self.complete(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                timeout=30.0)
        except Exception as e:
            msg = str(e)
            if msg.startswith(("401", "403")):
                return False, "Anthropic refused the key."
            if msg.startswith("404"):
                return False, f"No model called '{self.model}' on this key."
            if msg.startswith("429"):
                return False, "Rate limited or out of credit."
            return False, _terse(msg, 90)
        took = time.monotonic() - t0
        return True, f"Answered in {took:.2f}s  |  can see images"


def _make(cfg: dict) -> AnthropicLLM:
    return AnthropicLLM(cfg)


PROVIDER = {
    "id": "anthropic",
    "kind": "llm",
    "label": "Claude (Anthropic)",
    "hint": "Claude Opus · Sonnet · Haiku · Fable — cloud, needs an API key",
    "requires": ("anthropic",),
    "install": "anthropic",
    "needs": (
        Field("api_key", "API key", kind="secret",
              hint="From console.anthropic.com — billed per use"),
        Field("model", "Model", kind="choice", choices=MODELS,
              default=DEFAULT_MODEL,
              hint="Opus is the most capable; Haiku the fastest and cheapest"),
        Field("effort", "Effort", kind="choice", choices=EFFORTS,
              default="low", required=False,
              hint="How hard it thinks. low suits a voice assistant"),
    ),
    "caps": {"tools": True, "tool_format": "anthropic", "vision": True},
    "factory": _make,
}
