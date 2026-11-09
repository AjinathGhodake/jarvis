"""
Claude through the Claude Code CLI — runs on the user's Claude subscription.

Each turn runs `claude -p` headless as a plain language model:
  --tools ""               none of Claude Code's own tools (files, bash, web)
  --strict-mcp-config      no MCP servers
  --setting-sources ""     no user/project settings, so no hooks or plugins
  --system-prompt          the assistant's own prompt replaces Claude Code's
  --no-session-persistence nothing saved to Claude Code's session history
and from a neutral working directory, so no project CLAUDE.md is picked up.
The CLI authenticates with its own login; this provider never reads or copies
a credential. ANTHROPIC_API_KEY is removed from the child's environment so a
stray key can never turn subscription use into per-token billing.

TOOLS
    The CLI cannot be handed the assistant's tool schemas as native tools, so
    they are described in the system prompt and Claude answers a tool call with
    one JSON object. Unlike the pipeline's generic "prompt" mode, the JSON is
    caught HERE — a reply that opens with "{" is held back, parsed, and yielded
    as a real tool call — so it is never streamed to the voice and read aloud.

COST OF THIS SHAPE
    Every turn is a fresh process (~2-3 s before the first word) and resends the
    whole transcript. Fine for a voice assistant; the API-key provider
    (llm_anthropic) is faster and keeps Claude's thinking between turns.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from importlib.util import find_spec  # noqa: F401  (parity with other providers)
from pathlib import Path
from typing import Any, AsyncIterator

from .base import DEFAULT_CAPS, Field, FunctionCall

MODELS = ("sonnet", "opus", "haiku")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_WORKDIR = Path.home() / ".mark-lvi" / "claude-cwd"


def _claude_bin(cfg: dict | None = None) -> str:
    explicit = str((cfg or {}).get("claude_bin") or "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    return shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")


def available() -> bool:
    return Path(_claude_bin()).exists()


def _terse(text: str, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _tool_instructions(tools: list[dict]) -> str:
    lines = [
        "# Tools",
        "You can use the tools below. To call one, reply with exactly ONE JSON "
        "object and nothing else — no words before or after it, no code fence:",
        '{"tool": "<tool name>", "args": {<arguments>}}',
        "The result comes back in the next message, and then you answer the "
        "user in plain spoken words. When no tool is needed, just answer in "
        "plain words with no JSON. Never describe a tool call instead of "
        "making it.",
        "",
    ]
    for t in tools or []:
        name = t.get("name") or ""
        if not name:
            continue
        schema = t.get("input_schema") or t.get("parameters") or {}
        lines.append(f"## {name}")
        desc = " ".join(str(t.get("description") or "").split())
        if desc:
            lines.append(desc)
        lines.append("args schema: " + json.dumps(schema, separators=(",", ":")))
        lines.append("")
    return "\n".join(lines)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(str(p.get("text") or ""))
        elif isinstance(p, dict) and p.get("type") == "image_url":
            parts.append("[an image was attached; this model cannot see it]")
    return "\n".join(parts)


def _transcript(messages: list[dict]) -> tuple[str, str]:
    """(system prompt, transcript prompt) from the pipeline's history."""
    system_parts, lines = [], []
    for m in messages:
        role = m.get("role")
        if role == "system":
            if not lines:
                system_parts.append(_text_of(m.get("content")))
            else:
                lines.append(f"[Note] {_text_of(m.get('content'))}")
        elif role == "user":
            lines.append(f"User: {_text_of(m.get('content'))}")
        elif role == "assistant":
            text = _text_of(m.get("content")).strip()
            if text:
                lines.append(f"Assistant: {text}")
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                lines.append(f"Assistant called tool: "
                             f'{{"tool": "{fn.get("name")}", '
                             f'"args": {fn.get("arguments") or "{}"}}}')
        elif role == "tool":
            lines.append(f"Tool result ({m.get('name') or 'tool'}): "
                         f"{_text_of(m.get('content'))}")
    prompt = ("<conversation>\n" + "\n".join(lines) + "\n</conversation>\n\n"
              "Reply as the Assistant to the latest message — either one tool "
              "call as JSON, or a spoken answer in plain words.")
    return "\n\n".join(p for p in system_parts if p), prompt


def _parse_call(text: str, known: set[str], n: int) -> FunctionCall | None:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    if "{" not in s:
        return None
    s = s[s.find("{"):s.rfind("}") + 1]
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("tool") or obj.get("name") or "").strip()
    if name not in known:
        return None
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    return FunctionCall(id=f"cc_{n}_{name}", name=name, args=args)


class ClaudeCodeLLM:
    """Claude via `claude -p`, on the Claude Code login."""

    def __init__(self, cfg: dict):
        self.bin = _claude_bin(cfg)
        self.model = (cfg.get("model") or "").strip() or "sonnet"
        self.effort = (cfg.get("effort") or "low").strip().lower()
        if self.effort not in EFFORTS:
            self.effort = "low"
        self.timeout = float(cfg.get("timeout") or 180.0)
        self._calls = 0
        self.caps = dict(DEFAULT_CAPS)
        self.caps.update({"tools": True, "tool_format": "anthropic",
                          "vision": False, "max_tools": None,
                          "max_tool_chars": None})

    # -- process ---------------------------------------------------------------
    def _cmd(self, system: str, stream: bool) -> list[str]:
        cmd = [self.bin, "-p",
               "--model", self.model,
               "--effort", self.effort,
               "--tools", "",
               "--strict-mcp-config",
               "--setting-sources", "",
               "--no-session-persistence",
               "--disable-slash-commands",
               "--system-prompt", system or "You are a helpful assistant."]
        if stream:
            cmd += ["--output-format", "stream-json", "--verbose",
                    "--include-partial-messages"]
        else:
            cmd += ["--output-format", "json"]
        return cmd

    @staticmethod
    def _env() -> dict:
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)      # subscription only, never billed per token
        return env

    def _prepare(self, messages: list[dict], tools: list[dict] | None):
        system, prompt = _transcript(messages)
        if tools:
            system = (system + "\n\n" + _tool_instructions(tools)).strip()
        _WORKDIR.mkdir(parents=True, exist_ok=True)
        return system, prompt, {t.get("name") for t in tools or [] if t.get("name")}

    # -- streaming, async: the pipeline engine ----------------------------------
    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   *, stream: bool = True,
                   temperature: float | None = None) -> AsyncIterator[dict]:
        system, prompt, known = self._prepare(messages, tools)
        proc = await asyncio.create_subprocess_exec(
            *self._cmd(system, True), cwd=str(_WORKDIR), env=self._env(),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024)
        mode = ""           # "" undecided, "talk" streaming words, "json" held back
        held = ""
        result: dict = {}
        try:
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            async for raw in proc.stdout:
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                kind = obj.get("type")
                if kind == "result":
                    result = obj
                    continue
                if kind != "stream_event":
                    continue
                ev = obj.get("event") or {}
                delta = ev.get("delta") or {}
                if ev.get("type") != "content_block_delta" or delta.get("type") != "text_delta":
                    continue
                piece = delta.get("text") or ""
                if mode == "talk":
                    yield {"text": piece}
                    continue
                held += piece
                if mode == "" and held.strip():
                    mode = "json" if held.lstrip()[:1] in "{`" else "talk"
                    if mode == "talk":
                        yield {"text": held}
                        held = ""
            await proc.wait()
        finally:
            if proc.returncode is None:     # barge-in / cancelled turn
                proc.kill()
                await proc.wait()

        if result.get("is_error") or (proc.returncode not in (0, None) and not result):
            err = (await proc.stderr.read()).decode("utf-8", "replace")
            raise RuntimeError("Claude Code: " + _terse(result.get("result") or err
                                                       or f"exit {proc.returncode}"))
        if mode == "json":
            self._calls += 1
            call = _parse_call(held, known, self._calls)
            if call:
                yield {"tool_calls": [call]}
            elif held.strip():
                yield {"text": held}

    # -- one-shot, sync: core/gemini.py's ladder ---------------------------------
    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 *, timeout: float | None = None,
                 temperature: float | None = None) -> tuple[str, list[FunctionCall]]:
        system, prompt, known = self._prepare(messages, tools)
        try:
            r = subprocess.run(self._cmd(system, False), input=prompt.encode(),
                               cwd=str(_WORKDIR), env=self._env(),
                               capture_output=True, timeout=timeout or self.timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("timed out waiting for Claude Code") from None
        try:
            obj = json.loads(r.stdout or b"{}")
        except Exception:
            obj = {}
        if r.returncode != 0 or obj.get("is_error"):
            raise RuntimeError("Claude Code: " + _terse(
                obj.get("result") or r.stderr.decode("utf-8", "replace")
                or f"exit {r.returncode}"))
        text = str(obj.get("result") or "")
        if tools and text.lstrip()[:1] in "{`":
            self._calls += 1
            call = _parse_call(text, known, self._calls)
            if call:
                return "", [call]
        return text, []

    # -- for the settings panel ---------------------------------------------------
    def models(self, timeout: float = 8.0) -> list[str]:
        return list(MODELS)

    def ensure_vision_known(self, save=None) -> bool:
        return False

    def probe(self, check_vision: bool = True) -> tuple[bool, str]:
        import time
        if not Path(self.bin).exists():
            return False, f"Claude Code not found at {self.bin}"
        t0 = time.monotonic()
        try:
            text, _ = self.complete(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                timeout=60.0)
        except Exception as e:
            msg = str(e)
            if "login" in msg.lower() or "auth" in msg.lower():
                return False, "Claude Code is not logged in — run `claude` once."
            return False, _terse(msg, 90)
        return True, f"Answered in {time.monotonic() - t0:.2f}s  |  text only"


def _make(cfg: dict) -> ClaudeCodeLLM:
    return ClaudeCodeLLM(cfg)


PROVIDER = {
    "id": "claude_code",
    "kind": "llm",
    "label": "Claude (your subscription)",
    "hint": "Runs the Claude Code CLI on your Claude login — no API key",
    "requires": (),
    "needs": (
        Field("model", "Model", kind="choice", choices=MODELS, default="sonnet",
              hint="sonnet: balanced · opus: smartest · haiku: fastest"),
        Field("effort", "Effort", kind="choice", choices=EFFORTS,
              default="low", required=False,
              hint="How hard it thinks. low suits a voice assistant"),
        Field("claude_bin", "Claude Code path", kind="path", required=False,
              advanced=True, hint="Leave empty to find `claude` on PATH"),
    ),
    "caps": {"tools": True, "tool_format": "anthropic", "vision": False},
    "factory": _make,
}
