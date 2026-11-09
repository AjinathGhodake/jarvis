"""
Life OS — read-only access to a Life OS vault (an Obsidian markdown wiki).

Read-only by design: the vault is written only by its own sync operations,
never by this assistant. Pages marked `private: true` in their frontmatter are
never returned, and `raw/` plus tooling folders are skipped, so private content
cannot end up in a spoken answer, a saved memory, or a web-search query.

Every read and search is reported to the Life OS Brain dashboard's Agents tab
(POST /api/access on loopback) as agent "mark-lvi" — path and size only, never
content. A dashboard that is not running is silently ignored.

Vault location: plugin_config.life_os.vault_path in config/api_keys.json,
default ~/life-os.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

PLUGIN = {
    "name": "life_os",
    "tier": "core",
    "description": (
        "Read the user's personal Life OS wiki (goals, projects, people, daily "
        "notes, career, learning). Use for questions about the user's own life, "
        "plans, schedule, goals, open loops, projects or people they know. "
        "action: today (latest daily note), goals (active goals), "
        "search (needs query), read (needs query = page name). "
        "Read-only. Not for general web facts — use web_search for those."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING",
                       "description": "today | goals | search | read"},
            "query": {"type": "STRING",
                      "description": "Search words, or a page name for read"},
        },
        "required": ["action"],
    },
}

MAX_CHARS = 2500          # a local model pays for every character, every turn
SEARCH_FILES = 6
SKIP_DIRS = {"raw", ".git", ".obsidian", ".claude", "docs", "tools", "scripts",
             "tests", "node_modules", ".trash"}
# Agent operating instructions, not wiki content — and CLAUDE.md describes
# where secrets live.
SKIP_FILES = {"CLAUDE.md", "AGENTS.md", "SKILL.md"}
_FRONTMATTER_LINE = re.compile(
    r"^(title|type|updated|tags|confidence|sources|private|status|started|relationship):",
    re.I)
BRAIN_URLS = os.environ.get("LIFEOS_BRAIN_URLS", "http://127.0.0.1:8831").split()


def _vault() -> Path:
    try:
        from memory.config_manager import get_plugin_setting
        p = get_plugin_setting("life_os", "vault_path", "") or ""
    except Exception:
        p = ""
    return Path(os.path.expanduser(p or "~/life-os")).resolve()


def _report(op: str, path: Path, chars: int) -> None:
    body = json.dumps({"agent": "mark-lvi", "op": op, "path": str(path),
                       "chars": chars}).encode()

    def send():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for url in BRAIN_URLS:
            req = urllib.request.Request(url.rstrip("/") + "/api/access", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                opener.open(req, timeout=1.0).close()
            except Exception:
                pass
    threading.Thread(target=send, daemon=True).start()


def _allowed(vault: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(vault)
    except ValueError:
        return False                       # outside the vault (symlink, ../)
    if (path.suffix != ".md" or path.name in SKIP_FILES
            or any(part in SKIP_DIRS for part in rel.parts[:-1])):
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(2000)
    except OSError:
        return False
    if head.startswith("---"):
        fm = head.split("\n---", 1)[0]
        if re.search(r"^private:\s*true\s*$", fm, re.M | re.I):
            return False
    return True


def _body(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        parts = text.split("\n---", 1)
        if len(parts) == 2:
            text = parts[1].lstrip("-\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n…(truncated)"


def _page(vault: Path, path: Path) -> str:
    if not _allowed(vault, path):
        return "That page is private or outside the Life OS wiki, so I can't read it."
    out = _clip(_body(path))
    _report("read", path, len(out))
    return f"[{path.relative_to(vault).with_suffix('')}]\n{out}"


def _today(vault: Path) -> str:
    daily = vault / "journal" / "daily"
    today = daily / f"{_dt.date.today():%Y-%m-%d}.md"
    if today.exists():
        return _page(vault, today)
    notes = sorted(p for p in daily.glob("20??-??-??.md"))
    if not notes:
        return "There are no daily notes in the Life OS wiki yet."
    return f"No note for today yet; the latest is {notes[-1].stem}.\n" + _page(vault, notes[-1])


def _all_pages(vault: Path) -> list[Path]:
    out = []
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        out.extend(Path(root) / f for f in files if f.endswith(".md"))
    return out


def _read(vault: Path, query: str) -> str:
    q = query.strip().strip("[]").removesuffix(".md").strip("/")
    if not q:
        return "Tell me which page to read."
    direct = (vault / f"{q}.md")
    if direct.is_file():
        return _page(vault, direct)
    key = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")
    pages = [p for p in _all_pages(vault) if _allowed(vault, p)]
    exact = [p for p in pages if p.stem.lower() == key]
    loose = [p for p in pages if key and key in p.stem.lower()]
    hits = exact or loose
    if not hits:
        return _search(vault, query)
    hits.sort(key=lambda p: (len(p.stem), str(p)))
    return _page(vault, hits[0])


def _search(vault: Path, query: str) -> str:
    q = query.strip()
    if not q:
        return "Tell me what to search for."
    words = [w for w in re.findall(r"\w{3,}", q) if w.lower() not in
             {"the", "and", "what", "my", "about", "for", "with", "from"}] or [q]
    pattern = "|".join(re.escape(w) for w in words)
    rg = shutil.which("rg") or "/opt/homebrew/bin/rg"
    counts: dict[Path, int] = {}
    lines: dict[Path, list[str]] = {}
    try:
        res = subprocess.run(
            [rg, "-i", "-n", "--no-heading", "-g", "*.md",
             *sum((["-g", f"!{d}/**"] for d in SKIP_DIRS), []),
             "-e", pattern, str(vault)],
            capture_output=True, text=True, timeout=10)
        for line in res.stdout.splitlines():
            m = re.match(r"^(.*?\.md):(\d+):(.*)$", line)
            if not m:
                continue
            p = Path(m.group(1))
            if _FRONTMATTER_LINE.match(m.group(3).strip()):
                continue
            counts[p] = counts.get(p, 0) + 1
            if len(lines.setdefault(p, [])) < 2:
                lines[p].append(m.group(3).strip()[:200])
    except Exception as e:
        return f"Life OS search failed: {e}"
    # A filename match outranks a passing mention.
    for p in list(counts):
        if any(w.lower() in p.stem.lower() for w in words):
            counts[p] += 50
    ranked = [p for p in sorted(counts, key=lambda p: -counts[p]) if _allowed(vault, p)]
    if not ranked:
        return f"Nothing in the Life OS wiki matches '{q}'."
    out = [f"Life OS pages matching '{q}':"]
    for p in ranked[:SEARCH_FILES]:
        out.append(f"- {p.relative_to(vault).with_suffix('')}: " + " | ".join(lines[p]))
    text = _clip("\n".join(out))
    _report("search", vault, len(text))
    return text + "\n(Use action=read with a page name for the full page.)"


def run(parameters: dict, player=None, session_memory=None) -> str:
    action = str(parameters.get("action") or "").strip().lower()
    query = str(parameters.get("query") or "")
    vault = _vault()
    if not vault.is_dir():
        return f"I can't find the Life OS wiki at {vault}."
    try:
        if action == "today":
            result = _today(vault)
        elif action == "goals":
            result = _page(vault, vault / "goals" / "active.md")
        elif action == "read":
            result = _read(vault, query)
        elif action == "search" or query:
            result = _search(vault, query)
        else:
            result = "Use action today, goals, search or read."
    except Exception as e:
        result = f"Life OS lookup failed: {e}"
    if player:
        try:
            player.write_log(f"LIFE OS: {action} {query}".strip())
        except Exception:
            pass
    return result
