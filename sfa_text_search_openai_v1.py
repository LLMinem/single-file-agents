# /// script
# Dependencies for uv single-file execution
# rapidfuzz==3.6.1
# rich==13.7.0
# tqdm==4.66.4
# ripgrep==14.1.0  # installs precompiled binary across platforms
# openai>=1.23.0  # optional, only required when --llm flag is used
# ///

"""SFA Text Search Agent (OpenAI) v1

An agentic markdown / text search tool that scans directories for files whose
content matches a natural-language prompt. It performs a two-phase search:
    1. Quick snippet triage (reads the first *N* lines of each file).
    2. Full-file scan if snippet appears relevant.

Optionally, the top candidates can be re-ranked using an LLM (OpenAI).

Example usage
-------------
```bash
uv run sfa_text_search_openai_v1.py \
    -d ~/notes ~/projects \
    -p "Find markdown files describing project ideas using GPT" \
    --depth 7 --top-k 50
```
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import shutil

from rapidfuzz import fuzz  # type: ignore
from rich.console import Console
from rich.table import Table
from rich.theme import Theme
from tqdm import tqdm

# --------------------------- CLI -------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search markdown / text files for content relevant to a prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dirs",
        nargs="+",
        required=True,
        help="Directories to search (one or more).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="Natural-language query describing what to look for.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Maximum folder depth to traverse below each root directory.",
    )
    parser.add_argument(
        "--ext",
        default=".md,.txt",
        help="Comma-separated list of file extensions to include.",
    )
    parser.add_argument(
        "--snippet-lines",
        type=int,
        default=40,
        help="Number of initial lines to read for quick triage phase.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10000,
        help="Hard cap on number of files to inspect (safety guard).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of matches to display after ranking.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use OpenAI to re-rank the top candidates for higher accuracy.",
    )
    parser.add_argument(
        "-c",
        "--compute-loops",
        type=int,
        default=10,
        help="Compatibility flag with other agents (unused but preserved).",
    )
    return parser


# ----------------------- File collection ------------------------------------ #

def calc_depth(base: Path, child: Path) -> int:
    try:
        return len(child.relative_to(base).parts)
    except ValueError:
        return 0  # child not under base


def collect_files(root_dirs: Iterable[str], depth: int, extensions: set[str]) -> List[Path]:
    """Collect files up to *depth* levels below *root_dirs* with given extensions.

    Attempts to use `ripgrep --files` for speed; falls back to Python os.walk.
    """
    collected: List[Path] = []

    # Try ripgrep first
    rg_path = shutil.which("rg")  # type: ignore
    if rg_path:
        for rd in root_dirs:
            for ext in extensions:
                cmd = [
                    rg_path,
                    "--files",
                    "--max-depth",
                    str(depth),
                    "-g",
                    f"*{ext}",
                    str(rd),
                ]
                try:
                    out = subprocess.check_output(cmd, text=True)
                    collected.extend(Path(p).resolve() for p in out.splitlines())
                except subprocess.CalledProcessError:
                    # ripgrep returns non-zero if no files match; ignore.
                    pass
    else:
        for rd in root_dirs:
            base = Path(rd).expanduser().resolve()
            for current_root, dirs, files in os.walk(base):
                cur_path = Path(current_root)
                if calc_depth(base, cur_path) > depth:
                    # Prune traversal by clearing dirs in-place
                    dirs.clear()
                    continue
                for fname in files:
                    if any(fname.endswith(ext) for ext in extensions):
                        collected.append(cur_path / fname)

    return collected


# ----------------------- Scoring helpers ------------------------------------ #


def read_snippet(path: Path, lines: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return "\n".join([next(f).rstrip("\n") for _ in range(lines)])
    except (FileNotFoundError, StopIteration):
        return ""


def fuzzy_score(prompt: str, text: str) -> int:
    return int(fuzz.partial_ratio(prompt, text))


def full_scan_score(path: Path, prompt: str) -> Tuple[int, str]:
    """Return best fuzzy score across lines & the best-matching snippet."""
    best_score = 0
    best_line = ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                score = fuzzy_score(prompt, line)
                if score > best_score:
                    best_score = score
                    best_line = line.strip()
    except FileNotFoundError:
        pass
    return best_score, best_line[:120]


# ----------------------- LLM re-ranking ------------------------------------- #


def llm_rerank(prompt: str, records: List[dict]) -> List[dict]:
    """Call OpenAI to rerank snippets. Requires OPENAI_API_KEY env var."""
    try:
        import openai
    except ImportError:
        print("[LLM] openai package not installed. Skipping rerank.")
        return records

    client = openai.OpenAI()
    # Build messages: we feed a JSON list of {index, path, snippet}
    import json

    content = json.dumps([
        {"index": i, "path": r["path"], "snippet": r["snippet"]}
        for i, r in enumerate(records)
    ])

    system_msg = (
        "You are a helpful assistant that ranks file snippets for relevance to a user query. "
        "Return a JSON array of indexes sorted from most to least relevant."
    )
    user_msg = f"QUERY: {prompt}\nSNIPPETS: {content}"

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",  # small default model
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        import json as _json

        order = _json.loads(resp.choices[0].message.content)
        ordered = [records[i] for i in order if 0 <= i < len(records)]
        return ordered
    except Exception as e:
        print(f"[LLM] Re-ranking failed: {e}")
        return records


# ----------------------- Rendering ------------------------------------------ #

def render_results(records: List[dict]):
    console = Console(theme=Theme({"repr.str": "green"}))
    if not console.is_terminal:
        # Plain text if not a TTY
        for r in records:
            print(f"{r['score']:>4} | {r['path']} | {r['snippet']}")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Path", overflow="fold")
    table.add_column("Score", justify="right")
    table.add_column("Snippet", overflow="fold")

    for idx, rec in enumerate(records, 1):
        table.add_row(str(idx), str(rec["path"]), str(rec["score"]), rec["snippet"])

    console.print(table)


# ----------------------- Main ------------------------------------------------ #

def main(argv: List[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.ext.split(",")}
    prompt = args.prompt.strip()

    # Collect files
    print("[INFO] Collecting candidate files…", file=sys.stderr)
    files = collect_files(args.dirs, args.depth, extensions)
    if not files:
        print("No files found matching the given extensions.", file=sys.stderr)
        sys.exit(1)
    if len(files) > args.max_files:
        files = files[: args.max_files]
        print(f"[WARN] Truncated file list to first {args.max_files} entries.", file=sys.stderr)

    # Phase 1: snippet triage
    triaged: List[Tuple[Path, int]] = []
    for path in tqdm(files, desc="Snippet scan", unit="file", disable=len(files) < 100):
        snippet_text = read_snippet(path, args.snippet_lines)
        score = fuzzy_score(prompt, snippet_text)
        if score >= 30:
            triaged.append((path, score))

    if not triaged:
        print("No relevant files after snippet triage.")
        sys.exit(0)

    # Phase 2: full scan (multi-threaded)
    print("[INFO] Performing full-file relevance scoring…", file=sys.stderr)
    scored: List[dict] = []
    with ThreadPoolExecutor() as ex:
        fut_map = {ex.submit(full_scan_score, p, prompt): p for p, _ in triaged}
        for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Full scan", unit="file"):
            path = fut_map[fut]
            score, best_line = fut.result()
            scored.append({"path": path, "score": score, "snippet": best_line})

    # Sort and trim
    scored.sort(key=lambda r: r["score"], reverse=True)
    scored = scored[: args.top_k]

    # Optional LLM re-rank
    if args.llm and scored:
        print("[INFO] Re-ranking with LLM…", file=sys.stderr)
        scored = llm_rerank(prompt, scored)

    # Display
    render_results(scored)


if __name__ == "__main__":
    main() 