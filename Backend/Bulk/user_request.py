#!/usr/bin/env python3
"""Bulk entry point: choose the kind of image, then one topic in, many PNGs out.

    python Backend/Bulk/user_request.py

Asks, in order:
    1. the kind of image: 1 = Infographic, 2 = Motivational
    2. the topic (or theme)
    3. how many images to create

    Infographic/   templates from Schema/template_dictonary filled with topic
                   assets, colours from asset usage, one Ollama call per image
                   for the text (also holds the shared rendering engine).
    Motivational/  a scenic background (Assets/backgrounds), a recoloured person
                   (Assets/Humans) and a short quote, one Ollama call per image.

All PNGs of one batch are saved in one folder, gen_bulk/<topic>_<YYYYMMDD_HHMM>/,
named by generation time like the Core pipeline (17102909.png, 17102909_2.png).
Intermediate plan files are written to gen_bulk/.work/ and deleted per image.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

BULK_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(BULK_DIRECTORY / "Motivational"))
sys.path.insert(0, str(BULK_DIRECTORY / "Infographic"))

from ollama_template_planner import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SCHEMA_FILE,
    PROJECT_ROOT,
    PlannerError,
    _slugify,
)
from template_designer import TEMPLATE_FILE  # noqa: E402
from template_text_layer import unload_ollama_model  # noqa: E402

DEFAULT_BULK_ROOT = PROJECT_ROOT / "gen_bulk"
MAX_IMAGES = 50
KINDS = {"1": "Infographic", "2": "Motivational"}


def _ask(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def ask_request() -> tuple[str, str, int] | None:
    """(kind, topic, number of images); None when input ends."""

    print("What kind of images should be created?")
    for key, name in KINDS.items():
        print(f"  {key} = {name}")
    while True:
        kind = _ask("Choose 1 or 2: ")
        if kind is None:
            return None
        if kind in KINDS:
            break
        print("Please enter 1 or 2.")
    noun = "theme of the motivational posters" if kind == "2" else "topic of the infographics"
    while True:
        topic = _ask(f"The {noun}: ")
        if topic is None:
            return None
        if topic:
            break
        print("Please type a topic.")
    while True:
        raw = _ask(f"How many images to create (1-{MAX_IMAGES}): ")
        if raw is None:
            return None
        if raw.isdigit() and 1 <= int(raw) <= MAX_IMAGES:
            return kind, topic, int(raw)
        print(f"Please enter a whole number from 1 to {MAX_IMAGES}.")


def create_batch_directory(root: Path, topic: str) -> Path:
    base = f"{_slugify(topic, 'batch')[:40]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    candidate, number = root / base, 2
    while candidate.exists():
        candidate, number = root / f"{base}_{number}", number + 1
    candidate.mkdir(parents=True)
    return candidate


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create many infographic or motivational PNGs for one topic.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_FILE)
    parser.add_argument("--templates", type=Path, default=TEMPLATE_FILE,
                        help="Infographic templates made with Backend/template_creator.py.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BULK_ROOT)
    parser.add_argument("--timeout", type=float, default=180.0, help="Seconds per Ollama request.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Creativity of the text writer, 0-2 (default: 0.7 for variety).")
    parser.add_argument("--seed", type=int, help="Repeat the same layout, colour and font choices.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.timeout <= 0 or not 0 <= args.temperature <= 2:
        print("ERROR: --timeout must be positive and --temperature between 0 and 2.", file=sys.stderr)
        return 2
    answer = ask_request()
    if answer is None:
        print("ERROR: The kind of image, a topic and a number of images are required.", file=sys.stderr)
        return 2
    kind, topic, total = answer
    if kind == "1":
        import infographic as pipeline
    else:
        import motivational as pipeline

    root = args.output_root.expanduser().resolve()
    work_root = root / ".work"
    batch_dir = create_batch_directory(root, topic)
    print(f"[Bulk] {KINDS[kind]}: creating {total} image(s) in {batch_dir}")
    try:
        saved, failed = pipeline.run(topic, total, args, batch_dir, work_root)
    except (PlannerError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if not any(batch_dir.iterdir()):
            batch_dir.rmdir()
        return 1
    finally:
        unload_ollama_model(args.ollama_url, args.model)
        if work_root.is_dir() and not any(work_root.iterdir()):
            work_root.rmdir()

    print(f"\n[Bulk] Done: {len(saved)} image(s) saved, {failed} failed.")
    print(f"[Bulk] Folder: {batch_dir}")
    for png in saved:
        print(f"    {png.name}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
