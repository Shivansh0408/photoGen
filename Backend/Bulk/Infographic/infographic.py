"""Infographic pipeline for the bulk generator (choice 1 in ../user_request.py).

For a topic and a number of images:

1. Asset scout (once per batch): goes through every asset's name and usage in
   svg_dictionary.py and keeps the ones relevant to the topic (Ollama's related
   words are used only when nothing matches directly).
2. Per image, no planner and no design call:
   * a template from Schema/template_dictonary/template_dictionary.py is picked
     programmatically (a shuffled rotation, so every template is used);
   * template_designer edits it: the topic's assets go into its image slots
     (an icon in every text box, illustrations in free slots and empty boxes),
     colours come from the leading asset's usage, fonts are varied;
   * ONE Ollama call writes the text (heading, subheading, one point per text box,
     footer), sized to the template's boxes and different from earlier images;
   * the text stage places the text and the renderer saves the PNG.

This folder also holds the shared rendering engine (planner helpers, text stage,
renderer) that the Motivational pipeline imports. Templates are made with
Backend/template_creator.py.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_template_planner import (  # noqa: E402
    GENERIC_ASSET_WORDS,
    PlannerError,
    _asset_words,
    _word_root,
    create_run_directory,
    load_svg_dictionary,
    materialize_plan_assets,
)
from photogen import ThinkingEngine, _clean_words  # noqa: E402
from template_designer import (  # noqa: E402
    TEMPLATE_FILE,
    PaletteMaker,
    block_lines,
    box_word_limit,
    build_plan,
    build_text_spec,
    edit_template,
    load_templates,
    usable,
)
from template_image_renderer import RendererError, render_template  # noqa: E402
from template_text_layer import (  # noqa: E402
    TextLayerError,
    apply_text_spec,
    load_font_dictionary,
    unload_ollama_model,
)

FRAME_PATTERN = r"border|frame|bracket|divider"


# --------------------------------------------------------------------------- assets


def repair_dimensions(assets: dict[str, dict[str, Any]]) -> None:
    """Use the SVG viewBox when an entry's stated size has the wrong shape.

    A broken entry (e.g. width 0.09 for a 4318x2541 drawing) would otherwise
    squash the asset or stop the whole image.
    """

    for filename, asset in assets.items():
        match = re.search(r"viewBox\s*=\s*[\"']\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)",
                          asset["xml"])
        if not match:
            continue
        view_w, view_h = float(match.group(1)), float(match.group(2))
        width, height = asset["dimensions"]["width"], asset["dimensions"]["height"]
        if view_w <= 0 or view_h <= 0:
            continue
        if abs((width / height) / (view_w / view_h) - 1) > 0.05 or min(width, height) < 8:
            print(f"[Bulk] Warning: {filename} is listed as {width:g}x{height:g} but its drawing is "
                  f"{view_w:g}x{view_h:g}; using {view_w:g}x{view_h:g}. Fix it in svg_dictionary.py.")
            asset["dimensions"]["width"], asset["dimensions"]["height"] = view_w, view_h


def scout_assets(engine: ThinkingEngine, topic: str) -> list[str]:
    """Compare each asset's name and usage with the topic; return the relevant ones.

    Direct matches (a topic word in the name or usage, e.g. 'business' in 'tech in
    business') win. Only when nothing matches directly are Ollama's related words
    used ('space' -> moon). Ollama never sees the catalog here, so it cannot just
    echo asset names back as 'related' words. Page borders are left out: templates
    have no slot for them.
    """

    decorations = engine.decorations
    print(f"\n[Bulk] Scanning the name and usage of {len(decorations)} decoration asset(s) for '{topic}'.")
    frames = {f for f, a in decorations.items() if re.search(FRAME_PATTERN, a["usage"], re.I)}

    def words_of(text: str) -> set[str]:
        return {_word_root(w) for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in GENERIC_ASSET_WORDS}

    topic_words = words_of(topic)
    direct = {f: _asset_words(f, a) & topic_words for f, a in decorations.items() if f not in frames}
    direct = {f: overlap for f, overlap in direct.items() if overlap}

    related: dict[str, set[str]] = {}
    keywords: list[str] = []
    if not direct:
        raw = engine._ask_ollama("\n".join([
            "List up to 8 single-word nouns for things an illustration about this topic could show.",
            f"Topic: {topic[:500]}",
            'Return ONLY this JSON object: {"keywords": ["..."]}',
        ]), "Topic keywords") or {}
        keywords = [str(k).lower() for k in raw.get("keywords") or [] if isinstance(k, str)][:8]
        keyword_words = words_of(" ".join(keywords))
        related = {f: _asset_words(f, a) & keyword_words for f, a in decorations.items() if f not in frames}
        related = {f: overlap for f, overlap in related.items() if overlap}

    for filename, asset in decorations.items():
        if filename in frames:
            why = "page border (not used by templates)"
        elif filename in direct:
            why = f"MATCH: '{', '.join(sorted(direct[filename]))}' in name/usage"
        elif filename in related:
            why = f"related via Ollama keyword '{', '.join(sorted(related[filename]))}'"
        else:
            why = "not related"
        print(f"    {Path(filename).stem[:30]:<30} usage: {asset['usage'][:44]!r:<48} {why}")
    if keywords:
        print(f"[Bulk] No direct match; Ollama's related keywords: {', '.join(keywords)}")
    relevant = list(direct or related)
    print(f"[Bulk] Relevant assets: {', '.join(Path(f).stem for f in relevant) or 'none (text-only images)'}")
    return relevant


# --------------------------------------------------------------------------- per-image choices


class Variety:
    """Remembers earlier images of the batch so fonts, angles and headings differ."""

    def __init__(self, fonts: dict[str, dict[str, Any]], rng: random.Random) -> None:
        self.fonts, self.rng = fonts, rng
        self.headings: list[str] = []
        self.angles: list[str] = []
        self.font_pairs: list[tuple[str, str]] = []

    def pick_fonts(self) -> dict[str, str]:
        # 'Headlines only' styles (Bold Impact, Script Signature) never set body text.
        bodies = [n for n, f in self.fonts.items() if "only" not in f["usage"].lower()]
        used_headings = {h for h, _ in self.font_pairs}
        used_bodies = {b for _, b in self.font_pairs}
        heading = self.rng.choice([n for n in self.fonts if n not in used_headings] or list(self.fonts))
        body = self.rng.choice([n for n in bodies if n not in used_bodies and n != heading] or bodies)
        self.font_pairs.append((heading, body))
        return {"heading": heading, "body": body}


def write_content(engine: ThinkingEngine, topic: str, template: dict[str, Any], edited: dict[str, Any],
                  variety: Variety) -> dict[str, Any]:
    """The one Ollama call per image: all text, sized to this template's boxes."""

    count = len(edited["text_boxes"])
    icon_boxes = {id(b) for b in edited["text_boxes"]} if any(
        i["id"].startswith("icon-") for i in edited["images"]) else set()
    detail = min((box_word_limit(b, id(b) in icon_boxes) for b in edited["text_boxes"]), default=12)
    notes = max(sum(block_lines(template)) - 1, 0)  # The footer is the first line.
    wants_footer = bool(template.get("text_blocks"))  # Only where the template has room for it.
    prompt = "\n".join(filter(None, [
        "You write the text of ONE infographic in a series about a topic. Be accurate, specific "
        "and concise; no filler, markdown, emojis, numbering or quotes.",
        f"Topic: {topic[:500]}",
        "First choose a specific angle of the topic (e.g. history, benefits, facts, how-to, "
        "comparison, myths, tips) that differs from the earlier images.",
        f"Earlier angles (do not repeat): {'; '.join(variety.angles)}" if variety.angles else "",
        f"Earlier headings (do not reuse): {'; '.join(variety.headings)}" if variety.headings else "",
        f"Write exactly {count} points; each has a title of at most 4 words and a detail of at most "
        f"{detail} words." if count else "Write no points.",
        "Heading: at most 7 words. Subheading: at most 14 words.",
        "Footer: one closing takeaway of at most 14 words." if wants_footer else "",
        f"Notes: {notes} extra short facts of at most 12 words each." if notes else "",
        'Return ONLY this JSON object: {"angle": "...", "heading": "...", "subheading": "...", '
        '"points": [{"title": "...", "detail": "..."}], "footer": "...", "notes": ["..."]}',
    ]))
    raw = engine._ask_ollama(prompt, "Content") or {}

    def parse_points(answer: dict[str, Any]) -> list[dict[str, str]]:
        found = []
        for entry in answer.get("points") if isinstance(answer.get("points"), list) else []:
            if isinstance(entry, dict):
                title, text = _clean_words(entry.get("title"), 5), _clean_words(entry.get("detail"), detail)
                if title or text:
                    found.append({"title": title, "detail": text})
        return found

    points = parse_points(raw)
    if raw and 0 < count - len(points):
        # Small models sometimes stop early: ask only for the missing points.
        missing = count - len(points)
        print(f"[Bulk] Ollama wrote {len(points)} of {count} points; asking for the other {missing}.")
        more = engine._ask_ollama("\n".join([
            f"Infographic about {topic[:300]}, angle: {raw.get('angle') or topic}.",
            f"Existing points: {'; '.join(p['title'] for p in points) or 'none'}.",
            f"Write exactly {missing} NEW points; each has a title of at most 4 words and a detail "
            f"of at most {detail} words. Plain words only.",
            'Return ONLY this JSON object: {"points": [{"title": "...", "detail": "..."}]}',
        ]), "More points") or {}
        points += parse_points(more)
    while len(points) < count:
        points.append({"title": f"Point {len(points) + 1}", "detail": ""})
    heading = _clean_words(raw.get("heading") or topic, 7).title()
    if heading.lower() in (h.lower() for h in variety.headings):
        heading = _clean_words(f"{raw.get('angle') or 'More On'} {topic}", 7).title()
    content = {
        "angle": _clean_words(raw.get("angle") or topic, 10),
        "heading": heading,
        "subheading": _clean_words(raw.get("subheading"), 14),
        "points": points[:count],
        "footer": _clean_words(raw.get("footer"), 14) if wants_footer else "",
        "notes": [_clean_words(n, 12) for n in (raw.get("notes") or [])[:notes] if isinstance(n, str)],
    }
    variety.angles.append(content["angle"])
    variety.headings.append(content["heading"])
    return content


# --------------------------------------------------------------------------- one image


def build_image(index: int, total: int, template: dict[str, Any], pool: list[str], topic: str,
                engine: ThinkingEngine, palettes: PaletteMaker, variety: Variety,
                assets: dict[str, dict[str, Any]], batch_dir: Path, work_root: Path,
                args: argparse.Namespace, rng: random.Random) -> Path | None:
    print(f"\n[Bulk] ===== Image {index}/{total}: template '{template['name']}' =====")
    edited = edit_template(template, pool, pool, assets, rng)
    icon = next((i["asset"] for i in edited["images"] if i["id"].startswith("icon-")), None)
    lead = icon or (edited["images"][0]["asset"] if edited["images"] else None)
    palette = palettes.palette(lead, topic, rng, icon)
    fonts = variety.pick_fonts()
    for note in edited["notes"]:
        print(f"[Bulk] Template edit: {note}")
    print(f"[Bulk] Colours from {palette['reason']}: background {' -> '.join(palette['background'])}, "
          f"boxes {' -> '.join(palette['boxes'])}")
    print(f"[Bulk] Fonts: heading '{fonts['heading']}', body '{fonts['body']}'")

    content = write_content(engine, topic, template, edited, variety)
    print(f"[Bulk] Angle: {content['angle']} | Heading: {content['heading']!r}")

    plan = build_plan(template, edited, palette, assets, topic)
    spec = build_text_spec(template, edited, content, palette, fonts)
    run_directory = create_run_directory(work_root, template["name"])
    try:
        plan_path = materialize_plan_assets(plan, assets, run_directory)
        apply_text_spec(plan_path, spec, original_prompt=f"{topic}: {content['angle']}",
                        endpoint=args.ollama_url, model=args.model, timeout=args.timeout,
                        interactive=False)
        before = set(batch_dir.glob("*.png"))
        render_template(plan_path, png_dir=batch_dir)
        created = sorted(set(batch_dir.glob("*.png")) - before)
    finally:
        shutil.rmtree(run_directory, ignore_errors=True)  # Only the PNG is kept.
    return created[0] if created else None


IMAGE_ERRORS = (PlannerError, TextLayerError, RendererError, OSError, ValueError, KeyError)


def load_usable_templates(path: Path) -> dict[str, dict[str, Any]]:
    """Templates with somewhere to put points; raises ValueError when there are none."""

    try:
        templates = load_templates(path)
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    for name in [n for n, t in templates.items() if not usable(t)]:
        print(f"[Bulk] Skipping template '{name}': it has no text boxes or text areas for the points.")
        templates.pop(name)
    if not templates:
        raise ValueError(f"No usable templates in {path}. Create some with "
                         "'python Backend/template_creator.py' first.")
    return templates


def run(topic: str, total: int, args: argparse.Namespace, batch_dir: Path,
        work_root: Path) -> tuple[list[Path], int]:
    """Create ``total`` infographics in ``batch_dir``; returns (saved PNGs, failures)."""

    templates = load_usable_templates(args.templates or TEMPLATE_FILE)
    assets = load_svg_dictionary(args.schema)
    repair_dimensions(assets)
    fonts = load_font_dictionary()
    if not fonts:
        raise ValueError("The font dictionary is empty or unreadable.")
    rng = random.Random(args.seed)
    engine = ThinkingEngine(assets=assets, fonts=fonts, endpoint=args.ollama_url, model=args.model,
                            timeout=args.timeout, temperature=args.temperature)
    relevant = scout_assets(engine, topic)
    palettes = PaletteMaker(engine, assets)
    variety = Variety(fonts, rng)

    # Programmatic template choice: a shuffled rotation, so every template is used evenly.
    names = list(templates)
    rng.shuffle(names)
    print(f"[Bulk] {len(templates)} template(s): {', '.join(names)}")

    saved: list[Path] = []
    failed = 0
    for index in range(1, total + 1):
        template = templates[names[(index - 1) % len(names)]]
        # Each image leads with a different relevant asset.
        shift = (index - 1) % len(relevant) if relevant else 0
        pool = relevant[shift:] + relevant[:shift]
        try:
            png = build_image(index, total, template, pool, topic, engine, palettes, variety,
                              assets, batch_dir, work_root, args, rng)
        except IMAGE_ERRORS as exc:
            print(f"[Bulk] Image {index} failed: {exc}")
            png = None
        if png:
            saved.append(png)
        else:
            failed += 1
    return saved, failed
