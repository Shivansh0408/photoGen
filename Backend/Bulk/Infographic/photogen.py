#!/usr/bin/env python3
"""Main entry point: turn one plain request into a finished infographic.

    python Backend/Core/photogen.py

The program first asks whether to use automatic generation.

* No  -> the manual pipeline runs (ollama_template_planner.py asks for the
         template prompt, then the text stage asks for every field).
* Yes -> type one request, for example::

         create me 5 points on maple leaf on a white background,
         make the boxes brown gradient

         The thinking engine then:
           1. scans svg_dictionary.py and font_dictionary.py and locks every fact the
              user stated (colours, gradients, number of points, layout);
           2. asks Ollama to design the infographic: layout, palette, which approved
              SVG assets fit the topic and where, and which fonts to use;
           3. asks Ollama to write the content: heading, subheading, one short point
              per box and a footer line.
         Every answer is validated against the dictionaries. The design is written
         as a planner prompt (in the phrasing the planner parses), the planner builds
         the layout, the text stage places the generated text without questions,
         and the renderer produces the image.

Everything for a run is saved in gen_assets/<template>_<timestamp>/, including
autogen_brief.json with the engine's reasoning, choices and planner prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ollama_template_planner import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHEMA_FILE,
    NAMED_COLOR_HEX,
    NUMBER_WORDS,
    PlannerError,
    GENERIC_ASSET_WORDS,
    _asset_name_words,
    _asset_words,
    _match_assets_by_usage,
    _word_root,
    create_run_directory,
    extract_hard_constraints,
    generate_plan,
    load_svg_dictionary,
    materialize_plan_assets,
    parse_json_object,
    preprocess_user_prompt,
    request_ollama,
)
from ollama_template_planner import main as planner_main  # noqa: E402
from template_image_renderer import RendererError, render_template  # noqa: E402
from template_text_layer import (  # noqa: E402
    TextLayerError,
    _plan_objects,
    apply_text_spec,
    load_font_dictionary,
    parse_font,
    unload_ollama_model,
)

MAX_POINTS = 12
MAX_EXTRA_ASSETS = 3
LAYOUTS = ("vertical_stack", "grid", "horizontal_row")
# Where a chosen asset may go; each value is the planner phrase for that spot.
ASSET_PLACEMENTS = {
    "inside boxes left": "inside each box on the left",
    "inside boxes right": "inside each box on the right",
    "top": "at the top center",
    "top-left": "in the top left corner",
    "top-right": "in the top right corner",
    "bottom": "at the bottom center",
    "bottom-left": "in the bottom left corner",
    "bottom-right": "in the bottom right corner",
}
WORD_LIMITS = {"heading": 7, "subheading": 16, "title": 5, "detail": 14, "footer": 16}
DEFAULT_HEADING_FONT = "Geometric Modern"
DEFAULT_BODY_FONT = "Corporate Sans"
COUNTED_NOUNS = (r"points?|facts?|tips?|steps?|reasons?|items?|ideas?|benefits?|ways?|"
                 r"stages?|sections?|boxes|cards|panels")


class AutoGenError(RuntimeError):
    """The automatic pipeline cannot continue."""


# --------------------------------------------------------------------------- small helpers


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _ask_yes_no(prompt: str) -> bool:
    while True:
        answer = _ask(f"{prompt} [Y/N]: ").lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False
        print("Please enter Y or N.")


def _hex_color(value: Any) -> str | None:
    """A hex colour or a colour name the planner knows, as #RRGGBB; otherwise None."""

    text = str(value or "").strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}", text):
        return text.upper()
    if re.fullmatch(r"#[0-9a-f]{3}", text):
        return "#" + "".join(ch * 2 for ch in text[1:]).upper()
    return NAMED_COLOR_HEX.get(text.replace("_", " ").replace("-", " "))


def _luminance(color: str) -> float:
    value = color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _readable_on(color: str | None, backgrounds: list[str], fallback: str | None = None) -> str:
    """Keep ``color`` when it stands out from the backgrounds; otherwise pick one that does."""

    average = sum(_luminance(c) for c in backgrounds) / len(backgrounds)
    for candidate in (color, fallback):
        if candidate and abs(_luminance(candidate) - average) >= 0.4:
            return candidate
    return "#1A1A1A" if average >= 0.5 else "#FFFFFF"


def _clean_words(value: Any, limit: int) -> str:
    text = re.sub(r"[*_`#|]+", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip().strip("\"'“”")
    words = text.split()
    return " ".join(words[:limit]).rstrip(",;:-") if len(words) > limit else text


def _explicit_count(request: str) -> int | None:
    """'5 points', 'four key facts', '4 phases of the moon' -> the number of boxes."""

    number = r"\d{1,2}|" + "|".join(NUMBER_WORDS)
    noun = rf"(?:{COUNTED_NOUNS}|(?!(?:colou?rs|fonts|pixels|words|lines)\b)[a-z]{{3,}}s)"
    match = re.search(rf"\b({number})\s+(?:[a-z-]+\s+){{0,2}}{noun}\b", request.lower())
    if not match:
        return None
    token = match.group(1)
    count = int(token) if token.isdigit() else NUMBER_WORDS.get(token)
    return count if count and 1 <= count <= MAX_POINTS else None


def _asset_role(filename: str, asset: dict[str, Any]) -> str:
    """Backgrounds and boxes are chosen by the planner itself; the rest are decorations."""

    words = _asset_name_words(filename, asset)
    if words & {"background", "canvas", "backdrop"}:
        return "background"
    # A plain recolourable shape is a box; a 'credit card' or 'boxer' illustration is not.
    if words & {"box", "card", "panel"} and asset.get("recolorable"):
        return "container"
    return "decoration"


# --------------------------------------------------------------------------- thinking engine


class ThinkingEngine:
    """Plans an infographic from a one-line request, using only approved assets and fonts."""

    def __init__(self, *, assets: dict[str, dict[str, Any]], fonts: dict[str, dict[str, Any]],
                 endpoint: str, model: str, timeout: float, temperature: float) -> None:
        self.assets, self.fonts = assets, fonts
        self.endpoint, self.model = endpoint, model
        self.timeout, self.temperature = timeout, temperature
        self.decorations = {f: a for f, a in assets.items() if _asset_role(f, a) == "decoration"}
        self.ollama_down = False

    # -- step 1: scan the dictionaries and lock what the user stated

    def scan(self, request: str) -> dict[str, Any]:
        print(f"\n[Thinking] Step 1/3: scanning {len(self.assets)} SVG asset(s) and "
              f"{len(self.fonts)} font style(s).")
        matched = _match_assets_by_usage(request, self.decorations, set())
        for filename, asset in self.assets.items():
            role = _asset_role(filename, asset)
            note = "  <- matches the request" if filename in matched else ""
            print(f"    [{role:<10}] {Path(filename).stem:<34} {asset['usage']}{note}")
        if not matched:
            print("    No decoration asset matches the request words; Ollama may still pick "
                  "a generic one that fits the topic.")

        hard = extract_hard_constraints(request, self.assets)
        facts: dict[str, Any] = {
            "point_count": _explicit_count(request) or hard.get("primary_count"),
            "layout": hard.get("layout") if hard.get("layout") in LAYOUTS else None,
            "background": (list(hard["background_gradient"]["colors"]) if hard.get("background_gradient")
                           else [hard["background_color"]] if hard.get("background_color") else None),
            "boxes": (list(hard["primary_gradient"]["colors"]) if hard.get("primary_gradient")
                      else [hard["primary_color"]] if hard.get("primary_color") else None),
            "matched_assets": list(matched),
        }
        stated = {k: v for k, v in facts.items() if v and k != "matched_assets"}
        print(f"[Thinking] Locked from your request: {json.dumps(stated) if stated else 'nothing explicit'}")
        return facts

    # -- Ollama calls

    def _ask_ollama(self, prompt: str, stage: str) -> dict[str, Any] | None:
        if self.ollama_down:
            return None
        for attempt in (1, 2):
            print(f"[Thinking] {stage}: asking Ollama ({self.model})"
                  f"{' again' if attempt == 2 else ''}...")
            try:
                return parse_json_object(request_ollama(
                    endpoint=self.endpoint, model=self.model, prompt=prompt,
                    timeout=self.timeout, temperature=self.temperature))
            except PlannerError as exc:
                print(f"[Thinking] {stage}: {exc}")
                if re.search(r"connect|HTTP", str(exc)):
                    self.ollama_down = True
                    break
        print(f"[Thinking] {stage}: using built-in defaults.")
        return None

    # -- step 2: design

    def _design_prompt(self, request: str, facts: dict[str, Any]) -> str:
        catalog = [{"name": Path(f).stem, "usage": a["usage"],
                    "matches_request": f in facts["matched_assets"]}
                   for f, a in self.decorations.items()]
        fonts = [{"name": name, "usage": font["usage"]} for name, font in self.fonts.items()]
        locked = {k: v for k, v in facts.items() if v and k != "matched_assets"}
        return "\n".join([
            "You are the art director of an INFOGRAPHIC generator. Think step by step about the "
            "request, then design one clear, visually appealing infographic.",
            "Infographic rules:",
            "- One heading, one short subheading, one box per point, optional footer line.",
            "- Strong hierarchy and contrast: heading colour must stand out from the background; "
            "box text must be readable on the box colour.",
            "- A harmonious palette of 2-3 colours that suits the topic and respects the locked facts.",
            "- 'vertical_stack' suits 3-6 points, 'grid' suits 4, 6 or 8 points, "
            "'horizontal_row' suits 2-3 points.",
            "- Decoration assets: pick ONLY names from the asset catalog that genuinely fit the topic "
            "(at most 3, and at most ONE placed inside the boxes). Return an empty list when none "
            "fits. Never invent names.",
            f"- Asset placement must be one of: {', '.join(ASSET_PLACEMENTS)}.",
            "- Fonts: a display font for the heading and a readable font for the body, names "
            "copied exactly from the font catalog.",
            "- Colours are #RRGGBB hex codes.",
            "Locked facts from the user (never change these): " + json.dumps(locked),
            "Asset catalog: " + json.dumps(catalog),
            "Font catalog: " + json.dumps(fonts),
            f"User request: {request[:1500]}",
            "Return ONLY this JSON object:",
            '{"thinking": "2-4 sentences of reasoning about topic, audience, mood and choices",'
            ' "topic": "short topic", "point_count": 5, "layout": "vertical_stack",'
            ' "background": {"color": "#FFFFFF", "gradient_to": null},'
            ' "boxes": {"color": "#8B5A2B", "gradient_to": "#D2A679"},'
            ' "assets": [{"name": "asset name", "placement": "inside boxes left", "reason": "why"}],'
            ' "fonts": {"heading": "font name", "body": "font name"},'
            ' "colors": {"heading": "#hex", "subheading": "#hex", "box_text": "#hex", "footer": "#hex"},'
            ' "visual_prompt": "1-2 sentences describing the finished infographic"}',
        ])

    def design(self, request: str, facts: dict[str, Any]) -> dict[str, Any]:
        print("\n[Thinking] Step 2/3: designing the infographic (layout, palette, assets, fonts).")
        raw = self._ask_ollama(self._design_prompt(request, facts), "Design") or {}
        if raw.get("thinking"):
            print(f"[Thinking] Ollama's reasoning: {_clean_words(raw['thinking'], 90)}")

        count = facts["point_count"]
        if not count:
            try:
                count = int(raw.get("point_count") or 0)
            except (TypeError, ValueError):
                count = 0
        count = min(max(count or 5, 1), MAX_POINTS)

        layout = facts["layout"] or (raw.get("layout") if raw.get("layout") in LAYOUTS else None)
        if layout is None or (layout == "horizontal_row" and count > 4) or (layout == "grid" and count < 4):
            layout = "grid" if count in (4, 8) else "vertical_stack"

        def palette(locked: list[str] | None, proposed: Any, default: list[str]) -> list[str]:
            if locked:
                return locked
            proposed = proposed if isinstance(proposed, dict) else {}
            first, second = _hex_color(proposed.get("color")), _hex_color(proposed.get("gradient_to"))
            return [c for c in (first, second) if c] if first else default

        background = palette(facts["background"], raw.get("background"), ["#FFFFFF"])
        boxes = palette(facts["boxes"], raw.get("boxes"), ["#1F4E79", "#2E75B6"])

        topic_text = f"{request} {raw.get('topic') or ''}"
        assets = self._choose_assets(raw.get("assets"), facts["matched_assets"], topic_text)

        def font(value: Any, default: str) -> str:
            for candidate in (value, default):
                try:
                    return parse_font(str(candidate or ""), self.fonts)
                except ValueError:
                    continue
            return next(iter(self.fonts))

        fonts = raw.get("fonts") if isinstance(raw.get("fonts"), dict) else {}
        colors = raw.get("colors") if isinstance(raw.get("colors"), dict) else {}
        # The darkest box colour is a natural heading colour on a light background.
        accent = min(boxes, key=_luminance) if _luminance(background[0]) >= 0.5 else max(boxes, key=_luminance)
        heading_color = _readable_on(_hex_color(colors.get("heading")), background, accent)
        design = {
            "topic": _clean_words(raw.get("topic") or request, 12),
            "point_count": count,
            "count_locked": bool(facts["point_count"]),
            "layout": layout,
            "background": background,
            "boxes": boxes,
            "assets": assets,
            "fonts": {"heading": font(fonts.get("heading"), DEFAULT_HEADING_FONT),
                      "body": font(fonts.get("body"), DEFAULT_BODY_FONT)},
            "colors": {
                "heading": heading_color,
                "subheading": _readable_on(_hex_color(colors.get("subheading")), background, heading_color),
                "box_text": _readable_on(_hex_color(colors.get("box_text")), boxes),
                "footer": _readable_on(_hex_color(colors.get("footer")), background, heading_color),
            },
            "thinking": _clean_words(raw.get("thinking"), 120),
            "visual_prompt": _clean_words(raw.get("visual_prompt"), 80),
        }
        print(f"[Thinking] Design: {count} point(s), {layout}, background {' -> '.join(background)}, "
              f"boxes {' -> '.join(boxes)}")
        print(f"[Thinking] Fonts: heading '{design['fonts']['heading']}', body '{design['fonts']['body']}'")
        for chosen in assets:
            print(f"[Thinking] Asset: {Path(chosen['file']).stem} {chosen['placement']} ({chosen['reason']})")
        if not assets:
            print("[Thinking] Assets: none (no approved decoration fits this topic).")
        return design

    def _choose_assets(self, proposed: Any, matched: list[str], topic_text: str) -> list[dict[str, str]]:
        # Small models pick loosely (a moon for a maple leaf), so a pick must share a
        # meaningful word with the request or topic; page borders suit any topic.
        topic_words = {_word_root(w) for w in re.findall(r"[a-z]{3,}", topic_text.lower())
                       if w not in GENERIC_ASSET_WORDS}

        def relevant(filename: str) -> bool:
            asset = self.decorations[filename]
            return bool(re.search(r"border|frame|bracket|divider", asset["usage"], re.I)
                        or _asset_words(filename, asset) & topic_words)

        lookup: dict[str, str] = {}
        for filename, asset in self.decorations.items():
            for key in (filename, Path(filename).stem, asset["name"]):
                lookup[key.lower()] = filename
        chosen: dict[str, dict[str, str]] = {}
        for entry in proposed if isinstance(proposed, list) else []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip().lower()
            filename = lookup.get(name) or lookup.get(Path(name).stem)
            if filename is None:
                if name:
                    print(f"[Thinking] Ignoring asset '{entry.get('name')}': not in the SVG dictionary.")
                continue
            if filename not in matched and not relevant(filename):
                print(f"[Thinking] Ignoring asset '{Path(filename).stem}': unrelated to the topic.")
                continue
            placement = str(entry.get("placement") or "").strip().lower()
            placement = placement if placement in ASSET_PLACEMENTS else "inside boxes left"
            # One icon per box reads cleanly; a second one would crowd the text.
            if placement.startswith("inside") and any(
                    c["placement"].startswith("inside") for c in chosen.values()):
                print(f"[Thinking] Skipping asset '{Path(filename).stem}': the boxes already hold an icon.")
                continue
            chosen.setdefault(filename, {
                "file": filename,
                "placement": placement,
                "reason": _clean_words(entry.get("reason") or "fits the topic", 16),
            })
        # An asset the request names outright is always used.
        if matched and not set(matched) & set(chosen):
            chosen[matched[0]] = {"file": matched[0], "placement": "inside boxes left",
                                  "reason": "named in the request"}
        return list(chosen.values())[:MAX_EXTRA_ASSETS]

    # -- step 3: content

    def content(self, request: str, design: dict[str, Any]) -> dict[str, Any]:
        count = design["point_count"]
        print(f"\n[Thinking] Step 3/3: writing the infographic text ({count} point(s)).")
        prompt = "\n".join([
            "You write the text of an INFOGRAPHIC. Be accurate, specific and concise; no filler.",
            f"Topic: {design['topic']}",
            f"User request: {request[:1500]}",
            f"Write exactly {count} points. Each point has a title of at most {WORD_LIMITS['title']} "
            f"words and a detail of at most {WORD_LIMITS['detail']} words.",
            f"Heading: at most {WORD_LIMITS['heading']} words. Subheading: at most "
            f"{WORD_LIMITS['subheading']} words. Footer: one closing line or takeaway, at most "
            f"{WORD_LIMITS['footer']} words.",
            "Plain words only: no markdown, emojis, numbering or quotes.",
            'Return ONLY this JSON object: {"heading": "...", "subheading": "...", '
            '"points": [{"title": "...", "detail": "..."}], "footer": "..."}',
        ])
        raw = self._ask_ollama(prompt, "Content") or {}
        points = []
        for entry in raw.get("points") if isinstance(raw.get("points"), list) else []:
            if isinstance(entry, dict):
                title = _clean_words(entry.get("title"), WORD_LIMITS["title"])
                detail = _clean_words(entry.get("detail"), WORD_LIMITS["detail"])
            else:
                title, detail = _clean_words(entry, WORD_LIMITS["title"]), ""
            if title or detail:
                points.append({"title": title, "detail": detail})
        points = points[:count]
        if points and len(points) < count and not design["count_locked"]:
            # No count was asked for: draw as many boxes as there are real points.
            print(f"[Thinking] Ollama wrote {len(points)} point(s); using {len(points)} boxes.")
            count = design["point_count"] = len(points)
        while len(points) < count:
            points.append({"title": f"Point {len(points) + 1}", "detail": ""})
        content = {
            "heading": _clean_words(raw.get("heading") or design["topic"], WORD_LIMITS["heading"]).title(),
            "subheading": _clean_words(raw.get("subheading"), WORD_LIMITS["subheading"]),
            "points": points,
            "footer": _clean_words(raw.get("footer"), WORD_LIMITS["footer"]),
        }
        print(f"[Thinking] Heading: {content['heading']!r}")
        for number, point in enumerate(points, 1):
            print(f"    {number}. {point['title']} - {point['detail']}")
        return content


# --------------------------------------------------------------------------- pipeline glue


def build_planner_prompt(design: dict[str, Any]) -> str:
    """Write the design in the phrasing ollama_template_planner.py parses reliably."""

    background, boxes, count = design["background"], design["boxes"], design["point_count"]
    parts = [f"Create a {background[0]} to {background[1]} gradient background" if len(background) > 1
             else f"Create a {background[0]} background"]
    box_word = "box" if count == 1 else "boxes"
    parts.append(f"add {count} {box_word} in a {boxes[0]} to {boxes[1]} gradient colour" if len(boxes) > 1
                 else f"add {count} {box_word} in {boxes[0]} colour")
    if design["layout"] == "grid":
        parts.append("arrange the boxes in a grid with 2 in each row centered and equally spaced")
    elif design["layout"] == "horizontal_row":
        parts.append("place the boxes side by side horizontally centered and equally spaced")
    else:
        parts.append("stack the boxes vertically one under the other centered and equally spaced")
        parts.append("make the boxes width 86% of the background")
    parts.append("leave space at the top for heading")
    parts.append("adjust the height of the background if necessary to fit")
    for chosen in design["assets"]:
        parts.append(f"place {Path(chosen['file']).stem} {ASSET_PLACEMENTS[chosen['placement']]}")
    return ", ".join(parts) + "."


def build_text_spec(plan: dict[str, Any], design: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
    """The same spec the text stage collects interactively, filled in automatically."""

    colors, fonts = design["colors"], design["fonts"]
    items: list[dict[str, Any]] = [
        {"id": "heading", "role": "heading", "label": "Heading", "content": content["heading"],
         "placement": "top", "color": colors["heading"], "font": fonts["heading"], "style": ""},
    ]
    if content["subheading"]:
        items.append({"id": "subheading", "role": "subheading", "label": "Subheading",
                      "content": content["subheading"], "placement": "below heading",
                      "color": colors["subheading"], "font": fonts["heading"], "style": ""})
    boxes = sorted((o.label for o in _plan_objects(plan) if o.label.startswith("box-")),
                   key=lambda label: int(label.split("-")[1]))
    if len(boxes) != len(content["points"]):
        print(f"[AutoGen] The plan has {len(boxes)} box(es) for {len(content['points'])} point(s); "
              "extra points are left out.")
    for label, point in zip(boxes, content["points"]):
        text = " | ".join(part for part in (point["title"], point["detail"]) if part)
        items.append({"id": f"{label}-text", "role": "box", "label": f"Text in {label}",
                      "content": text, "placement": "center", "color": colors["box_text"],
                      "font": fonts["body"], "style": "", "box": label})
    if content["footer"]:
        items.append({"id": "text-1", "role": "text", "label": "Footer", "content": content["footer"],
                      "placement": "below boxes" if len(boxes) > 1 else "bottom",
                      "color": colors["footer"], "font": fonts["body"], "style": "small"})
    return {"default_color": colors["box_text"], "default_font": fonts["body"], "items": items}


def run_auto(request: str, args: argparse.Namespace) -> int:
    assets = load_svg_dictionary(args.schema)
    fonts = load_font_dictionary()
    if not fonts:
        raise AutoGenError("The font dictionary is empty or unreadable.")
    engine = ThinkingEngine(assets=assets, fonts=fonts, endpoint=args.ollama_url, model=args.model,
                            timeout=args.timeout, temperature=args.temperature)
    facts = engine.scan(request)
    design = engine.design(request, facts)
    content = engine.content(request, design)
    if design["visual_prompt"]:
        print(f"\n[AutoGen] Visual prompt: {design['visual_prompt']}")
    planner_prompt = build_planner_prompt(design)
    print(f"[AutoGen] Planner prompt: {planner_prompt}\n")

    semantic_spec = preprocess_user_prompt(
        user_prompt=planner_prompt, assets=assets, endpoint=args.ollama_url,
        model=args.model, timeout=args.timeout, temperature=args.temperature)
    plan = generate_plan(
        user_prompt=planner_prompt, assets=assets, semantic_spec=semantic_spec,
        schema_file=args.schema, endpoint=args.ollama_url, model=args.model,
        timeout=args.timeout, temperature=args.temperature)
    run_directory = create_run_directory(args.output_root, plan["template_name"])
    plan_path = materialize_plan_assets(plan, assets, run_directory)
    print(f"[Planner] Plan saved: {plan_path}")
    unload_ollama_model(args.ollama_url, args.model)

    spec = build_text_spec(json.loads(plan_path.read_text(encoding="utf-8")), design, content)
    apply_text_spec(plan_path, spec, original_prompt=request, endpoint=args.ollama_url,
                    model=args.model, timeout=args.timeout, interactive=False)

    brief = {"request": request, "facts": facts, "design": design, "content": content,
             "planner_prompt": planner_prompt}
    brief_path = run_directory / "autogen_brief.json"
    brief_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[AutoGen] Brief saved: {brief_path}")

    if args.plan_only:
        print("[AutoGen] Plan-only mode: rendering skipped.")
        return 0
    svg_path, png_path = render_template(plan_path, png_path=run_directory / "generated_image.png"
                                         if args.png else None)
    print(f"[Renderer] Image saved: {svg_path}")
    if png_path:
        print(f"[Renderer] PNG saved: {png_path}")
    return 0


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photo generator entry point (auto or manual).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout", type=float, default=180.0, help="Seconds per Ollama request.")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Creativity of the thinking engine, 0-2 (default: 0.4).")
    parser.add_argument("--plan-only", action="store_true", help="Do not render the image.")
    parser.add_argument("--png", action="store_true", help="Also render a PNG (needs CairoSVG).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.timeout <= 0 or not 0 <= args.temperature <= 2:
        print("ERROR: --timeout must be positive and --temperature between 0 and 2.", file=sys.stderr)
        return 2

    if not _ask_yes_no("Use automatic infographic generation?"):
        print("[AutoGen] Manual mode: starting the template planner.\n")
        manual = ["--model", args.model, "--ollama-url", args.ollama_url, "--schema", str(args.schema),
                  "--output-root", str(args.output_root), "--timeout", str(args.timeout)]
        manual += ["--plan-only"] if args.plan_only else []
        manual += ["--png"] if args.png else []
        return planner_main(manual)

    print("Describe the infographic (e.g. 'create me 5 points on maple leaf on a white background, "
          "make the boxes brown gradient'):")
    request = _ask("> ")
    if not request:
        print("ERROR: A request is required.", file=sys.stderr)
        return 2
    try:
        return run_auto(request, args)
    except (AutoGenError, PlannerError, TextLayerError, RendererError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
