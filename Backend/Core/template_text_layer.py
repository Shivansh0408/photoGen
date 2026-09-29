#!/usr/bin/env python3
"""Structured text stage between template planning and image rendering.

The planner is not changed by this stage. After a plan is saved, the user fills
in one field at a time (heading, subheading, text for each box, extra text
blocks, colours, and styles), and gives each field its own placement:

    Heading text:        Sales Report
      placement:         right of boxes
      colour:            white
      style:             upper, bold

Each field is a separate chunk. Exact wording is used as typed. Content that
starts with ``ai:`` goes to Ollama as its own small request, which also
receives the text already resolved for earlier fields (so "a number bigger than
box-1" can be honoured). Ollama timeouts are retried once with a longer limit;
after that the user can type the text or skip the field, and the image is still
produced.

The layout engine then places every block. If a requested placement has too
little room, it first moves the planned objects slightly (at most
``MAX_SHIFT_FRACTION`` of the canvas, keeping stacked objects together), then
reduces the font size, and only then accepts a best-effort position.

This module can also run on its own against an existing plan::

    python Backend/Core/template_text_layer.py [template_plan.json] [--spec text_spec.json]
"""

from __future__ import annotations

import argparse
import ast
import http.client
import json
import math
import re
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class TextLayerError(RuntimeError):
    """The requested text could not be safely placed or generated."""


class _OllamaTimeout(TextLayerError):
    """Ollama did not answer within the time limit."""


class _OllamaUnavailable(TextLayerError):
    """Ollama could not be reached or refused the request."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_DIRECTORY = PROJECT_ROOT / "Schema" / "font_dictnoary"
FONT_SCHEMA_FILE = FONT_DIRECTORY / "font_dictionary.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "gen_assets"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

LINE_HEIGHT = 1.28            # Must match template_image_renderer.py.
MAX_SHIFT_FRACTION = 0.12     # Largest object move, as a fraction of the canvas side.
MIN_SCALE = 0.45              # Smallest font scale before a placement is accepted as is.
MIN_FONT_SIZE = 10.0
GROW_BOXES_BELOW_SCALE = 0.75  # Boxes grow once their text has shrunk to 75% of normal.
MAX_BOX_GROWTH = 0.20          # Boxes grow at most 20% (all boxes of one kind together).
MIN_OBJECT_SCALE = 0.70        # Illustrations shrink at most to 70% to make room for text.
HEADING_TO_BOX_RATIO = 1.25   # The heading is always at least this much bigger than box text.
MAX_NUDGES = 60               # Object moves per layout; afterwards text only shrinks.
ROLE_BASE_SIZE = {"heading": 72, "subheading": 40, "box": 44, "text": 32}  # px on a 1080 canvas
ROLE_WEIGHT = {"heading": "bold", "subheading": "600", "box": "bold", "text": "normal"}
ROLE_WORD_LIMIT = {"heading": 8, "subheading": 18, "box": 16, "text": 60}
SIZE_WORDS = {"xs": 0.6, "small": 0.75, "medium": 1.0, "normal-size": 1.0,
              "large": 1.35, "big": 1.35, "xl": 1.7, "huge": 1.7}


# --------------------------------------------------------------------------- Ollama


def _generate_url(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        raise TextLayerError("The Ollama endpoint cannot be empty.")
    return endpoint if endpoint.endswith("/api/generate") else endpoint + "/api/generate"


def unload_ollama_model(endpoint: str, model: str) -> bool:
    """Unload this model from Ollama; leave the Ollama server running."""

    request = urllib.request.Request(
        _generate_url(endpoint),
        data=json.dumps({"model": model, "prompt": "", "keep_alive": 0, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        return True
    except (OSError, urllib.error.URLError, TimeoutError, http.client.HTTPException):
        return False


def _request_json(endpoint: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    """Send one chunk to Ollama. Timeouts and connection failures raise distinct errors."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        # Keep the model loaded between chunks; it is unloaded once all fields are done.
        "keep_alive": "5m",
        "options": {"temperature": 0.3, "num_predict": 300},
    }
    request = urllib.request.Request(
        _generate_url(endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _OllamaUnavailable(f"Ollama returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise _OllamaTimeout(f"no answer within {timeout:g}s") from exc
        raise _OllamaUnavailable(f"cannot reach Ollama ({exc.reason})") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise _OllamaTimeout(f"no answer within {timeout:g}s") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise _OllamaUnavailable(f"connection to Ollama failed ({exc})") from exc

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TextLayerError("Ollama returned an invalid response body.") from exc
    answer = envelope.get("response") if isinstance(envelope, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        error = envelope.get("error") if isinstance(envelope, dict) else None
        if error:
            raise _OllamaUnavailable(f"Ollama error: {error}")
        raise TextLayerError("Ollama returned no text.")
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(answer[answer.index("{"):])
        except (ValueError, json.JSONDecodeError) as exc:
            raise TextLayerError("Ollama did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise TextLayerError("Ollama must return a JSON object.")
    return parsed


def _clean_text(text: str) -> str:
    """Normalise wording; '|' (or a newline) is an explicit line break."""

    text = text.replace("\\n", "\n")
    lines = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\s*\|\s*|\n", text)]
    return "\n".join(line for line in lines if line)


def _limit_words(text: str, limit: int) -> str:
    tokens = re.findall(r"\S+\s*", text)
    if len(tokens) <= limit:
        return text
    cut = "".join(tokens[:limit]).rstrip()
    sentence_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if sentence_end > len(cut) * 0.6:
        return cut[:sentence_end + 1]
    return cut.rstrip(",;:-") + "..."


class _ChunkWriter:
    """Writes 'ai:' fields one at a time, with timeout recovery per chunk."""

    def __init__(self, *, endpoint: str, model: str, timeout: float,
                 original_prompt: str, interactive: bool) -> None:
        self.endpoint, self.model, self.timeout = endpoint, model, timeout
        self.original_prompt = original_prompt
        self.interactive = interactive
        self.unavailable: str | None = None
        self.used = False

    def write(self, item: "TextItem", instruction: str,
              context: list[tuple[str, str]]) -> str | None:
        if self.unavailable:
            return self._fallback(item, self.unavailable)
        limit = _word_limit(instruction, item.role)
        feedback: str | None = None
        timeout = self.timeout
        timed_out_once = False
        for _ in range(4):
            prompt = self._prompt(item, instruction, context, limit, feedback)
            print(f"[Text] Chunk '{item.label}': asking Ollama (timeout {timeout:g}s)...")
            self.used = True
            try:
                answer = _request_json(self.endpoint, self.model, prompt, timeout)
            except _OllamaTimeout as exc:
                if timed_out_once:
                    return self._fallback(item, f"Ollama timed out twice ({exc})")
                timed_out_once = True
                timeout *= 2
                print(f"[Text] Ollama timed out ({exc}); the model may still be loading. "
                      f"Retrying with {timeout:g}s...")
                continue
            except _OllamaUnavailable as exc:
                self.unavailable = str(exc)
                return self._fallback(item, str(exc))
            except TextLayerError as exc:
                feedback = f"Your previous answer was invalid: {exc}"
                continue
            text = answer.get("text")
            if not isinstance(text, str) or not _clean_text(text):
                feedback = 'Your previous answer had no "text" string.'
                continue
            text = _clean_text(re.sub(r"[*_`#]+", "", text).strip().strip("\"'“”"))
            words = len(text.split())
            if words > limit * 1.5 and feedback is None:
                feedback = f"Your previous answer used {words} words; use at most {limit}."
                continue
            return _limit_words(text, limit)
        return self._fallback(item, "Ollama did not return usable text")

    def _prompt(self, item: "TextItem", instruction: str, context: list[tuple[str, str]],
                limit: int, feedback: str | None) -> str:
        parts = [
            "You write the wording for ONE text field of a finished graphic design.",
            'Return ONLY a JSON object of the form {"text": "..."}.',
            "Rules:",
            "- Write only the final words shown on the image. Never repeat, describe or "
            "explain the instruction itself (no words like 'bigger than' or 'label').",
            "- Plain words only: no markdown, no surrounding quotes, no coordinates, colours or SVG.",
            f"- Use at most {limit} words.",
            "- Put ' | ' between lines only when the instruction asks for separate lines "
            "(for example a label followed by a number).",
            "- When a number is requested, write one specific, realistic value.",
            "- Stay consistent with text already written for other fields: reuse their "
            "numbers exactly and respect comparisons such as 'bigger than'.",
            f"Field: {item.label}",
            f"Instruction: {instruction[:1500]}",
        ]
        if context:
            parts.append("Text already written for other fields:")
            parts.extend(f'- {label}: "{text}"' for label, text in context)
        if self.original_prompt:
            parts.append(f"Overall design brief (context only): {self.original_prompt[:1500]}")
        if feedback:
            parts.append(feedback)
        return "\n".join(parts)

    def _fallback(self, item: "TextItem", reason: str) -> str | None:
        print(f"[Text] Could not generate '{item.label}': {reason}.")
        if not self.interactive:
            print(f"[Text] Skipping '{item.label}'.")
            return None
        typed = _ask(f"  Type the text for {item.label} yourself (blank to skip)")
        return _clean_text(typed) or None


def _word_limit(instruction: str, role: str) -> int:
    match = re.search(r"\b(\d{1,3})\s+words?\b", instruction, re.I)
    return max(1, int(match.group(1))) if match else ROLE_WORD_LIMIT[role]


# --------------------------------------------------------------------------- geometry


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def moved(self, dx: float = 0.0, dy: float = 0.0) -> "Rect":
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def overlaps(self, other: "Rect", tolerance: float = 0.5) -> bool:
        return (min(self.right, other.right) - max(self.x, other.x) > tolerance
                and min(self.bottom, other.bottom) - max(self.y, other.y) > tolerance)


def _contains(outer: Rect, inner: Rect, tolerance: float = 1.0) -> bool:
    return (inner.x >= outer.x - tolerance and inner.y >= outer.y - tolerance
            and inner.right <= outer.right + tolerance and inner.bottom <= outer.bottom + tolerance)


def _union(rects: list[Rect]) -> Rect:
    x, y = min(r.x for r in rects), min(r.y for r in rects)
    return Rect(x, y, max(r.right for r in rects) - x, max(r.bottom for r in rects) - y)


@dataclass
class PlanObject:
    id: str
    label: str
    asset: str
    index: int
    base: Rect
    rotation: float
    dx: float = 0.0
    dy: float = 0.0
    dw: float = 0.0  # Size changes made by the text stage (grown boxes, shrunk art).
    dh: float = 0.0

    @property
    def rect(self) -> Rect:
        return Rect(self.base.x + self.dx, self.base.y + self.dy,
                    self.base.w + self.dw, self.base.h + self.dh)

    def set_rect(self, rect: Rect) -> None:
        self.dx, self.dy = rect.x - self.base.x, rect.y - self.base.y
        self.dw, self.dh = rect.w - self.base.w, rect.h - self.base.h


def _plan_objects(plan: dict[str, Any]) -> list[PlanObject]:
    """Every placed element except the full-canvas background."""

    canvas_w, canvas_h = float(plan["canvas"]["width"]), float(plan["canvas"]["height"])
    objects: list[PlanObject] = []
    for index, element in enumerate(plan.get("elements", [])):
        if not isinstance(element, dict):
            continue
        try:
            x, y = float(element["x"]), float(element["y"])
            w, h = float(element["width"]), float(element["height"])
            rotation = float(element.get("rotation", 0))
            z_index = float(element.get("z_index", 0))
        except (TypeError, ValueError, KeyError) as exc:
            raise TextLayerError(f"Invalid placement for element {element.get('id')}.") from exc
        if not all(math.isfinite(v) for v in (x, y, w, h, rotation)) or w <= 0 or h <= 0:
            raise TextLayerError(f"Invalid dimensions for element {element.get('id')}.")
        if z_index <= -1000 or (w >= canvas_w * 0.98 and h >= canvas_h * 0.98):
            continue
        identifier = str(element.get("id") or f"element-{index + 1}")
        objects.append(PlanObject(identifier, identifier, str(element.get("asset", "")),
                                  index, Rect(x, y, w, h), rotation))
    boxes = sorted((o for o in objects if re.search(r"box|card|panel", o.asset, re.I)),
                   key=lambda o: (o.base.y, o.base.x))
    for number, box in enumerate(boxes, 1):
        box.label = f"box-{number}"
    return objects


# --------------------------------------------------------------------------- field parsing

CSS_COLOR_NAMES = set("""
aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue
blueviolet brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk
crimson cyan darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki
darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen
darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue
dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite
gold goldenrod gray grey green greenyellow honeydew hotpink indianred indigo ivory khaki
lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan
lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen
lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen
magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen
mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream
mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid
palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum
powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown
seagreen seashell sienna silver skyblue slateblue slategray slategrey snow springgreen
steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow yellowgreen
""".split())
EXTRA_COLORS = {"cream": "#FFFDD0", "offwhite": "#FAF9F6", "navyblue": "navy"}


def parse_color(raw: str) -> str:
    value = raw.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{3}(?:[0-9a-f]{3})?", value):
        return value.upper()
    if re.fullmatch(r"(?:rgb|rgba|hsl|hsla)\([0-9.,%\s]+\)", value):
        return value
    joined = re.sub(r"[\s_-]+", "", value)
    if joined in EXTRA_COLORS:
        return EXTRA_COLORS[joined]
    if joined in CSS_COLOR_NAMES:
        return joined
    raise ValueError(f"'{raw}' is not a colour. Use a name (white, dark green) or #RRGGBB.")


@dataclass
class Style:
    case: str | None = None
    weight: str | None = None
    scale: float = 1.0
    size_px: float | None = None
    align: str | None = None
    shadow: bool = False


def parse_style(raw: str) -> Style:
    style = Style()
    for token in re.split(r"[\s,;/]+", raw.strip().lower()):
        if not token or token in ("none", "default", "all", "text", "font", "-"):
            continue
        if token in ("upper", "uppercase", "caps", "allcaps", "capital", "capitals"):
            style.case = "upper"
        elif token in ("lower", "lowercase"):
            style.case = "lower"
        elif token in ("title", "titlecase"):
            style.case = "title"
        elif token in ("as-typed", "astyped", "keep", "keepcase", "original"):
            style.case = "keep"
        elif token in ("shadow", "dropshadow", "drop-shadow"):
            style.shadow = True
        elif token in ("bold", "heavy"):
            style.weight = "bold"
        elif token in ("semibold", "medium-weight", "600"):
            style.weight = "600"
        elif token in ("normal", "regular", "light", "plain"):
            style.weight = "normal"
        elif token in SIZE_WORDS:
            style.scale = SIZE_WORDS[token]
        elif match := re.fullmatch(r"(\d{1,3}(?:\.\d+)?)px", token):
            style.size_px = float(match.group(1))
        elif token in ("left", "right"):
            style.align = token
        elif token in ("center", "centre", "centered", "centred", "middle"):
            style.align = "center"
        else:
            raise ValueError(
                f"Unknown style word '{token}'. Use upper/lower/title/as-typed, "
                "bold/semibold/normal, small/medium/large/xl or a size like 40px, "
                "left/center/right, and shadow."
            )
    if style.size_px is not None and not 8 <= style.size_px <= 400:
        raise ValueError("Font size must be between 8px and 400px.")
    return style


def load_font_dictionary(schema_file: Path = FONT_SCHEMA_FILE) -> dict[str, dict[str, Any]]:
    """Read FONT_DICTIONARY without executing the file; invalid entries are skipped."""

    try:
        tree = ast.parse(schema_file.read_text(encoding="utf-8"), filename=str(schema_file))
    except (OSError, UnicodeError, SyntaxError) as exc:
        print(f"[Text] Font dictionary unavailable ({exc}); using the default font.")
        return {}
    raw: Any = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "FONT_DICTIONARY" for target in node.targets):
            try:
                raw = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                raw = None
            break
    fonts: dict[str, dict[str, Any]] = {}
    for key, entry in (raw or {}).items() if isinstance(raw, dict) else []:
        try:
            family = str(entry["family"])
            if not re.fullmatch(r"[A-Za-z0-9 ,'\"-]{1,200}", family):
                raise ValueError("unsafe family")
            fonts[str(key)] = {
                "name": str(entry.get("name") or key),
                "usage": str(entry.get("usage") or ""),
                "family": family,
                "weights": {role: str((entry.get("weights") or {}).get(role) or ROLE_WEIGHT[role])
                            for role in ROLE_WEIGHT},
                "letter_spacing": {
                    "heading": float((entry.get("letter_spacing") or {}).get("heading") or 0),
                    "body": float((entry.get("letter_spacing") or {}).get("body") or 0),
                },
                "line_height": min(max(float(entry.get("line_height") or LINE_HEIGHT), 0.9), 2.5),
                "width_factor": min(max(float(entry.get("width_factor") or 1.0), 0.5), 2.0),
                "heading_case": entry.get("heading_case") if entry.get("heading_case") in (
                    "upper", "lower", "title") else None,
                "file": entry.get("file") or None,
            }
        except (KeyError, TypeError, ValueError, AttributeError):
            print(f"[Text] Skipping invalid font entry '{key}'.")
    return fonts


def parse_font(raw: str, fonts: dict[str, dict[str, Any]]) -> str:
    """Accept a font's number (in dictionary order) or its name (or a unique prefix)."""

    value = raw.strip().lower()
    names = list(fonts)
    if value.isdigit() and 1 <= int(value) <= len(names):
        return names[int(value) - 1]
    exact = [name for name in names if name.lower() == value]
    if exact:
        return exact[0]
    prefix = [name for name in names if name.lower().startswith(value)] if value else []
    if len(prefix) == 1:
        return prefix[0]
    raise ValueError(f"Unknown font '{raw}'. Pick 1-{len(names)} or a name: {', '.join(names)}.")


def _font_overlay_fields(item: "TextItem", size: float) -> dict[str, Any]:
    """Rendering data for the renderer; empty for legacy (pre-font) specs."""

    fields: dict[str, Any] = {}
    if item.font:
        fields.update({
            "font": item.font_key,
            "font_family": item.font["family"],
            "letter_spacing": round(item.tracking * size, 2),
            "line_height": item.line_height,
        })
        if item.font.get("file"):
            fields["font_file"] = str(item.font["file"])
    if item.style.shadow:
        fields["shadow"] = True
    return fields


@dataclass
class Placement:
    kind: str                     # anchor | inside | above | below | left | right | xy
    raw: str
    target_kind: str = ""         # objects | text
    target_ids: list[str] = field(default_factory=list)
    vertical: str = "center"      # anchor row, or alignment inside/beside a target
    horizontal: str = "center"    # anchor column
    align_hint: str | None = None
    x: float = 0.0
    y: float = 0.0
    w: float | None = None


CANVAS_WORDS = {"screen", "canvas", "page", "image", "picture", "frame", "poster", "design"}
ANCHOR_FILLER = {"the", "of", "at", "in", "on", "a", "to", "corner", "hand", "side", "edge",
                 "area", "part", "place", "it", "put"} | CANVAS_WORDS
RELATIONS = [
    (r"(?:above|over|on top of|atop)", "above"),
    (r"(?:to the )?left(?: hand)?(?: side)? of", "left"),
    (r"(?:to the )?right(?: hand)?(?: side)? of|next to|beside", "right"),
    (r"(?:below|under|beneath|underneath)", "below"),
    (r"(?:inside|within|in|on)", "inside"),
]
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6}


def _anchor_words(text: str) -> tuple[str, str] | None:
    vertical = horizontal = None
    centred = False
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in ANCHOR_FILLER:
            continue
        if word in ("left", "right"):
            if horizontal not in (None, word):
                return None
            horizontal = word
        elif word in ("top", "upper", "bottom", "lower"):
            value = "top" if word in ("top", "upper") else "bottom"
            if vertical not in (None, value):
                return None
            vertical = value
        elif word in ("center", "centre", "middle", "centered", "centred"):
            centred = True
        else:
            return None
    if vertical is None and horizontal is None and not centred:
        return None
    return vertical or "center", horizontal or "center"


class _Targets:
    """Names a placement can refer to: planned objects and earlier text fields."""

    def __init__(self, objects: list[PlanObject]) -> None:
        self.objects = objects
        self.boxes = [o for o in objects if o.label.startswith("box-")]
        self.text_ids: list[str] = []

    def resolve(self, phrase: str) -> tuple[str, list[str]]:
        p = re.sub(r"\s+", " ", phrase.strip().lower().strip(" .,:;"))
        p = re.sub(r"^(?:the|a|an|both)\s+", "", p)
        if p in CANVAS_WORDS:
            return "canvas", []

        text_name = None
        if re.fullmatch(r"(?:main )?(?:heading|title|headline)", p):
            text_name = "heading"
        elif re.fullmatch(r"sub[ -]?(?:heading|title|headline)", p):
            text_name = "subheading"
        elif match := re.fullmatch(r"(?:text|block|additional text)[ _-]?(\d+)", p):
            text_name = f"text-{match.group(1)}"
        elif match := re.fullmatch(r"(?:box|card|panel)[ _-]?(\d+)[ _-]?text", p):
            text_name = f"box-{match.group(1)}-text"
        if text_name:
            if text_name not in self.text_ids:
                raise ValueError(f"'{text_name}' has not been entered yet; text can only be "
                                 "placed relative to fields entered before it.")
            return "text", [text_name]

        if re.fullmatch(r"(?:all )?(?:boxes|cards|panels)|box group", p) or (
                p in ("box", "card", "panel") and len(self.boxes) > 1):
            if not self.boxes:
                raise ValueError("The plan has no boxes.")
            return "objects", [b.label for b in self.boxes]
        if p in ("objects", "all objects", "everything", "shapes", "elements"):
            if not self.objects:
                raise ValueError("The plan has no objects besides the background.")
            return "objects", [o.label for o in self.objects]

        number = None
        if match := re.fullmatch(r"(first|second|third|fourth|fifth|sixth|1st|2nd|3rd|4th|5th|6th|"
                                 r"last|top|bottom) (?:box|card|panel)", p):
            word = match.group(1)
            number = (len(self.boxes) if word in ("last", "bottom") else
                      1 if word == "top" else ORDINALS[word])
        elif match := re.fullmatch(r"(?:box|card|panel)[ _-]?(\d+)", p):
            number = int(match.group(1))
        elif p in ("box", "card", "panel") and self.boxes:
            number = 1
        if number is not None:
            if not 1 <= number <= len(self.boxes):
                raise ValueError(f"There is no box {number}; the plan has {len(self.boxes)}.")
            return "objects", [f"box-{number}"]

        for obj in self.objects:
            if p in (obj.label.lower(), obj.id.lower(), Path(obj.asset).stem.lower(), obj.asset.lower()):
                return "objects", [obj.label]
        # Name words also work: "mallet" or "gavel" finds 'gavel-mallet-1'.
        wanted = {w for w in re.findall(r"[a-z]{3,}", p) if w not in ("the", "image", "icon", "picture")}
        if wanted:
            found = [
                obj for obj in self.objects
                if wanted <= set(re.findall(r"[a-z]{3,}", re.sub(
                    r"(?<=[a-z])(?=[A-Z])", " ", f"{obj.id} {Path(obj.asset).stem}").lower()))
            ]
            if len(found) == 1:
                return "objects", [found[0].label]
        raise ValueError(f"Unknown target '{phrase.strip()}'. Known targets: {', '.join(self.names())}.")

    def names(self) -> list[str]:
        names = ["screen"]
        if len(self.boxes) > 1:
            names.append("boxes")
        names += [o.label for o in self.objects] + self.text_ids
        return names


def parse_placement(raw: str, targets: _Targets, *, own_box: str | None = None) -> Placement:
    text = re.sub(r"\s+", " ", raw.strip().lower())
    if not text:
        raise ValueError("A placement is required.")

    # Explicit pixels: "x,y" or "x,y,width".
    if match := re.fullmatch(r"(\d+(?:\.\d+)?)\s*[, ]\s*(\d+(?:\.\d+)?)(?:\s*[, ]\s*(\d+(?:\.\d+)?))?", text):
        width = float(match.group(3)) if match.group(3) else None
        if width is not None and width < 20:
            raise ValueError("The width in x,y,width must be at least 20px.")
        return Placement("xy", raw, x=float(match.group(1)), y=float(match.group(2)), w=width)

    # A box field accepts a bare position inside its own box: "top", "bottom left".
    if own_box and (anchor := _anchor_words(text)):
        return Placement("inside", raw, "objects", [own_box], vertical=anchor[0],
                         align_hint=None if anchor[1] == "center" else anchor[1])

    # "top of box 1", "bottom left of the boxes", "top right of the screen".
    if match := re.fullmatch(r"(?:at |in |on )?(?:the )?(top|bottom|center|centre|middle)"
                             r"(?:[ -](left|right|center|centre))? (?:of|in|inside) (.+)", text):
        kind, ids = targets.resolve(match.group(3))
        vertical = "center" if match.group(1) in ("center", "centre", "middle") else match.group(1)
        horizontal = match.group(2) if match.group(2) in ("left", "right") else "center"
        if kind == "canvas":
            return Placement("anchor", raw, vertical=vertical, horizontal=horizontal)
        if kind == "text":
            raise ValueError("Use 'above' or 'below' to place text relative to other text.")
        return Placement("inside", raw, kind, ids, vertical=vertical,
                         align_hint=None if horizontal == "center" else horizontal)

    for pattern, relation in RELATIONS:
        match = re.fullmatch(r"(?:place (?:it )?|put (?:it )?|at |to )?(?:the )?"
                             + f"(?:{pattern})" + r" (.+)", text)
        if not match:
            continue
        phrase, vertical, horizontal = match.group(1), None, None
        trailing = re.fullmatch(r"(.+?)[ ,(]+(?:at (?:the )?)?(top|bottom|center|centre|middle)"
                                r"(?: (left|right|center|centre))?\)?", phrase)
        if trailing and trailing.group(1).strip() not in ("", "the"):
            try:
                targets.resolve(trailing.group(1))
                phrase, vertical, horizontal = trailing.groups()
            except ValueError:
                pass
        kind, ids = targets.resolve(phrase)
        vertical = "center" if vertical in ("centre", "middle") else vertical
        if kind == "canvas":
            if relation in ("above", "below"):
                raise ValueError("Use 'top' or 'bottom' for the edges of the screen.")
            return Placement("anchor", raw, vertical=vertical or "center",
                             horizontal=relation if relation in ("left", "right") else "center")
        if relation == "inside" and kind == "text":
            raise ValueError("Text cannot be placed inside other text.")
        default_vertical = "center" if relation == "inside" else "top"
        return Placement(relation, raw, kind, ids, vertical=vertical or default_vertical,
                         align_hint=horizontal if horizontal in ("left", "right") else None)

    if anchor := _anchor_words(text):
        return Placement("anchor", raw, vertical=anchor[0], horizontal=anchor[1])

    # "Top center above the box": the relation decides; leading position words are extra.
    relation = re.search(r"\b(?:above|over|on top of|below|under|beneath|underneath|inside|"
                         r"within|(?:to the )?(?:left|right)(?: hand)?(?: side)? of|next to|beside)\b", text)
    if relation and relation.start() > 0:
        placement = parse_placement(text[relation.start():], targets, own_box=own_box)
        placement.raw = raw
        return placement
    raise ValueError(
        f"Could not understand placement '{raw}'. Examples: top, top-left, bottom-right, "
        "right of boxes, below box-2, inside box-1, top of box-1, below heading, 40,120"
    )


@dataclass
class TextItem:
    id: str
    role: str
    label: str
    content: str
    generate: bool
    placement: Placement
    color: str
    style: Style
    text: str = ""
    font_key: str | None = None
    font: dict[str, Any] | None = None  # Entry from FONT_DICTIONARY; None = legacy Arial.

    @property
    def weight(self) -> str:
        if self.style.weight:
            return self.style.weight
        if self.font:
            return str(self.font["weights"].get(self.role) or ROLE_WEIGHT[self.role])
        return ROLE_WEIGHT[self.role]

    @property
    def width_factor(self) -> float:
        return float(self.font["width_factor"]) if self.font else 1.0

    @property
    def tracking(self) -> float:
        """Letter spacing in em."""
        if not self.font:
            return 0.0
        return float(self.font["letter_spacing"]["heading" if self.role == "heading" else "body"])

    @property
    def line_height(self) -> float:
        return float(self.font["line_height"]) if self.font else LINE_HEIGHT

    @property
    def case(self) -> str | None:
        if self.style.case:
            return None if self.style.case == "keep" else self.style.case
        if self.font and self.role == "heading":
            return self.font.get("heading_case")
        return None


def _apply_case(text: str, case: str | None) -> str:
    if case == "upper":
        return text.upper()
    if case == "lower":
        return text.lower()
    if case == "title":
        return "\n".join(" ".join(w[:1].upper() + w[1:] for w in line.split(" "))
                         for line in text.split("\n"))
    return text


# --------------------------------------------------------------------------- measuring


def _char_width(ch: str) -> float:
    if ch == " ":
        return 0.28
    if ord(ch) > 0x2E80:
        return 1.0
    if ch in "il.,:;'!|`":
        return 0.28
    if ch in "fjrtI()[]-":
        return 0.36
    if ch in "mwMW@%":
        return 0.9
    if ch.isupper() or ch in "&#":
        return 0.7
    if ch.isdigit():
        return 0.58
    return 0.54


def _is_heavy(weight: str) -> bool:
    return weight == "bold" or (weight.isdigit() and int(weight) >= 600)


def _text_width(text: str, size: float, weight: str, width_factor: float = 1.0,
                tracking: float = 0.0) -> float:
    factor = (1.08 if _is_heavy(weight) else 1.0) * width_factor
    return sum(_char_width(ch) for ch in text) * size * factor + tracking * size * len(text)


def _wrap(text: str, width: float, size: float, weight: str, *, break_words: bool,
          width_factor: float = 1.0, tracking: float = 0.0) -> list[str]:
    def measure(line: str) -> float:
        return _text_width(line, size, weight, width_factor, tracking)

    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            if break_words and measure(word) > width:
                if current:
                    lines.append(current)
                piece = ""
                for character in word:
                    if piece and measure(piece + character) > width:
                        lines.append(piece)
                        piece = ""
                    piece += character
                current = piece
            elif current and measure(current + " " + word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
    return lines


# --------------------------------------------------------------------------- layout


@dataclass
class _Placed:
    item: TextItem
    rect: Rect
    column: Rect
    lines: list[str]
    size: float
    align: str
    rotation: float


@dataclass
class _Conflict:
    item: TextItem
    reason: str
    key: str
    push: list[PlanObject] = field(default_factory=list)
    moves: list[tuple[float, float]] = field(default_factory=list)
    obstacle: PlanObject | None = None
    lift: float = 0.0  # How far a stacked-above heading must rise to clear the obstacle.
    rect: Rect | None = None  # The text block that hit the obstacle.


class _Layout:
    """Places text blocks, nudging objects slightly when a placement has no room."""

    def __init__(self, width: float, height: float, objects: list[PlanObject],
                 items: list[TextItem], grid: dict[str, Any] | None = None) -> None:
        self.W, self.H = width, height
        side = min(width, height)
        self.unit = side / 1080
        # Use the planner's layout grid when present, so text shares the assets'
        # margins and rhythm; older plans keep the previous proportions.
        self.margin = float(grid["margin"]) if grid and grid.get("margin") else 0.03 * side
        self.gap = 0.75 * float(grid["gutter"]) if grid and grid.get("gutter") else 0.02 * side
        header = (grid or {}).get("header")
        self.header = Rect(*header) if header and header[3] > 0 else None
        self.resizes = 0
        self.growth: dict[str, float] = {}  # Uniform growth applied to boxes of one asset.
        self.inside_targets = {label for item in items if item.placement.kind == "inside"
                               for label in item.placement.target_ids}
        self.bounds = Rect(self.margin / 2, self.margin / 2, width - self.margin, height - self.margin)
        self.objects = objects
        self.by_label = {o.label: o for o in objects}
        self.items = items
        self.by_id = {i.id: i for i in items}
        self.scale = {i.id: 1.0 for i in items}
        self.tolerate: dict[str, set[str]] = {i.id: set() for i in items}
        self.nudges = 0
        self.size_cap: dict[str, float] = {}  # Box text kept below the heading's size.
        self.lift = {i.id: 0.0 for i in items}  # Extra rise for 'above' text with stacked dependents.
        self.lifts = 0
        # Edge-to-edge decorations (bars, frames) stay put; screen-anchored text flows past them.
        self.frames = {o.label for o in objects if self._is_frame(o.base)}

    def _is_frame(self, r: Rect) -> bool:
        touches = r.x <= 2 or r.y <= 2 or r.right >= self.W - 2 or r.bottom >= self.H - 2
        return touches and (r.w >= 0.9 * self.W or r.h >= 0.9 * self.H)

    # -- driver
    def solve(self) -> list[_Placed]:
        """Settle the layout, then keep every box text smaller than the heading.

        The heading may shrink while settling (or box text may be styled large), so
        oversized box text is capped at heading / HEADING_TO_BOX_RATIO and the layout
        is settled again. Smaller box text only ever fits more easily, so this ends.
        """

        for _ in range(10):
            placed = self._settle()
            heading = next((p for p in placed if p.item.role == "heading"), None)
            if heading is None:
                return placed
            limit = heading.size / HEADING_TO_BOX_RATIO
            too_big = [p for p in placed if p.item.role == "box" and p.size > limit + 0.01]
            if not too_big:
                return placed
            for p in too_big:
                if p.item.style.size_px or p.item.style.scale > 1:
                    print(f"[Text] {p.item.label}: reduced from {p.size:.0f}px to {limit:.0f}px "
                          "so the heading stays bigger than box text.")
                self.size_cap[p.item.id] = min(self.size_cap.get(p.item.id, limit), limit)
        return placed

    def _settle(self) -> list[_Placed]:
        for _ in range(1000):
            placed, conflict = self._place_all()
            if conflict is None:
                return placed
            if not self._resolve(conflict):
                break
        raise TextLayerError("The text layout could not be settled; try fewer or shorter fields.")

    def _resolve(self, conflict: _Conflict) -> bool:
        item = conflict.item
        # Text stacked under an 'above <object>' heading (e.g. its subheading) that runs into
        # that object: raise the heading so both fit. If the heading then leaves the canvas,
        # the usual bounds conflict pushes the object down instead.
        root = item
        while root.placement.target_kind == "text" and root.placement.target_ids[0] in self.by_id:
            root = self.by_id[root.placement.target_ids[0]]
        if (root is not item and conflict.obstacle is not None and conflict.lift > 0.5
                and root.placement.kind == "above" and self.lifts < 40
                and conflict.obstacle.label in root.placement.target_ids
                and "bounds" not in self.tolerate[root.id]):
            self.lift[root.id] += conflict.lift
            self.lifts += 1
            return True
        if conflict.push and self.nudges < MAX_NUDGES:
            for index, (dx, dy) in enumerate(conflict.moves):
                # Only the preferred direction (or an edge overflow) may move partway;
                # partial moves in several directions would just jitter objects around.
                partial = conflict.key == "bounds" or index == 0
                if self._nudge(conflict.push, dx, dy, partial=partial):
                    self.nudges += 1
                    return True
        # Resize objects: boxes grow (all of one kind together) once their text has
        # become noticeably smaller; an illustration in the way shrinks a little.
        if (item.placement.kind == "inside" and conflict.key == "bounds"
                and self.scale[item.id] <= GROW_BOXES_BELOW_SCALE
                and self._grow_boxes(item.placement.target_ids)):
            return True
        if conflict.obstacle is not None and conflict.rect is not None and self._shrink_object(
                conflict.obstacle, conflict.rect):
            return True
        # Shrink this field together with the text fields it is stacked on
        # (a subheading below a heading shrinks both, keeping their proportion).
        chain, current = [item], item
        while current.placement.target_kind == "text" and current.placement.target_ids[0] in self.by_id:
            current = self.by_id[current.placement.target_ids[0]]
            chain.append(current)
        shrinkable = [i for i in chain
                      if self.scale[i.id] > MIN_SCALE + 1e-6 and self._size(i) > MIN_FONT_SIZE]
        if shrinkable:
            for member in shrinkable:
                self.scale[member.id] = max(MIN_SCALE, self.scale[member.id] * 0.92)
            return True
        if conflict.key not in self.tolerate[item.id]:
            print(f"[Text] Warning: {item.label}: {conflict.reason}; "
                  "placing it as close to the request as possible.")
            self.tolerate[item.id].add(conflict.key)
            return True
        return False

    def _size(self, item: TextItem) -> float:
        base = item.style.size_px or ROLE_BASE_SIZE[item.role] * self.unit * item.style.scale
        size = max(MIN_FONT_SIZE if item.style.size_px is None else 8.0, base * self.scale[item.id])
        if item.id in self.size_cap:  # Stay below the heading (renderer minimum is 8px).
            size = min(size, max(8.0, self.size_cap[item.id]))
        return size

    # -- placing
    def _place_all(self) -> tuple[list[_Placed], _Conflict | None]:
        placed: dict[str, _Placed] = {}
        for item in self.items:
            result = self._place(item, placed)
            if isinstance(result, _Conflict):
                return list(placed.values()), result
            placed[item.id] = result
        return list(placed.values()), None

    def _measure(self, item: TextItem, width: float, size: float) -> tuple[list[str], float, float]:
        at_minimum = self.scale[item.id] <= MIN_SCALE + 1e-6
        lines = _wrap(item.text, max(width, 1.0), size, item.weight, break_words=at_minimum,
                      width_factor=item.width_factor, tracking=item.tracking)
        block_w = max((_text_width(line, size, item.weight, item.width_factor, item.tracking)
                       for line in lines), default=0.0)
        return lines, block_w, len(lines) * size * item.line_height

    @staticmethod
    def _align_x(column: Rect, block_w: float, align: str) -> float:
        if align == "left":
            return column.x
        if align == "right":
            return column.right - block_w
        return column.x + (column.w - block_w) / 2

    def _place(self, item: TextItem, placed: dict[str, _Placed]) -> _Placed | _Conflict:
        pl, W, H, m, gap = item.placement, self.W, self.H, self.margin, self.gap
        size = self._size(item)
        bounds = self.bounds
        seeds: list[PlanObject] = []
        exclude: set[str] = set()
        rotation = 0.0
        flow = 1

        if pl.kind == "inside":
            seeds = [self.by_label[label] for label in pl.target_ids]
            target = _union([o.rect for o in seeds])
            pad = max(6 * self.unit, 0.08 * min(target.w, target.h))
            inner = Rect(target.x + pad, target.y + pad, target.w - 2 * pad, target.h - 2 * pad)
            # Assets sitting inside this box (e.g. an icon at flex-start) take their
            # share of the row; the text uses the space beside them.
            children = [o for o in self.objects if o.label not in pl.target_ids
                        and o.label not in self.frames and _contains(target, o.rect)]
            for child in children:
                c = child.rect
                if c.x + c.w / 2 < target.x + target.w / 2:
                    new_left = max(inner.x, c.right + self.gap)
                    inner = Rect(new_left, inner.y, inner.right - new_left, inner.h)
                else:
                    new_right = min(inner.right, c.x - self.gap)
                    inner = Rect(inner.x, inner.y, new_right - inner.x, inner.h)
            if inner.w < 10 or inner.h < 10:
                return _Conflict(item, f"{'/'.join(pl.target_ids)} is too small for text", "bounds")
            lines, bw, bh = self._measure(item, inner.w, size)
            align = item.style.align or pl.align_hint or "center"
            x = self._align_x(inner, bw, align)
            y = (inner.y if pl.vertical == "top" else inner.bottom - bh if pl.vertical == "bottom"
                 else inner.y + (inner.h - bh) / 2)
            column, bounds, flow = inner, inner, -1 if pl.vertical == "bottom" else 1
            exclude = set(pl.target_ids) | {child.label for child in children}
            rotation = seeds[0].rotation if len(seeds) == 1 else 0.0
            seeds = []  # Text that does not fit inside a box is shrunk, not moved.

        elif pl.kind in ("above", "below", "left", "right"):
            if pl.target_kind == "text":
                reference = placed[pl.target_ids[0]]
                target, source_column = reference.rect, reference.column
            else:
                seeds = [self.by_label[label] for label in pl.target_ids]
                target, source_column = _union([o.rect for o in seeds]), None

            if pl.kind in ("above", "below"):
                if source_column is not None:
                    column = Rect(source_column.x, 0, source_column.w, 0)
                else:
                    want = target.w if target.w >= 0.35 * W else min(W - 2 * m, max(target.w, 0.6 * W))
                    column_x = min(max(target.x + target.w / 2 - want / 2, m), W - m - want)
                    column = Rect(column_x, 0, want, 0)
                lines, bw, bh = self._measure(item, column.w, size)
                inherited = placed[pl.target_ids[0]].align if pl.target_kind == "text" else "center"
                align = item.style.align or pl.align_hint or inherited
                x = min(max(self._align_x(column, bw, align), m), max(m, W - m - bw))
                y = (target.y - gap - bh - self.lift[item.id] if pl.kind == "above"
                     else target.bottom + gap)
                flow = -1 if pl.kind == "above" else 1
            else:
                if pl.kind == "right":
                    start = target.right + gap
                    column = Rect(start, 0, max(W - m - start, 1.0), 0)
                else:
                    end = target.x - gap
                    column = Rect(m, 0, max(end - m, 1.0), 0)
                lines, bw, bh = self._measure(item, column.w, size)
                align = item.style.align or pl.align_hint or ("left" if pl.kind == "right" else "right")
                if bw <= column.w:
                    x = self._align_x(column, bw, align)
                else:  # Too wide: extend away from the target so the overflow is detected.
                    x = column.x if pl.kind == "right" else column.right - bw
                y = (target.y if pl.vertical == "top" else target.bottom - bh if pl.vertical == "bottom"
                     else target.y + (target.h - bh) / 2)
            column = Rect(column.x, y, column.w, bh)

        elif pl.kind == "anchor":
            full = W - 2 * m
            column_w = full if pl.horizontal == "center" else full / 2 - gap / 2
            column_x = W - m - column_w if pl.horizontal == "right" else m
            column = Rect(column_x, 0, column_w, 0)
            lines, bw, bh = self._measure(item, column.w, size)
            align = item.style.align or pl.horizontal
            x = self._align_x(column, bw, align)
            # 'top' text sits in the planner's header band (reserved for the title).
            top = self.header.y if self.header else m
            y = top if pl.vertical == "top" else H - m - bh if pl.vertical == "bottom" else (H - bh) / 2
            flow = -1 if pl.vertical == "bottom" else 1
            column = Rect(column.x, y, column.w, bh)

        else:  # xy
            column = Rect(pl.x, pl.y, pl.w or max(W - m - pl.x, 20.0), 0)
            lines, bw, bh = self._measure(item, column.w, size)
            align = item.style.align or "left"
            x, y = self._align_x(column, bw, align), pl.y
            column = Rect(column.x, y, column.w, bh)

        rect = Rect(x, y, bw, bh)

        # Stack after earlier text (and past objects this item is allowed to slide by).
        obstacles = [p.rect for p in placed.values() if f"text:{p.item.id}" not in self.tolerate[item.id]]
        if pl.kind in ("anchor", "xy"):
            obstacles += [self.by_label[label].rect for label in self.frames]
        for _ in range(len(obstacles) + 1):
            hit = next((r for r in obstacles if rect.overlaps(r)), None)
            if hit is None:
                break
            step = gap * 0.6
            rect = rect.moved(dy=(hit.bottom + step - rect.y) if flow > 0 else (hit.y - step - rect.bottom))

        result = _Placed(item, rect, column, lines, size, align, rotation)

        # 1) Not enough room: move the object the text is attached to.
        over_left, over_right = bounds.x - rect.x, rect.right - bounds.right
        over_top, over_bottom = bounds.y - rect.y, rect.bottom - bounds.bottom
        if max(over_left, over_right, over_top, over_bottom) > 0.5 and "bounds" not in self.tolerate[item.id]:
            moves = []
            if over_top > 0.5:
                moves.append((0.0, over_top))
            if over_bottom > 0.5:
                moves.append((0.0, -over_bottom))
            if over_left > 0.5:
                moves.append((over_left, 0.0))
            if over_right > 0.5:
                moves.append((-over_right, 0.0))
            return _Conflict(item, "not enough room for the requested placement", "bounds",
                             push=seeds, moves=moves)

        # 2) Overlapping an object: move that object away by the smallest amount.
        for obj in self.objects:
            key = f"obj:{obj.label}"
            if (obj.label in exclude or key in self.tolerate[item.id]
                    or not rect.overlaps(obj.rect)):
                continue
            if obj.label in self.frames:
                return _Conflict(item, f"it overlaps {obj.label}", key)
            o = obj.rect
            options = [(0.0, rect.bottom + gap - o.y), (0.0, -(o.bottom - (rect.y - gap))),
                       (rect.right + gap - o.x, 0.0), (-(o.right - (rect.x - gap)), 0.0)]
            options.sort(key=lambda move: abs(move[0]) + abs(move[1]))
            return _Conflict(item, f"it overlaps {obj.label}", key, lift=rect.bottom + gap - o.y,
                             push=[obj], moves=options, obstacle=obj, rect=rect)

        # 3) Still on top of other text after stacking.
        for other in placed.values():
            key = f"text:{other.item.id}"
            if key not in self.tolerate[item.id] and rect.overlaps(other.rect):
                return _Conflict(item, f"it overlaps {other.item.label}", key)
        return result

    # -- moving objects
    def _group(self, seeds: list[PlanObject], dx: float, dy: float) -> list[PlanObject]:
        """Seeds, the objects stacked beyond them, and same-asset siblings in their row or
        column, so planned spacing and alignment survive the move."""

        def span(a0: float, a1: float, b0: float, b1: float) -> bool:
            return min(a1, b1) - max(a0, b0) > 0.5

        members = {o.label: o for o in seeds}
        reach = abs(dy or dx) + self.gap  # Unrelated objects further away than this stay put.
        changed = True
        while changed:
            changed = False
            for seed in list(members.values()):
                s = seed.rect
                for obj in self.objects:
                    if obj.label in members or obj.label in self.frames:
                        continue
                    r = obj.rect
                    sibling = obj.asset == seed.asset
                    if dy:
                        clearance = r.y - s.bottom if dy > 0 else s.y - r.bottom
                        beyond = span(r.x, r.right, s.x, s.right) and (
                            r.y >= s.y - 0.5 if dy > 0 else r.bottom <= s.bottom + 0.5
                        ) and (sibling or clearance < reach)
                        joined = beyond or (sibling and span(r.y, r.bottom, s.y, s.bottom))
                    else:
                        clearance = r.x - s.right if dx > 0 else s.x - r.right
                        beyond = span(r.y, r.bottom, s.y, s.bottom) and (
                            r.x >= s.x - 0.5 if dx > 0 else r.right <= s.right + 0.5
                        ) and (sibling or clearance < reach)
                        joined = beyond or (sibling and span(r.x, r.right, s.x, s.right))
                    if joined:
                        members[obj.label] = obj
                        changed = True
        return list(members.values())

    def _fits(self, changed: dict[str, Rect]) -> bool:
        """New rects stay on the canvas and create no new overlaps between objects."""

        for label, rect in changed.items():
            if rect.x < -0.5 or rect.y < -0.5 or rect.right > self.W + 0.5 or rect.bottom > self.H + 0.5:
                return False
            for other in self.objects:
                if other.label == label or other.label in self.frames:
                    continue
                other_rect = changed.get(other.label, other.rect)
                if rect.overlaps(other_rect) and not self.by_label[label].rect.overlaps(other.rect):
                    return False
        return True

    def _grow_boxes(self, target_labels: list[str]) -> bool:
        """Grow the target box, and every box of the same kind equally, around their centres.

        Infographic cards stay identical in size; if they cannot all grow, none do.
        """

        asset = self.by_label[target_labels[0]].asset
        growth = self.growth.get(asset, 1.0)
        if growth >= 1 + MAX_BOX_GROWTH - 1e-6 or self.resizes >= 40:
            return False
        new_growth = min(growth + 0.05, 1 + MAX_BOX_GROWTH)
        siblings = [o for o in self.objects if o.asset == asset and o.label not in self.frames]
        changed: dict[str, Rect] = {}
        for obj in siblings:
            current = obj.rect
            width, height = obj.base.w * new_growth, obj.base.h * new_growth
            changed[obj.label] = Rect(current.x + current.w / 2 - width / 2,
                                      current.y + current.h / 2 - height / 2, width, height)
        if not self._fits(changed):
            return False
        for obj in siblings:
            obj.set_rect(changed[obj.label])
        self.growth[asset] = new_growth
        self.resizes += 1
        return True

    def _shrink_object(self, obj: PlanObject, text: Rect) -> bool:
        """Shrink an illustration (never a text box) just enough to clear a text block.

        Proportions are kept and the side away from the text stays in place.
        """

        if (obj.label in self.frames or obj.label in self.inside_targets
                or obj.label.startswith("box-") or self.resizes >= 40):
            return False
        r = obj.rect
        minimum = MIN_OBJECT_SCALE * obj.base.h / r.h  # Relative to the planner's size.
        text_above = text.y + text.h / 2 < r.y + r.h / 2
        vertical_need = (text.bottom + self.gap - r.y) if text_above else (r.bottom - (text.y - self.gap))
        text_left = text.x + text.w / 2 < r.x + r.w / 2
        horizontal_need = (text.right + self.gap - r.x) if text_left else (r.right - (text.x - self.gap))
        factor = max(1 - vertical_need / r.h, 1 - horizontal_need / r.w)
        factor = max(factor, minimum)
        if factor >= 0.995:
            return False
        width, height = r.w * factor, r.h * factor
        if 1 - vertical_need / r.h >= 1 - horizontal_need / r.w:  # Shrink away vertically.
            y = r.bottom - height if text_above else r.y
            x = r.x + (r.w - width) / 2
        else:
            x = r.right - width if text_left else r.x
            y = r.y + (r.h - height) / 2
        obj.set_rect(Rect(x, y, width, height))
        self.resizes += 1
        return True

    def _nudge(self, seeds: list[PlanObject], dx: float, dy: float, *, partial: bool) -> bool:
        if any(o.label in self.frames for o in seeds):
            return False
        group = self._group(seeds, dx, dy)
        vertical = abs(dy) >= abs(dx)
        amount = abs(dy if vertical else dx)
        sign = 1.0 if (dy if vertical else dx) > 0 else -1.0
        limit = MAX_SHIFT_FRACTION * (self.H if vertical else self.W)
        for obj in group:
            shifted = obj.dy if vertical else obj.dx
            r = obj.rect
            room = ((self.H - r.bottom if sign > 0 else r.y) if vertical
                    else (self.W - r.right if sign > 0 else r.x))
            amount = min(amount, limit - sign * shifted, room)
        requested = abs(dy if vertical else dx)
        if amount < 0.5 or (not partial and amount < requested - 0.5):
            return False
        move_x, move_y = (0.0, sign * amount) if vertical else (sign * amount, 0.0)
        labels = {o.label for o in group}
        for obj in group:
            new_rect = obj.rect.moved(move_x, move_y)
            for other in self.objects:
                if (other.label not in labels and new_rect.overlaps(other.rect)
                        and not obj.rect.overlaps(other.rect)):
                    return False
        for obj in group:
            obj.dx += move_x
            obj.dy += move_y
        return True


# --------------------------------------------------------------------------- spec -> plan


def _build_items(spec: dict[str, Any], targets: _Targets) -> list[TextItem]:
    try:
        default_color = parse_color(str(spec.get("default_color") or "black"))
    except ValueError as exc:
        raise TextLayerError(f"default_color: {exc}") from exc
    raw_items = spec.get("items")
    if not isinstance(raw_items, list):
        raise TextLayerError("The text spec needs an 'items' list.")

    # Fonts are optional: specs saved before fonts existed keep the old Arial look.
    uses_fonts = bool(spec.get("default_font")) or any(
        isinstance(raw, dict) and raw.get("font") for raw in raw_items)
    fonts = load_font_dictionary() if uses_fonts else {}
    default_font: str | None = None
    if fonts and spec.get("default_font"):
        try:
            default_font = parse_font(str(spec["default_font"]), fonts)
        except ValueError as exc:
            raise TextLayerError(f"default_font: {exc}") from exc
    items: list[TextItem] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise TextLayerError(f"items[{index}] must be an object.")
        role = raw.get("role", "text")
        if role not in ROLE_BASE_SIZE:
            raise TextLayerError(f"items[{index}].role must be one of {', '.join(ROLE_BASE_SIZE)}.")
        identifier = str(raw.get("id") or f"text-{index + 1}")
        label = str(raw.get("label") or identifier)
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        generate = bool(re.match(r"ai\s*:", content, re.I))
        own_box = raw.get("box") if role == "box" else None
        try:
            placement = parse_placement(str(raw.get("placement") or ""), targets, own_box=own_box)
            color_raw = str(raw.get("color") or "").strip()
            color = parse_color(color_raw) if color_raw and color_raw.lower() != "default" else default_color
            style = parse_style(str(raw.get("style") or ""))
            font_raw = str(raw.get("font") or "").strip()
            font_key = (parse_font(font_raw, fonts)
                        if fonts and font_raw and font_raw.lower() != "default" else default_font)
        except ValueError as exc:
            raise TextLayerError(f"{label}: {exc}") from exc
        items.append(TextItem(identifier, role, label,
                              re.sub(r"^ai\s*:\s*", "", content, flags=re.I) if generate else content,
                              generate, placement, color, style,
                              font_key=font_key, font=fonts.get(font_key) if font_key else None))
        targets.text_ids.append(identifier)
    return items


NAMED_RGB = {"white": "#FFFFFF", "black": "#000000", "navy": "#000080", "red": "#FF0000",
             "blue": "#0000FF", "green": "#008000", "yellow": "#FFFF00", "gray": "#808080",
             "grey": "#808080", "orange": "#FFA500", "purple": "#800080", "pink": "#FFC0CB",
             "gold": "#FFD700", "silver": "#C0C0C0", "darkgreen": "#006400", "lime": "#00FF00"}


def _luminance(color: Any) -> float | None:
    value = NAMED_RGB.get(str(color or "").strip().lower(), str(color or "").strip()).lstrip("#")
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return None
    red, green, blue = (int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _ensure_readable(items: list[TextItem], plan: dict[str, Any], objects: list[PlanObject]) -> None:
    """Text inside a box in (almost) the box's own colour would vanish; switch it."""

    fills: dict[str, list[Any]] = {}
    for element in plan.get("elements", []):
        style = element.get("style") or {} if isinstance(element, dict) else {}
        gradient = style.get("gradient") or {}
        fills[str(element.get("id"))] = list(gradient.get("colors") or []) or [style.get("fill", "#FFFFFF")]
    by_label = {o.label: o for o in objects}
    for item in items:
        if item.placement.kind != "inside":
            continue
        text_luminance = _luminance(item.color)
        backgrounds = [_luminance(color) for label in item.placement.target_ids
                       for color in fills.get(by_label[label].id, [])]
        backgrounds = [value for value in backgrounds if value is not None]
        if text_luminance is None or not backgrounds:
            continue
        average = sum(backgrounds) / len(backgrounds)
        if abs(text_luminance - average) < 0.3:
            replacement = "#FFFFFF" if average < 0.5 else "#1A1A1A"
            print(f"[Text] Warning: {item.label} colour {item.color} would be unreadable on its box; "
                  f"using {replacement} instead.")
            item.color = replacement


def apply_text_spec(plan_path: Path, spec: dict[str, Any], *, original_prompt: str,
                    endpoint: str, model: str, timeout: float, interactive: bool = True,
                    reuse_text: bool = False) -> list[dict[str, Any]]:
    """Resolve each field as its own chunk, lay them out, and update the plan in place."""

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        canvas_w, canvas_h = float(plan["canvas"]["width"]), float(plan["canvas"]["height"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TextLayerError(f"Cannot read the planned image: {exc}") from exc

    # Re-running the text stage starts from the planner's original object positions.
    layout_meta = plan.setdefault("metadata", {}).get("text_layout") or {}
    originals = layout_meta.get("original_positions") or {}
    for element in plan.get("elements", []):
        if isinstance(element, dict) and str(element.get("id")) in originals:
            saved = originals[str(element["id"])]  # [x, y] (older plans) or [x, y, w, h]
            element["x"], element["y"] = saved[0], saved[1]
            if len(saved) == 4:
                element["width"], element["height"] = saved[2], saved[3]

    objects = _plan_objects(plan)
    targets = _Targets(objects)
    items = _build_items(spec, targets)
    if not items:
        raise TextLayerError("No text fields were filled in.")

    # Resolve content chunk by chunk, in the order the user entered it.
    previous = {str(raw.get("id")): raw.get("resolved_text") for raw in spec.get("items", [])
                if isinstance(raw, dict)} if reuse_text else {}
    writer = _ChunkWriter(endpoint=endpoint, model=model, timeout=timeout,
                          original_prompt=original_prompt, interactive=interactive)
    context: list[tuple[str, str]] = []
    resolved: list[TextItem] = []
    try:
        for item in items:
            if not item.generate:
                text: str | None = _clean_text(item.content)
            elif isinstance(previous.get(item.id), str) and previous[item.id].strip():
                text = previous[item.id]
                print(f"[Text] Chunk '{item.label}': reusing saved text.")
            else:
                text = writer.write(item, item.content, context)
            if not text:
                continue
            item.text = _apply_case(text, item.case)
            print(f"[Text] Chunk '{item.label}': {item.text!r}")
            context.append((item.label, item.text))
            resolved.append(item)
    finally:
        if writer.used:
            unload_ollama_model(endpoint, model)

    # Fields placed relative to a skipped field fall back to a sensible spot.
    kept = {i.id for i in resolved}
    for item in resolved:
        pl = item.placement
        if pl.target_kind == "text" and pl.target_ids[0] not in kept:
            print(f"[Text] '{item.label}' was relative to a skipped field; placing it at the top.")
            item.placement = Placement("anchor", "top", vertical="top", horizontal="center")
    if not resolved:
        raise TextLayerError("No text was produced for any field.")
    _ensure_readable(resolved, plan, objects)

    placed = _Layout(canvas_w, canvas_h, objects, resolved, grid=plan.get("layout_grid")).solve()

    # Write object moves and resizes back to the plan.
    adjustments = []
    elements = plan.get("elements", [])
    for obj in objects:
        if max(abs(obj.dx), abs(obj.dy), abs(obj.dw), abs(obj.dh)) < 0.01:
            continue
        element = elements[obj.index]
        base = obj.base
        originals[obj.id] = [base.x, base.y, base.w, base.h]
        r = obj.rect
        element["x"], element["y"] = round(r.x, 2), round(r.y, 2)
        element["width"], element["height"] = round(r.w, 2), round(r.h, 2)
        adjustments.append({"id": obj.id, "label": obj.label, "dx": round(obj.dx, 2),
                            "dy": round(obj.dy, 2), "dw": round(obj.dw, 2), "dh": round(obj.dh, 2)})
        if abs(obj.dw) >= 0.5 or abs(obj.dh) >= 0.5:
            print(f"[Text] Resized {obj.label} from {base.w:.0f}x{base.h:.0f} to "
                  f"{r.w:.0f}x{r.h:.0f}px (at {r.x:.0f},{r.y:.0f}) to make room for text.")
        else:
            print(f"[Text] Moved {obj.label} by ({obj.dx:+.0f}px, {obj.dy:+.0f}px) to make room for text.")

    overlays = []
    for p in placed:
        width = min(p.rect.w, canvas_w)
        height = min(p.rect.h, canvas_h)
        overlays.append({
            "id": p.item.id,
            "role": p.item.role,
            "text": p.item.text,
            "lines": p.lines,
            "x": round(min(max(p.rect.x, 0.0), canvas_w - width), 2),
            "y": round(min(max(p.rect.y, 0.0), canvas_h - height), 2),
            "width": round(width, 2),
            "height": round(height, 2),
            "font_size": round(p.size, 2),
            "color": p.item.color,
            "align": p.align,
            "vertical_align": "top",
            "rotation": p.rotation,
            "weight": p.item.weight,
            **_font_overlay_fields(p.item, p.size),
            "placement": p.item.placement.raw,
        })

    saved_spec = json.loads(json.dumps(spec))
    texts = {i.id: i.text for i in resolved}
    for raw in saved_spec.get("items", []):
        if isinstance(raw, dict) and raw.get("id") in texts:
            raw["resolved_text"] = texts[raw["id"]]
    plan["text_overlays"] = overlays
    plan["metadata"]["text_request"] = saved_spec
    plan["metadata"]["text_layout"] = {"original_positions": originals,
                                       "object_adjustments": adjustments}
    temporary = plan_path.with_name(plan_path.name + ".tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(plan_path)
    spec_path = plan_path.with_name("text_spec.json")
    spec_path.write_text(json.dumps(saved_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[Text] Placed {len(overlays)} text block(s). Fields saved to {spec_path.name}.")
    return overlays


# --------------------------------------------------------------------------- interactive input


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or (default or "")


def _ask_valid(prompt: str, default: str, check: Callable[[str], Any]) -> str:
    while True:
        answer = _ask(prompt, default)
        try:
            check(answer)
            return answer
        except ValueError as exc:
            print(f"  ! {exc}")
            if not sys.stdin.isatty():
                raise TextLayerError(str(exc)) from exc


def collect_text_spec(plan: dict[str, Any]) -> dict[str, Any]:
    """Ask for each text field separately: content, placement, colour, and style."""

    objects = _plan_objects(plan)
    targets = _Targets(objects)
    boxes = targets.boxes
    print("\n[Text] Objects you can place text around:")
    for obj in objects:
        r = obj.rect
        print(f"    {obj.label:<14} {obj.asset:<18} x={r.x:.0f} y={r.y:.0f} {r.w:.0f}x{r.h:.0f}")
    if len(boxes) > 1:
        print(f"    {'boxes':<14} all {len(boxes)} boxes as one group")
    print("[Text] Placement: top | top-left | bottom-right | center | right of boxes | below box-2 |\n"
          "       above boxes | inside box-1 | top of box-1 | below heading | x,y | x,y,width")
    print("[Text] Content: type the exact words, or start with 'ai:' to have Ollama write it\n"
          "       (e.g. 'ai: a sales figure bigger than box-1'). Use | for a line break.")
    print("[Text] Style (optional): upper/lower/title/as-typed, bold/semibold/normal, "
          "small/medium/large/xl or 40px, left/center/right, shadow")
    fonts = load_font_dictionary()
    if fonts:
        print("[Text] Fonts (type the number or the name):")
        for number, (name, font) in enumerate(fonts.items(), 1):
            print(f"    {number:>2}. {name:<18} {font['usage']}")
    print("[Text] Press Enter to accept a [default] or skip a field.\n")

    default_color = _ask_valid("Default text colour", "black", parse_color)
    default_font = (_ask_valid("Default font", next(iter(fonts)), lambda raw: parse_font(raw, fonts))
                    if fonts else "")
    if default_font:
        default_font = parse_font(default_font, fonts)
    items: list[dict[str, Any]] = []
    chosen_fonts: dict[str, str] = {}

    def field_(identifier: str, role: str, label: str, placement_default: str,
               style_default: str = "", own_box: str | None = None, content: str | None = None,
               font_default: str | None = None) -> bool:
        if content is None:
            content = _ask(f"{label} (blank to skip)")
        if not content:
            return False
        placement = _ask_valid(f"  {label} placement", placement_default,
                               lambda raw: parse_placement(raw, targets, own_box=own_box))
        color = _ask_valid(f"  {label} colour", "default",
                           lambda raw: raw.lower() == "default" or parse_color(raw))
        font = ""
        if fonts:
            font = _ask_valid(f"  {label} font", font_default or default_font,
                              lambda raw: raw.lower() == "default" or parse_font(raw, fonts))
            font = default_font if font.lower() == "default" else parse_font(font, fonts)
            chosen_fonts[identifier] = font
        style = _ask_valid(f"  {label} style", style_default or "none", parse_style)
        entry = {"id": identifier, "role": role, "label": label, "content": content,
                 "placement": placement, "color": color,
                 "style": "" if style == "none" else style}
        if font:
            entry["font"] = font
        if own_box:
            entry["box"] = own_box
        items.append(entry)
        targets.text_ids.append(identifier)
        return True

    # With fonts, the font's own weights apply unless a style word overrides them.
    has_heading = field_("heading", "heading", "Heading", "top", "" if fonts else "bold")
    field_("subheading", "subheading", "Subheading",
           "below heading" if has_heading else "top",
           font_default=chosen_fonts.get("heading"))  # Matches the heading by default.
    for box in boxes:
        field_(f"{box.label}-text", "box", f"Text in {box.label}", "center", own_box=box.label)
    number = 1
    while True:
        content = _ask(f"Additional text #{number} (blank to finish)")
        if not content:
            break
        field_(f"text-{number}", "text", f"Text {number}",
               "below boxes" if len(boxes) > 1 else "below box-1" if boxes else "bottom",
               content=content)
        number += 1
    spec: dict[str, Any] = {"default_color": default_color, "items": items}
    if default_font:
        spec["default_font"] = default_font
    return spec


def offer_text_layer(plan_path: Path, *, original_prompt: str, endpoint: str,
                     model: str, timeout: float) -> bool:
    """Ask Y/N after planning, collect each text field, and update the plan."""

    while True:
        choice = _ask("Add text to this image? [Y/N]").lower() or "n"
        if choice in ("y", "yes", "n", "no"):
            break
        print("[Text] Please enter Y or N.")
    if choice in ("n", "no"):
        print("[Text] Skipped; rendering the planned image.")
        return False
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextLayerError(f"Cannot read the planned image: {exc}") from exc
    spec = collect_text_spec(plan)
    if not spec["items"]:
        print("[Text] No text fields were filled in; rendering without text.")
        return False
    apply_text_spec(plan_path, spec, original_prompt=original_prompt,
                    endpoint=endpoint, model=model, timeout=timeout)
    return True


# --------------------------------------------------------------------------- standalone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add structured text to an existing template plan.")
    parser.add_argument("plan", nargs="?", type=Path,
                        help="template_plan.json (default: newest plan under gen_assets).")
    parser.add_argument("--spec", type=Path, help="Text fields JSON (e.g. a saved text_spec.json).")
    parser.add_argument("--reuse-text", action="store_true",
                        help="Reuse resolved_text saved in --spec instead of asking Ollama again.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--no-render", action="store_true", help="Update the plan only.")
    parser.add_argument("--png", action="store_true", help="Also render a PNG (needs CairoSVG).")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        print("ERROR: --timeout must be greater than zero.", file=sys.stderr)
        return 2
    try:
        from template_image_renderer import RendererError, find_latest_plan, render_template

        plan_path = (args.plan or find_latest_plan(DEFAULT_OUTPUT_ROOT)).expanduser().resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        original_prompt = str(plan.get("metadata", {}).get("user_prompt") or "")
        if args.spec:
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
            interactive = False
        else:
            spec = collect_text_spec(plan)
            interactive = True
        apply_text_spec(plan_path, spec, original_prompt=original_prompt, endpoint=args.ollama_url,
                        model=args.model, timeout=args.timeout, interactive=interactive,
                        reuse_text=args.reuse_text)
        if not args.no_render:
            png = plan_path.parent / "generated_image.png" if args.png else None
            svg_path, png_path = render_template(plan_path, png_path=png)
            print(f"[Renderer] Image saved: {svg_path}")
            if png_path:
                print(f"[Renderer] PNG saved: {png_path}")
        return 0
    except (TextLayerError, RendererError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
