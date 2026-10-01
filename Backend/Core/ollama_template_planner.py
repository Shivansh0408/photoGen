#!/usr/bin/env python3
"""Create a deterministic image-template plan with a local Ollama model.

This module uses Ollama only as a JSON layout/styling planner.  It never asks
Ollama (or any other model) to generate pixels.  Every visual element must map
to an SVG already present in ``Schema/asset_dictonary/svg_dictionary.py``.

For every run the planner creates::

    gen_assets/<template-name>_<timestamp>/
        assets/                 # styled copies of approved dictionary SVGs
        preprocessed_prompt.json # ordered, binding interpretation of the prompt
        template_plan.json      # validated placement/render instructions
        generated_image.svg     # produced by template_image_renderer.py

Run without a positional prompt to enter the prompt interactively::

    python Backend/Core/ollama_template_planner.py

Or provide it directly::

    python Backend/Core/ollama_template_planner.py \
        "Light blue background with three boxes aligned in the center"
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_FILE = (
    PROJECT_ROOT / "Schema" / "asset_dictonary" / "svg_dictionary.py"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "gen_assets"
DEFAULT_MODEL = (
    os.getenv("OLLAMA_MODEL")
    or os.getenv("LLAMA_MODEL")
    or "llama3.2:3b"
)
DEFAULT_OLLAMA_URL = (
    os.getenv("OLLAMA_API_URL")
    or os.getenv("LLAMA_API_URL")
    or "http://localhost:11434/api/generate"
)
DEFAULT_MAX_OUTPUT_TOKENS = 2048

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NAMESPACE)
ET.register_namespace("xlink", XLINK_NAMESPACE)

DRAWABLE_TAGS = {
    "circle",
    "ellipse",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "text",
}
MAX_CANVAS_SIZE = 16384.0
MAX_ELEMENTS = 100

# These values are used only to preserve colors that the user explicitly
# writes in the prompt. Ollama may still understand other safe CSS colors, but
# it is never allowed to override a color captured here.
NAMED_COLOR_HEX = {
    "light green": "#90EE90",
    "dark green": "#006400",
    "lime green": "#32CD32",
    "mint green": "#98FF98",
    "forest green": "#228B22",
    "olive green": "#808000",
    "light blue": "#ADD8E6",
    "dark blue": "#00008B",
    "sky blue": "#87CEEB",
    "navy blue": "#000080",
    "royal blue": "#4169E1",
    "baby blue": "#89CFF0",
    "light yellow": "#FFFFE0",
    "dark yellow": "#B8860B",
    "light pink": "#FFB6C1",
    "hot pink": "#FF69B4",
    "light purple": "#D8BFD8",
    "dark purple": "#301934",
    "light gray": "#D3D3D3",
    "light grey": "#D3D3D3",
    "dark gray": "#A9A9A9",
    "dark grey": "#A9A9A9",
    "off white": "#FAF9F6",
    "cream": "#FFFDD0",
    "beige": "#F5F5DC",
    "ivory": "#FFFFF0",
    "white": "#FFFFFF",
    "black": "#000000",
    "red": "#FF0000",
    "green": "#008000",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "orange": "#FFA500",
    "purple": "#800080",
    "pink": "#FFC0CB",
    "brown": "#A52A2A",
    "gray": "#808080",
    "grey": "#808080",
    "teal": "#008080",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "gold": "#FFD700",
    "silver": "#C0C0C0",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

VALID_LAYOUTS = {"vertical_stack", "horizontal_row", "grid", "overlap", "freeform"}
VALID_HORIZONTAL_ALIGNMENTS = {"left", "center", "right"}
VALID_VERTICAL_ALIGNMENTS = {"top", "center", "bottom"}


class PlannerError(RuntimeError):
    """Raised when a plan cannot be created safely."""


def _local_tag(tag: str) -> str:
    """Return an XML tag without its namespace."""

    return tag.rsplit("}", 1)[-1].lower()


def _format_number(value: float) -> str:
    """Format a finite number without unnecessary trailing zeros."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _finite_number(
    value: Any,
    *,
    field: str,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Convert a value to a bounded finite float."""

    if value is None and default is not None:
        number = float(default)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PlannerError(f"{field} must be a number.") from exc

    if not math.isfinite(number):
        raise PlannerError(f"{field} must be finite.")
    if minimum is not None and number < minimum:
        raise PlannerError(f"{field} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise PlannerError(f"{field} must be no more than {maximum}.")
    return number


def _boolean(value: Any, default: bool = False) -> bool:
    """Normalize common JSON/string boolean forms."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


def _safe_color(value: Any, *, field: str, default: str | None = None) -> str | None:
    """Allow ordinary SVG colors while rejecting CSS functions/URLs."""

    if value is None or str(value).strip() == "":
        return default

    color = str(value).strip()
    lowered = color.lower()
    if lowered in {"none", "transparent", "currentcolor"}:
        return lowered

    if re.fullmatch(
        r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})",
        color,
    ):
        return color

    if re.fullmatch(r"[A-Za-z]+", color):
        return color

    functional_color = re.fullmatch(
        r"(?i)(?:rgb|rgba|hsl|hsla)\([0-9.,%+\-\s]+\)",
        color,
    )
    if functional_color:
        return color

    raise PlannerError(
        f"{field} contains an unsupported color. Use a hex, named, rgb, or hsl color."
    )


def _slugify(value: str, fallback: str = "template") -> str:
    """Create a filesystem-safe, readable slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return (slug[:64] or fallback).rstrip("-")


def _safe_identifier(value: Any, fallback: str) -> str:
    """Create a stable identifier for an element."""

    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "")).strip("-")
    return text[:80] or fallback


def _is_single_color_svg(xml_text: str) -> bool:
    """True for flat shapes (boxes, bars) that can take one new fill safely.

    Multi-colour illustrations would be flattened into a silhouette by a single fill.
    """

    fills = {
        value.lower().replace(" ", "")
        for value in re.findall(
            r"fill\s*[:=]\s*[\"']?\s*(#[0-9a-fA-F]{3,8}|rgba?\([^)]*\)|[a-zA-Z]+)",
            xml_text,
        )
    } - {"none", "transparent", "currentcolor"}
    return len(fills) <= 1


def load_svg_dictionary(schema_file: Path) -> dict[str, dict[str, Any]]:
    """Read ``SVG_DICTIONARY`` with AST parsing instead of executing the file."""

    schema_file = schema_file.expanduser().resolve()
    if not schema_file.is_file():
        raise PlannerError(f"SVG dictionary was not found: {schema_file}")

    try:
        tree = ast.parse(schema_file.read_text(encoding="utf-8"), filename=str(schema_file))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise PlannerError(f"Could not read the SVG dictionary: {exc}") from exc

    dictionary_value: Any = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "SVG_DICTIONARY" for target in node.targets):
            try:
                dictionary_value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise PlannerError("SVG_DICTIONARY must contain literal Python data.") from exc
            break

    if not isinstance(dictionary_value, dict) or not dictionary_value:
        raise PlannerError("SVG_DICTIONARY is missing or empty.")

    validated: dict[str, dict[str, Any]] = {}
    for filename, asset in dictionary_value.items():
        if not isinstance(filename, str) or not filename.lower().endswith(".svg"):
            continue
        if not isinstance(asset, dict):
            continue
        xml_text = asset.get("xml")
        dimensions = asset.get("dimensions")
        if not isinstance(xml_text, str) or not xml_text.strip():
            continue
        if not isinstance(dimensions, dict):
            continue

        width = dimensions.get("width")
        height = dimensions.get("height")
        try:
            width_value = float(width)
            height_value = float(height)
        except (TypeError, ValueError):
            continue
        if width_value <= 0 or height_value <= 0:
            continue

        validated[filename] = {
            "name": str(asset.get("name") or Path(filename).stem),
            "usage": str(asset.get("usage") or "Reusable SVG asset"),
            "dimensions": {
                "width": width_value,
                "height": height_value,
                "unit": str(dimensions.get("unit") or "px"),
            },
            "xml": xml_text.strip(),
            "recolorable": _is_single_color_svg(xml_text),
        }

    if not validated:
        raise PlannerError("No usable SVG entries were found in SVG_DICTIONARY.")
    return validated


def build_asset_catalog(assets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Create the compact asset metadata sent to Ollama."""

    catalog: list[dict[str, Any]] = []
    for filename, asset in assets.items():
        catalog.append(
            {
                "filename": filename,
                "name": asset["name"],
                "usage": asset["usage"],
                "width": asset["dimensions"]["width"],
                "height": asset["dimensions"]["height"],
                "recolorable": bool(asset.get("recolorable")),
            }
        )
    return catalog


def _default_canvas_size(assets: dict[str, dict[str, Any]]) -> tuple[float, float]:
    """Prefer a dictionary asset described as a background for canvas size."""

    for asset in assets.values():
        description = f"{asset['name']} {asset['usage']}".lower()
        if "background" in description:
            return (
                float(asset["dimensions"]["width"]),
                float(asset["dimensions"]["height"]),
            )
    first = next(iter(assets.values()))
    return (
        float(first["dimensions"]["width"]),
        float(first["dimensions"]["height"]),
    )


def _asset_search_text(filename: str, asset: dict[str, Any]) -> str:
    """Return normalized text used to match a prompt noun to an approved asset."""

    stem = Path(filename).stem
    spaced_stem = re.sub(r"(?<!^)(?=[A-Z])", " ", stem)
    return " ".join(
        (
            filename,
            stem,
            spaced_stem,
            str(asset.get("name") or ""),
            str(asset.get("usage") or ""),
        )
    ).lower()


def _asset_name_words(filename: str, asset: dict[str, Any]) -> set[str]:
    """Whole words of an asset's filename, name and usage ('InternalBOX' -> internal, box).

    Whole words keep 'creditcard' from counting as a card and 'boxer' as a box.
    """

    text = " ".join((Path(filename).stem, str(asset.get("name") or ""), str(asset.get("usage") or "")))
    words = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", text)
    return {word.lower() for word in words} | {word.lower()[:-1] for word in words
                                                if len(word) > 3 and word.lower().endswith("s")}


def _find_asset_by_keywords(
    assets: dict[str, dict[str, Any]],
    keywords: tuple[str, ...],
    *,
    excluded: set[str] | None = None,
) -> str | None:
    """Select the best approved asset for a semantic role.

    Keywords must appear as whole words. On a tie the plain single-colour shape wins
    (InternalBOX over a card illustration, PNGBG over a scenic background).
    """

    excluded = excluded or set()
    ranked: list[tuple[int, bool, str]] = []
    for filename, asset in assets.items():
        if filename in excluded:
            continue
        words = _asset_name_words(filename, asset)
        score = sum(1 for keyword in keywords if keyword in words)
        if score:
            ranked.append((score, bool(asset.get("recolorable")), filename))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], not item[1], item[2].lower()))
    return ranked[0][2]


def _extract_color_mentions(prompt: str) -> list[dict[str, Any]]:
    """Locate explicit safe color phrases and preserve their source positions."""

    lowered = prompt.lower()
    mentions: list[dict[str, Any]] = []

    literal_pattern = re.compile(
        r"#[0-9a-fA-F]{3,8}|(?:rgb|rgba|hsl|hsla)\([0-9.,%+\-\s]+\)",
        re.IGNORECASE,
    )
    for match in literal_pattern.finditer(prompt):
        try:
            safe_value = _safe_color(match.group(0), field="prompt color")
        except PlannerError:
            continue
        mentions.append(
            {
                "start": match.start(),
                "end": match.end(),
                "name": match.group(0),
                "value": safe_value,
            }
        )

    occupied = [(item["start"], item["end"]) for item in mentions]
    for color_name in sorted(NAMED_COLOR_HEX, key=len, reverse=True):
        color_pattern = re.compile(rf"\b{re.escape(color_name)}\b", re.IGNORECASE)
        for match in color_pattern.finditer(lowered):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            mentions.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "name": color_name,
                    "value": NAMED_COLOR_HEX[color_name],
                }
            )
            occupied.append((match.start(), match.end()))

    mentions.sort(key=lambda item: item["start"])
    return mentions


def _interval_distance(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> int:
    """Return the character distance between two source-text intervals."""

    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0


def _nearest_color_to_keywords(
    prompt: str,
    mentions: list[dict[str, Any]],
    keyword_pattern: str,
    *,
    excluded_ranges: set[tuple[int, int]] | None = None,
    maximum_distance: int = 100,
) -> dict[str, Any] | None:
    """Find the explicit color closest to a semantic target such as background."""

    keyword_matches = list(re.finditer(keyword_pattern, prompt, flags=re.IGNORECASE))
    if not keyword_matches:
        return None
    excluded_ranges = excluded_ranges or set()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for mention in mentions:
        mention_range = (int(mention["start"]), int(mention["end"]))
        if mention_range in excluded_ranges:
            continue
        for keyword_match in keyword_matches:
            distance = _interval_distance(
                mention_range[0],
                mention_range[1],
                keyword_match.start(),
                keyword_match.end(),
            )
            if distance <= maximum_distance:
                candidates.append((distance, mention_range[0], mention))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _parse_count_token(token: str) -> int | None:
    """Convert a numeric or English count token to an integer."""

    normalized = token.strip().lower()
    if normalized.isdigit():
        return int(normalized)
    return NUMBER_WORDS.get(normalized)


def _extract_box_count(prompt: str) -> int | None:
    """Extract only a count explicitly connected to the word box/boxes."""

    number_pattern = r"\d{1,3}|" + "|".join(NUMBER_WORDS)
    patterns = (
        rf"\b(?P<count>{number_pattern})\b(?:\s+[a-zA-Z][a-zA-Z-]*){{0,6}}\s+box(?:es)?\b",
        rf"\bbox(?:es)?\b\s*(?:count|quantity|number)?\s*(?:is|=|:|of)?\s*\b(?P<count>{number_pattern})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not match:
            continue
        count = _parse_count_token(match.group("count"))
        if count is not None and 1 <= count <= MAX_ELEMENTS:
            return count
    return None


def extract_hard_constraints(
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract explicit user facts that Ollama is not allowed to change."""

    lowered = user_prompt.lower()
    color_mentions = _extract_color_mentions(user_prompt)
    # Gradients first: their colours belong to their own target only.
    gradients = _extract_gradients(user_prompt, color_mentions)
    background_gradient = gradients.get("background")
    primary_gradient = gradients.get("primary")
    excluded_color_ranges: set[tuple[int, int]] = set()
    for gradient in gradients.values():
        excluded_color_ranges.update(gradient.pop("source_ranges"))
    # A background gradient already names the background colour; searching again
    # would steal a nearby box colour.
    background_mention = None if background_gradient else _nearest_color_to_keywords(
        user_prompt,
        color_mentions,
        r"\b(?:background|canvas)\b",
        excluded_ranges=excluded_color_ranges,
    )
    if background_mention is not None:
        excluded_color_ranges.add(
            (int(background_mention["start"]), int(background_mention["end"]))
        )
    # A colour written next to another asset ('a red border') is that asset's colour,
    # not the boxes' colour.
    box_spans = [m.span() for m in re.finditer(r"\b(?:box|boxes|card|cards|panel|panels)\b", lowered)]
    other_asset_words = set().union(*(
        _asset_words(filename, asset)
        for filename, asset in assets.items()
        if not _asset_name_words(filename, asset) & {"box", "card", "panel", "background", "canvas", "backdrop"}
    )) if assets else set()
    other_spans = [
        m.span() for m in re.finditer(r"[a-z]{3,}", lowered)
        if _word_root(m.group(0)) in other_asset_words
    ]
    for mention in color_mentions:
        start, end = int(mention["start"]), int(mention["end"])
        to_other = min((_interval_distance(start, end, a, b) for a, b in other_spans), default=10**9)
        to_box = min((_interval_distance(start, end, a, b) for a, b in box_spans), default=10**9)
        if to_other < to_box:
            excluded_color_ranges.add((start, end))
    box_mention = _nearest_color_to_keywords(
        user_prompt,
        color_mentions,
        r"\b(?:box|boxes|card|cards|panel|panels)\b",
        excluded_ranges=excluded_color_ranges,
    )

    if background_gradient is not None:
        background_mention = {"value": background_gradient["colors"][0]}
    if primary_gradient is not None:
        box_mention = {"value": primary_gradient["colors"][0]}

    background_requested = bool(
        re.search(r"\b(?:background|canvas|gradient)\b", lowered)
    )
    boxes_requested = bool(re.search(r"\b(?:box|boxes)\b", lowered))
    background_asset = (
        _find_asset_by_keywords(assets, ("background", "canvas", "backdrop"))
        if background_requested
        else None
    )
    primary_asset = (
        _find_asset_by_keywords(assets, ("box", "card", "panel"))
        if boxes_requested
        else None
    )
    primary_count = _extract_box_count(user_prompt) if boxes_requested else None

    # In this pipeline "on top of each other" means a centered vertical stack,
    # matching the user's expected design rather than an overlapping pile.
    if re.search(
        r"\b(?:on top of each other|one (?:under|below|above) (?:the )?other|"
        r"vertical(?:ly)?|vertical stack|stacked vertically)\b",
        lowered,
    ):
        layout = "vertical_stack"
    elif any(
        not re.search(r"\b(?:not|no|never|without|avoid|don't|doesn't|dont|nothing)\b[\w\s']{0,20}$",
                      lowered[:match.start()])
        for match in re.finditer(r"\b(?:overlap|overlapping|same position)\b", lowered)
    ):  # 'does not overlap' is a spacing wish, not an overlap layout.
        layout = "overlap"
    elif re.search(r"\b(?:grid|rows? and columns?|x\s*axis.*y\s*axis|y\s*axis.*x\s*axis)\b", lowered):
        layout = "grid"
    elif re.search(r"\b(?:side by side|horizontal(?:ly)?|horizontal row|x\s*axis)\b", lowered):
        layout = "horizontal_row"
    elif re.search(r"\b(?:y\s*axis)\b", lowered):
        layout = "vertical_stack"
    else:
        layout = None

    centered = bool(re.search(r"\b(?:center|centre|centered|centred)\b", lowered))
    horizontal_alignment: str | None = "center" if centered else None
    vertical_alignment: str | None = "center" if centered else None
    if re.search(r"\balign(?:ed)?\s+(?:to\s+the\s+)?left\b", lowered):
        horizontal_alignment = "left"
    elif re.search(r"\balign(?:ed)?\s+(?:to\s+the\s+)?right\b", lowered):
        horizontal_alignment = "right"
    if re.search(r"\balign(?:ed)?\s+(?:to\s+the\s+)?top\b", lowered):
        vertical_alignment = "top"
    elif re.search(r"\balign(?:ed)?\s+(?:to\s+the\s+)?bottom\b", lowered):
        vertical_alignment = "bottom"

    equal_spacing: bool | None = None
    if re.search(
        r"\b(?:equally|equal(?:ly)? spaced|equal spacing|equal(?:ly)? distances?|same distance|evenly spaced)\b",
        lowered,
    ):
        equal_spacing = True

    # Grid shape: '2 in each row' / '2 per row' / '2 columns' set the column count;
    # '4 in each column' / '4 rows' set rows (columns = count / rows).
    number = r"(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")"
    per = r"\s+(?:boxes\s+|cards\s+|panels\s+)?(?:in\s+(?:each|every|a)|per|on\s+each)\s+"
    grid_columns: int | None = None
    grid_rows: int | None = None
    if match := re.search(number + per + r"row\b", lowered):
        grid_columns = _parse_count_token(match.group(1))
    elif match := re.search(number + r"\s+col[ou]*m[ns]*s?\b", lowered):
        grid_columns = _parse_count_token(match.group(1))
    if match := re.search(number + per + r"col[ou]*m[ns]*\b", lowered):
        grid_rows = _parse_count_token(match.group(1))
    elif match := re.search(number + r"\s+rows\b", lowered):
        grid_rows = _parse_count_token(match.group(1))
    if grid_columns is None and grid_rows and primary_count:
        grid_columns = max(1, math.ceil(primary_count / grid_rows))
    if (grid_columns or grid_rows) and layout in (None, "freeform"):
        layout = "grid"

    # A margin stated for the boxes/grid ('the box grid has a margin of 10px').
    primary_margin: float | None = None
    for clause in re.split(r"[,.;\n]", lowered):
        if not re.search(BOX_WORDS + r"|\bgrid\b", clause):
            continue
        if margin := re.search(
            r"\bmargin\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*px\b|\b(\d+(?:\.\d+)?)\s*px\s+margin\b", clause
        ):
            primary_margin = float(margin.group(1) or margin.group(2))
            break

    # 'adjust the height and width so it fits': stretch boxes to fill their cells.
    primary_fill = bool(re.search(
        r"\b(?:adjust|fit|fits|fill|fills|stretch)\b[^,.;]{0,60}\b(?:width|height|hieght|size|fits?|fill|background|page|screen)\b"
        r"|\b(?:width|height|hieght|size)\b[^,.;]{0,40}\b(?:fit|fits|fill|adjust)\b",
        lowered,
    )) and primary_asset is not None

    # 'make sure the width is 90% of the background' / '90% width boxes'.
    width_fraction: float | None = None
    for clause in re.split(r"[,.;\n]", lowered):
        match = re.search(r"\bwidth\b[^%\d]{0,25}(\d{1,3}(?:\.\d+)?)\s*%", clause) or re.search(
            r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:of\s+the\s+\w+\s+)?(?:width|wide)\b", clause)
        if match and (re.search(BOX_WORDS, clause) or primary_asset):
            value = float(match.group(1))
            if 5 <= value <= 100:
                width_fraction = value / 100
                break

    # 'leave space at the top for a heading': always keep a title band.
    header_requested = bool(re.search(
        r"\b(?:space|room|gap|area)\b[^,.;]{0,30}\b(?:heading|title|header|headline)\b", lowered))
    # 'adjust the height of the background if necessary': the canvas may grow taller.
    canvas_can_grow = bool(re.search(
        r"\b(?:adjust|increase|extend|grow|change)\b[^,.;]{0,20}\b(?:height|hieght|length)\b"
        r"[^,.;]{0,15}\b(?:background|canvas|page|image|poster|screen)\b"
        r"|\b(?:taller|longer)\s+(?:background|canvas|page|image)\b", lowered))

    # Wide boxes can only stack: '5 boxes at 90% width' means one column.
    if layout in (None, "freeform") and width_fraction and width_fraction >= 0.6 and (primary_count or 0) > 1:
        layout = "vertical_stack"

    explicitly_named_assets: list[str] = []
    for filename, asset in assets.items():
        if filename in {background_asset, primary_asset}:
            continue
        searchable_names = {
            Path(filename).stem.lower(),
            re.sub(r"(?<!^)(?=[A-Z])", " ", Path(filename).stem).lower(),
            str(asset.get("name") or "").lower(),
        }
        if any(name and name in lowered for name in searchable_names):
            explicitly_named_assets.append(filename)

    return {
        "background_asset": background_asset,
        "background_color": background_mention["value"] if background_mention else None,
        "background_gradient": background_gradient,
        "primary_asset": primary_asset,
        "primary_count": primary_count,
        "primary_color": box_mention["value"] if box_mention else None,
        "primary_gradient": primary_gradient,
        "grid_columns": grid_columns,
        "primary_margin": primary_margin,
        "primary_fill": primary_fill or None,
        "primary_width_fraction": width_fraction,
        "header_requested": header_requested or None,
        "canvas_can_grow": canvas_can_grow or None,
        "layout": layout,
        "horizontal_alignment": horizontal_alignment,
        "vertical_alignment": vertical_alignment,
        "equal_spacing": equal_spacing,
        "additional_assets": explicitly_named_assets,
        "strict_asset_set": bool(primary_asset and primary_count is not None),
    }


def _derive_second_gradient_color(color: str) -> str:
    """Pick a lighter partner colour when the user names only one gradient colour."""

    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return "#FFFFFF"
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    amount = 0.6 if luminance > 0.6 else 0.4
    mixed = (round(channel + (255 - channel) * amount) for channel in (red, green, blue))
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


BACKGROUND_WORDS = r"\b(?:background|canvas|backdrop)\b"
BOX_WORDS = r"\b(?:box|boxes|card|cards|panel|panels)\b"


def _extract_gradients(
    prompt: str,
    mentions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Read every 'gradient' request and decide what it colours.

    Returns up to two entries: "background" ('light blue gradient background',
    'make the background blue to white gradient') and "primary" ('8 boxes in navy
    blue gradient colour'). Each is a two-colour linear gradient.
    """

    lowered = prompt.lower()
    found: dict[str, dict[str, Any]] = {}
    for keyword in re.finditer(r"\bgradient\b", lowered):
        clause_start = max(lowered.rfind(mark, 0, keyword.start()) for mark in ",.;\n") + 1
        clause_ends = [lowered.find(mark, keyword.end()) for mark in ",.;\n"]
        clause_end = min([value for value in clause_ends if value >= 0] or [len(prompt)])
        before, after = lowered[clause_start:keyword.start()], lowered[keyword.end():clause_end]

        # What does this gradient colour? The background when it names it right
        # beside the word; otherwise the boxes when the clause is about boxes.
        background_after = re.match(
            r"\s*(?:colou?r\s+|fill\s+)?(?:(?:for|on|in)\s+the\s+)?" + BACKGROUND_WORDS, after)
        background_before = re.search(BACKGROUND_WORDS + r"(?:(?!" + BOX_WORDS + r")[^,.;]){0,40}$", before)
        if background_after or background_before:
            target = "background"
        elif re.search(BOX_WORDS, before) or re.match(
                r"\s*(?:colou?r\s+|fill\s+)?(?:(?:for|on|in)\s+the\s+)?" + BOX_WORDS, after):
            target = "primary"
        else:
            target = "background"
        if target in found:
            continue

        other_subject = BOX_WORDS if target == "background" else BACKGROUND_WORDS

        def clear_between(start: int, end: int) -> bool:
            # The other subject must not sit between colour and 'gradient'
            # ('red boxes on a blue gradient background' keeps red for the boxes).
            return not re.search(other_subject, lowered[start:end])

        chosen = [
            mention for mention in mentions
            if clause_start <= int(mention["start"]) and int(mention["end"]) <= keyword.start()
            and keyword.start() - int(mention["end"]) <= 30
            and clear_between(int(mention["end"]), keyword.start())
        ][-2:]
        chosen += [
            mention for mention in mentions
            if keyword.end() <= int(mention["start"]) and int(mention["end"]) <= clause_end
            and int(mention["start"]) - keyword.end() <= 45
            and re.search(r"(?:\bfrom|\bto|\band|\binto|\bthrough|\bof|/|->|-)\s*$",
                          lowered[:int(mention["start"])])
        ]
        chosen = chosen[:2]
        if not chosen:
            continue

        clause = lowered[clause_start:clause_end]
        if re.search(r"\bdiagonal(?:ly)?\b", clause):
            direction = "diagonal"
        elif re.search(r"\bhorizontal(?:ly)?\b|\bleft\s+to\s+right\b|\bright\s+to\s+left\b|\bsideways\b", clause):
            direction = "horizontal"
        else:
            direction = "vertical"
        colors = [str(mention["value"]) for mention in chosen]
        if len(colors) == 1:
            colors.append(_derive_second_gradient_color(colors[0]))
        if re.search(r"\bright\s+to\s+left\b|\bbottom\s+to\s+top\b", clause):
            colors.reverse()
        found[target] = {
            "type": "linear",
            "colors": colors,
            "direction": direction,
            "source_ranges": {(int(m["start"]), int(m["end"])) for m in chosen},
        }
    return found


# Words that say nothing about what an asset depicts; they never select an asset.
GENERIC_ASSET_WORDS = {
    "a", "an", "and", "any", "as", "asset", "at", "be", "by", "case", "element", "for",
    "from", "hold", "holds", "icon", "image", "images", "img", "in", "internal", "is",
    "it", "of", "on", "or", "page", "png", "requests", "request", "requested", "svg",
    "that", "the", "this", "to", "use", "used", "user", "users", "using", "when", "with",
    "text", "top", "bottom", "left", "right", "center", "centre", "background", "picture",
    "place", "add", "put", "make", "create", "graphic",
}


def _word_root(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _asset_words(filename: str, asset: dict[str, Any]) -> set[str]:
    """Meaningful words from an asset's filename, name, and usage description."""

    raw = " ".join(
        (
            re.sub(r"(?<=[a-z])(?=[A-Z])", " ", Path(filename).stem),
            re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(asset.get("name") or "")),
            str(asset.get("usage") or ""),
        )
    ).lower()
    return {
        _word_root(word)
        for word in re.findall(r"[a-z]{3,}", raw)
        if word not in GENERIC_ASSET_WORDS
    }


def _match_assets_by_usage(
    prompt: str,
    assets: dict[str, dict[str, Any]],
    excluded: set[str],
) -> dict[str, int]:
    """Local usage matching: asset filename -> position of its first mention."""

    matches: dict[str, int] = {}
    prompt_words = [
        (match.start(), _word_root(match.group(0)))
        for match in re.finditer(r"[a-z]{3,}", prompt.lower())
        if match.group(0) not in GENERIC_ASSET_WORDS
    ]
    for filename, asset in assets.items():
        if filename in excluded:
            continue
        words = _asset_words(filename, asset)
        for position, word in prompt_words:
            if word in words:
                matches[filename] = position
                break
    return matches


CLAUSE_BREAK = re.compile(r"[,.;\n]|\b(?:and|with|plus|while|then)\b", re.IGNORECASE)


def _narrow_to_qualified_assets(
    prompt: str,
    assets: dict[str, dict[str, Any]],
    matches: dict[str, int],
) -> None:
    """'a full moon' means moonFull, not every moon: a qualifier beside the shared word
    narrows the match. Without one ('all the 5 moon assets') every match stays."""

    lowered = prompt.lower()
    groups: dict[int, list[str]] = {}
    for filename, position in matches.items():
        groups.setdefault(position, []).append(filename)
    for position, group in groups.items():
        if len(group) < 2 or position < 0:
            continue
        word = re.match(r"[a-z]+", lowered[position:])
        before = re.findall(r"[a-z]{3,}", lowered[max(0, position - 30):position])[-2:]
        after = re.findall(r"[a-z]{3,}", lowered[position + len(word.group(0) if word else ""):][:30])[:1]
        shared = _word_root(word.group(0)) if word else ""
        qualifiers = {_word_root(w) for w in before + after if w not in GENERIC_ASSET_WORDS}
        qualifiers.discard(shared)
        # Assets (from any group) described by both the shared word and a qualifier:
        # moonFull has 'moon' and 'full', but may have matched on 'full' first.
        subset = [f for f in list(matches) if shared in _asset_words(f, assets[f])
                  and qualifiers & _asset_words(f, assets[f])]
        if subset:
            for filename in group:
                if filename not in subset:
                    matches.pop(filename, None)


def _clause_around(prompt: str, position: int) -> str:
    """The phrase describing one asset: bounded by punctuation and joining words."""

    start = max((match.end() for match in CLAUSE_BREAK.finditer(prompt, 0, position)), default=0)
    following = CLAUSE_BREAK.search(prompt, position)
    end = following.start() if following else len(prompt)
    return prompt[start:end].lower()


def _container_from_clause(clause: str) -> dict[str, Any] | None:
    """'inside the box on the left', 'in each card at the start': assets placed in the boxes.

    Works like a CSS flex row inside each box: justify start (left), center or end
    (right), with the asset vertically centred.
    """

    inside = re.search(
        r"\b(?:inside|within|into|in)\s+(?:of\s+)?(?:the\s+|each\s+|every\s+|all\s+(?:the\s+)?|a\s+)?"
        r"(?:box|boxes|card|cards|panel|panels)\b", clause)
    if not inside:
        return None
    rest = clause[inside.start():]
    if re.search(r"\b(?:right|end|ending|finish)\b", rest):
        justify = "right"
    elif re.search(r"\b(?:center|centre|middle|centered|centred)\b", rest):
        justify = "center"
    else:  # 'left', 'start', 'starting', 'beginning' or unstated: flex-start.
        justify = "left"
    return {
        "container": "primary",
        "justify": justify,
        "per_box": "each" if re.search(r"\b(?:each|every)\s+(?:box|card|panel)", clause) else "spread",
        "vertical": "inside",
        "horizontal": justify,
    }


def _count_before(prompt: str, position: int) -> int | None:
    """A number written just before the asset's word: 'two gavels', '3 small borders'."""

    match = re.search(
        r"\b(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s+(?:[a-z]+\s+){0,2}$",
        prompt[max(0, position - 40):position].lower(),
    )
    count = _parse_count_token(match.group(1)) if match else None
    return count if count and 1 <= count <= 10 else None


def _placement_from_clause(clause: str) -> dict[str, Any]:
    """Explicit position words for one asset; missing keys were not stated."""

    placement: dict[str, Any] = {}
    if re.search(r"\b(?:top|upper|header|above)\b", clause):
        placement["vertical"] = "top"
    elif re.search(r"\b(?:bottom|lower|footer|below|beneath|under|underneath|base)\b", clause):
        placement["vertical"] = "bottom"
    if re.search(r"\bleft\b", clause):
        placement["horizontal"] = "left"
    elif re.search(r"\bright\b", clause):
        placement["horizontal"] = "right"
    elif re.search(r"\b(?:center|centre|centered|centred|middle)\b", clause):
        placement["horizontal"] = "center"
        if "vertical" not in placement and not re.search(r"\b[xy][\s-]*axis\b", clause):
            placement["vertical"] = "center"
    margin = re.search(
        r"\bmargin\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*px\b|\b(\d+(?:\.\d+)?)\s*px\s+(?:margin|from|away)\b",
        clause,
    )
    if margin:
        placement["margin"] = float(margin.group(1) or margin.group(2))
    return placement


def _primary_offset_from_prompt(
    prompt: str,
    assets: dict[str, dict[str, Any]],
    candidates: list[str],
) -> dict[str, Any] | None:
    """Read 'box 30px below the top border' style spacing to an extra asset."""

    pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*px\s+(below|under|beneath|underneath|above|over)\s+"
        r"(?:the\s+)?((?:[a-z]+\s*){1,3})",
        re.IGNORECASE,
    )
    for match in pattern.finditer(prompt):
        target_words = {
            _word_root(word)
            for word in re.findall(r"[a-z]{3,}", match.group(3).lower())
            if word not in GENERIC_ASSET_WORDS
        }
        for filename in candidates:
            if target_words & _asset_words(filename, assets[filename]):
                relation = "below" if match.group(2).lower() in {
                    "below", "under", "beneath", "underneath"
                } else "above"
                return {
                    "relation": relation,
                    "asset": filename,
                    "gap": float(match.group(1)),
                    "source": "prompt",
                }
    return None


def build_asset_selection_prompt(
    user_prompt: str,
    candidates: list[dict[str, Any]],
    handled: dict[str, Any],
    *,
    want_notes: bool = False,
) -> str:
    """Ask Ollama which extra assets the request needs, judged by their usage text."""

    notes_field = (
        ',\n  "design_notes": ["short instruction about layout, sizes or colours in this part"]'
        if want_notes
        else ""
    )
    notes_rule = (
        "\n5. This is one chunk of a longer request. design_notes lists every layout, size, colour "
        "and position instruction in this chunk as short plain sentences (at most 6)."
        if want_notes
        else ""
    )
    return f"""You choose which approved SVG assets a design request needs.
Decide using each asset's "usage" description and name. Synonyms count
(for example: gavel = judge's mallet, border = page border, chart = graph).

CANDIDATE ASSETS:
{json.dumps(candidates, indent=2)}

ALREADY HANDLED ELSEWHERE (never list these): {json.dumps(handled)}

DESIGN REQUEST:
<user_request>
{user_prompt}
</user_request>

Return ONLY one JSON object:
{{
  "assets": [
    {{"asset": "exact filename", "evidence": "exact words from the request that ask for it",
      "count": 1, "vertical": "top|center|bottom", "horizontal": "left|center|right", "margin": null}}
  ],
  "main_object_offset": {{"relation": "below|above|none", "asset": "exact filename or null", "gap": null}}{notes_field}
}}

Rules:
1. List an asset only when the request asks for something its usage describes; otherwise return "assets": [].
2. "evidence" must be copied word for word from the request.
3. vertical/horizontal say where the request puts that asset. "center of the Y-axis" means horizontally centered.
4. main_object_offset describes a main object placed a distance below/above one listed asset (e.g. "30px below the border").{notes_rule}
"""


def _validated_selection(
    raw: dict[str, Any],
    *,
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
    excluded: set[str],
    local_matches: dict[str, int],
    full_prompt: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Keep only Ollama picks backed by the request's own words.

    ``user_prompt`` is the text Ollama saw (a chunk for long prompts); positions are
    reported in ``full_prompt`` so placement words can be read around them.
    """

    lowered_prompt = user_prompt.lower()
    lowered_full = (full_prompt or user_prompt).lower()
    chosen: dict[str, dict[str, Any]] = {}
    entries = raw.get("assets") if isinstance(raw.get("assets"), list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            filename = resolve_asset_filename(entry.get("asset"), assets)
        except PlannerError:
            continue
        if filename in excluded or filename in chosen:
            continue
        evidence = str(entry.get("evidence") or "").strip().lower()
        evidence_words = [
            word for word in re.findall(r"[a-z]{4,}", evidence)
            if word not in GENERIC_ASSET_WORDS
        ]
        if filename not in local_matches and not (
            evidence and evidence in lowered_prompt and evidence_words
        ):
            print(f"[Preprocessor] Ignoring unsupported Ollama asset pick: {filename}")
            continue
        placement: dict[str, Any] = {}
        if str(entry.get("vertical", "")).lower() in VALID_VERTICAL_ALIGNMENTS:
            placement["vertical"] = str(entry["vertical"]).lower()
        if str(entry.get("horizontal", "")).lower() in VALID_HORIZONTAL_ALIGNMENTS:
            placement["horizontal"] = str(entry["horizontal"]).lower()
        try:
            if entry.get("margin") is not None:
                placement["margin"] = _finite_number(
                    entry["margin"], field="margin", minimum=0, maximum=2000
                )
            if entry.get("count") is not None:
                placement["count"] = int(
                    _finite_number(entry["count"], field="count", minimum=1, maximum=10)
                )
        except PlannerError:
            pass
        position = lowered_full.find(evidence) if evidence else -1
        placement["_position"] = position if position >= 0 else local_matches.get(filename, -1)
        chosen[filename] = placement

    offset = None
    raw_offset = raw.get("main_object_offset")
    if isinstance(raw_offset, dict) and str(raw_offset.get("relation", "")).lower() in {
        "below", "above"
    }:
        try:
            target = resolve_asset_filename(raw_offset.get("asset"), assets)
            gap = _finite_number(
                raw_offset.get("gap"), field="gap", default=24, minimum=0, maximum=2000
            )
            offset = {
                "relation": str(raw_offset["relation"]).lower(),
                "asset": target,
                "gap": gap,
                "source": "ollama",
            }
        except PlannerError:
            offset = None
    return chosen, offset


def select_additional_assets(
    *,
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
    hard_constraints: dict[str, Any],
    endpoint: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Find extra assets by their usage text (Ollama, with local matching as backup)."""

    excluded = {
        value
        for value in (hard_constraints.get("background_asset"), hard_constraints.get("primary_asset"))
        if value
    }
    canvas_width, canvas_height = _default_canvas_size(assets)
    local_matches = _match_assets_by_usage(user_prompt, assets, excluded)
    _narrow_to_qualified_assets(user_prompt, assets, local_matches)
    # Assets named outright ('moonFull') keep the position of their name, so the
    # words around it ('inside each box on the left') still place them.
    for filename in hard_constraints.get("additional_assets") or []:
        local_matches.setdefault(filename, user_prompt.lower().find(Path(filename).stem.lower()))

    candidates = [
        entry for entry in build_asset_catalog(assets) if entry["filename"] not in excluded
    ]
    handled = {
        "background": hard_constraints.get("background_asset"),
        "main_object": hard_constraints.get("primary_asset"),
    }

    # Long prompts (> LONG_PROMPT_WORDS) are split at commas/full stops and sent as
    # separate chunks; each chunk also returns short design notes for the layout pass.
    chunks = split_long_prompt(user_prompt)
    pieces = chunks or [user_prompt]
    if chunks:
        print(
            f"[Preprocessor] Long prompt ({_word_count(user_prompt)} words): split at commas "
            f"and full stops into {len(chunks)} chunk(s)."
        )
    ollama_choice: dict[str, dict[str, Any]] = {}
    ollama_offset: dict[str, Any] | None = None
    notes: list[str] = []
    ollama_down = False
    for index, piece in enumerate(pieces, start=1):
        if not candidates and not chunks:
            break
        label = f" (chunk {index}/{len(pieces)})" if chunks else ""
        if ollama_down:
            notes.append(_trim_words(piece, 50))
            continue
        print(f"[Preprocessor] Asking Ollama which assets the request needs{label}...")
        try:
            response = request_ollama(
                endpoint=endpoint,
                model=model,
                prompt=build_asset_selection_prompt(
                    piece, candidates, handled, want_notes=bool(chunks)
                ),
                timeout=timeout,
                temperature=0.0,
            )
            raw_selection = parse_json_object(response)
            choice, offset = _validated_selection(
                raw_selection,
                user_prompt=piece,
                full_prompt=user_prompt,
                assets=assets,
                excluded=excluded,
                local_matches=local_matches,
            )
            for filename, suggestion in choice.items():
                ollama_choice.setdefault(filename, suggestion)
            ollama_offset = ollama_offset or offset
            if chunks:
                notes.extend(_validated_notes(raw_selection) or [_trim_words(piece, 50)])
        except PlannerError as exc:
            print(
                f"[Preprocessor] Ollama asset selection unavailable{label} ({exc}); "
                "using local usage matching."
            )
            if "could not connect" in str(exc).lower():
                ollama_down = True  # Do not wait on every remaining chunk.
            if chunks:
                notes.append(_trim_words(piece, 50))

    selected = list(dict.fromkeys([*local_matches, *ollama_choice]))
    placements: list[dict[str, Any]] = []
    for filename in selected:
        dimensions = assets[filename]["dimensions"]
        full_width = (
            float(dimensions["width"]) >= 0.9 * canvas_width
            and float(dimensions["height"]) <= 0.2 * canvas_height
        )
        placement: dict[str, Any] = {
            "asset": filename,
            "count": 1,
            "vertical": "top" if full_width else "center",
            "horizontal": "center",
            "margin": None,
            "full_width": full_width,
            "source": "default",
        }
        suggestion = dict(ollama_choice.get(filename) or {})
        position = suggestion.pop("_position", -1)
        if suggestion:
            placement.update(suggestion)
            placement["source"] = "ollama"
        position = local_matches.get(filename, position)
        if position is not None and position >= 0:
            stated = _placement_from_clause(_clause_around(user_prompt, position))
            if "vertical" not in stated and "horizontal" not in stated:
                # 'a gavel and a chart at the bottom right': the gavel shares the
                # position words that follow it in the same sentence.
                ends = [user_prompt.find(mark, position) for mark in ",.;\n"]
                end = min([value for value in ends if value >= 0] or [len(user_prompt)])
                shared = _placement_from_clause(user_prompt[position:end].lower())
                stated.update({key: shared[key] for key in ("vertical", "horizontal") if key in shared})
            count = _count_before(user_prompt, position)
            if count:
                stated["count"] = count
            if stated:
                placement.update(stated)  # The user's own words win over Ollama.
                placement["source"] = "prompt"
            color = _color_for_asset_mention(user_prompt, position)
            if color:
                placement["color"] = color  # Binding: 'a red border'.
            container = _container_from_clause(
                user_prompt[position:min([v for v in (user_prompt.find(m, position) for m in ",.;\n")
                                          if v >= 0] or [len(user_prompt)])].lower())
            if container and hard_constraints.get("primary_asset"):
                placement.update(container)
                placement["full_width"] = False
            placement["_mention"] = position
        if full_width and not placement.get("container"):
            placement["horizontal"] = "center"
        placements.append(placement)

    # 'all the 5 moon assets': one word matched five different assets and the number
    # counts those assets, so each is used once (not five copies of each).
    by_mention: dict[int, list[dict[str, Any]]] = {}
    for placement in placements:
        by_mention.setdefault(placement.pop("_mention", -1), []).append(placement)
    for mention, group in by_mention.items():
        if mention >= 0 and len(group) > 1 and all(p.get("count") in (len(group), 1) for p in group) \
                and any(p.get("count") == len(group) for p in group):
            for placement in group:
                placement["count"] = 1

    offset = _primary_offset_from_prompt(user_prompt, assets, selected)
    if offset is None and ollama_offset and ollama_offset["asset"] in selected:
        offset = ollama_offset
    if not hard_constraints.get("primary_asset"):
        offset = None
    return {
        "additional_assets": selected,
        "asset_placements": placements,
        "primary_offset": offset,
        "prompt_chunks": len(chunks),
        "request_digest": "\n".join(f"- {note}" for note in notes) if chunks else None,
    }


LONG_PROMPT_WORDS = 300
CHUNK_WORDS = 80


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w'-]+", text))


def _trim_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit]) + (" ..." if len(words) > limit else "")


def split_long_prompt(prompt: str) -> list[str]:
    """Split prompts over LONG_PROMPT_WORDS at commas and full stops.

    Short prompts return [] and are processed whole. The pieces are regrouped into
    chunks of up to CHUNK_WORDS words so each Ollama call gets a meaningful unit;
    decimals such as '10.5px' are not split.
    """

    if _word_count(prompt) <= LONG_PROMPT_WORDS:
        return []
    segments = [segment.strip() for segment in re.split(r",|\.(?!\d)", prompt) if segment.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for segment in segments:
        words = _word_count(segment)
        if current and current_words + words > CHUNK_WORDS:
            chunks.append(", ".join(current))
            current, current_words = [], 0
        current.append(segment)
        current_words += words
    if current:
        chunks.append(", ".join(current))
    return chunks


def _validated_notes(raw: dict[str, Any]) -> list[str]:
    notes = raw.get("design_notes")
    if not isinstance(notes, list):
        return []
    cleaned = [
        _trim_words(re.sub(r"\s+", " ", note).strip(), 30)
        for note in notes
        if isinstance(note, str) and note.strip()
    ]
    return cleaned[:6]


def _color_for_asset_mention(prompt: str, position: int) -> str | None:
    """A colour written right next to an extra asset's word ('a red border')."""

    lowered = prompt.lower()
    word = re.match(r"[a-z]+", lowered[position:])
    word_end = position + (len(word.group(0)) if word else 1)
    clause_start = max(lowered.rfind(mark, 0, position) for mark in ",.;\n") + 1
    clause_ends = [lowered.find(mark, position) for mark in ",.;\n"]
    clause_end = min([value for value in clause_ends if value >= 0] or [len(prompt)])
    others = [
        match.span()
        for match in re.finditer(
            r"\b(?:background|canvas|gradient|box|boxes|card|cards|panel|panels)\b", lowered
        )
    ]
    best: tuple[int, str] | None = None
    for mention in _extract_color_mentions(prompt):
        start, end = int(mention["start"]), int(mention["end"])
        if start < clause_start or end > clause_end:
            continue
        to_asset = _interval_distance(start, end, position, word_end)
        to_other = min((_interval_distance(start, end, a, b) for a, b in others), default=10**9)
        if to_asset <= 25 and to_asset < to_other and (best is None or to_asset < best[0]):
            best = (to_asset, str(mention["value"]))
    return best[1] if best else None


def build_preprocessor_prompt(
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
    hard_constraints: dict[str, Any],
    validation_feedback: str | None = None,
) -> str:
    """Build the first-pass prompt that converts prose into ordered meaning."""

    feedback = ""
    if validation_feedback:
        feedback = f"\nVALIDATION FEEDBACK TO FIX:\n{validation_feedback}\n"
    return f"""You are the semantic preprocessor for a deterministic SVG template engine.

Your only task is to break the request into an exact specification. Do not
create coordinates or image elements. Do not invent a count, color, or asset.
Values in HARD_CONSTRAINTS were read directly from the user's words and are
binding; copy every non-null/non-empty value exactly.

APPROVED ASSETS:
{json.dumps(build_asset_catalog(assets), indent=2)}

HARD_CONSTRAINTS:
{json.dumps(hard_constraints, indent=2)}

USER REQUEST:
<user_request>
{user_prompt}
</user_request>

Return exactly one JSON object with this structure:
{{
  "background_asset": "approved filename or null",
  "background_color": "safe CSS color or null",
  "primary_asset": "approved filename or null",
  "primary_count": 3,
  "primary_color": "safe CSS color or null",
  "layout": "vertical_stack|horizontal_row|grid|overlap|freeform",
  "horizontal_alignment": "left|center|right",
  "vertical_alignment": "top|center|bottom",
  "equal_spacing": true,
  "spacing": null,
  "grid_columns": null,
  "additional_assets": [],
  "strict_asset_set": true,
  "interpretation": "short plain-English summary"
}}

Important rules:
1. If a number is not explicitly stated, primary_count must be null.
2. A number must never be inferred from words like x-axis, y-axis, centered, or grid.
3. Preserve explicit background and object colors exactly.
4. In this project, "on top of each other" means vertical_stack unless the user explicitly says overlap.
5. Use only approved filenames and do not return XML, Markdown, or commentary.
{feedback}
"""


def _optional_asset(
    value: Any,
    assets: dict[str, dict[str, Any]],
    *,
    field: str,
) -> str | None:
    """Resolve an optional asset value without accepting invented filenames."""

    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    try:
        return resolve_asset_filename(value, assets)
    except PlannerError as exc:
        raise PlannerError(f"{field}: {exc}") from exc


def normalize_preprocessed_spec(
    raw_spec: dict[str, Any],
    hard_constraints: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    *,
    user_prompt: str,
) -> dict[str, Any]:
    """Validate the semantic pass and reapply direct user constraints last."""

    if not isinstance(raw_spec, dict):
        raise PlannerError("The preprocessed specification must be a JSON object.")

    def hard_or_raw(field: str) -> Any:
        hard_value = hard_constraints.get(field)
        if hard_value is not None and hard_value != []:
            return hard_value
        return raw_spec.get(field)

    background_asset = _optional_asset(
        hard_or_raw("background_asset"),
        assets,
        field="background_asset",
    )
    primary_asset = _optional_asset(
        hard_or_raw("primary_asset"),
        assets,
        field="primary_asset",
    )

    background_color = _safe_color(
        hard_or_raw("background_color"),
        field="preprocessing.background_color",
        default="#FFFFFF",
    )
    primary_color = _safe_color(
        hard_or_raw("primary_color"),
        field="preprocessing.primary_color",
        default=None,
    )

    raw_count = hard_or_raw("primary_count")
    if raw_count is None or str(raw_count).strip().lower() in {"", "none", "null"}:
        if primary_asset is not None and re.search(r"\bboxes\b", user_prompt, re.IGNORECASE):
            primary_count = 3
            count_source = "plural default (no explicit count)"
        else:
            primary_count = 1 if primary_asset is not None else None
            count_source = (
                "singular default (no explicit count)"
                if primary_asset is not None
                else "not applicable"
            )
    else:
        primary_count = int(
            _finite_number(
                raw_count,
                field="preprocessing.primary_count",
                minimum=1,
                maximum=MAX_ELEMENTS,
            )
        )
        count_source = (
            "explicit user constraint"
            if hard_constraints.get("primary_count") is not None
            else "semantic extraction"
        )

    layout_value = str(hard_or_raw("layout") or "freeform").strip().lower()
    if layout_value not in VALID_LAYOUTS:
        raise PlannerError(f"Unsupported preprocessed layout: {layout_value}")
    horizontal_alignment = str(
        hard_or_raw("horizontal_alignment") or "center"
    ).strip().lower()
    if horizontal_alignment not in VALID_HORIZONTAL_ALIGNMENTS:
        raise PlannerError(
            f"Unsupported horizontal alignment: {horizontal_alignment}"
        )
    vertical_alignment = str(
        hard_or_raw("vertical_alignment") or "center"
    ).strip().lower()
    if vertical_alignment not in VALID_VERTICAL_ALIGNMENTS:
        raise PlannerError(f"Unsupported vertical alignment: {vertical_alignment}")

    equal_spacing_value = hard_constraints.get("equal_spacing")
    if equal_spacing_value is None:
        equal_spacing_value = raw_spec.get("equal_spacing")
    equal_spacing = _boolean(equal_spacing_value, default=False)

    spacing: float | None = None
    raw_spacing = raw_spec.get("spacing")
    if raw_spacing is not None and str(raw_spacing).strip().lower() not in {
        "",
        "null",
        "none",
    }:
        spacing = _finite_number(
            raw_spacing,
            field="preprocessing.spacing",
            minimum=0,
            maximum=MAX_CANVAS_SIZE,
        )

    grid_columns: int | None = None
    raw_grid_columns = hard_or_raw("grid_columns")
    if raw_grid_columns is not None and str(raw_grid_columns).strip().lower() not in {
        "",
        "null",
        "none",
    }:
        grid_columns = int(
            _finite_number(
                raw_grid_columns,
                field="preprocessing.grid_columns",
                minimum=1,
                maximum=MAX_ELEMENTS,
            )
        )

    additional_values: list[Any] = []
    if hard_constraints.get("strict_asset_set"):
        additional_sources = (hard_constraints.get("additional_assets") or [],)
    else:
        additional_sources = (
            raw_spec.get("additional_assets"),
            hard_constraints.get("additional_assets"),
        )
    for source in additional_sources:
        if isinstance(source, list):
            additional_values.extend(source)
    additional_assets: list[str] = []
    for value in additional_values:
        resolved = _optional_asset(value, assets, field="additional_assets")
        if resolved and resolved not in additional_assets:
            additional_assets.append(resolved)

    strict_asset_set = _boolean(
        hard_constraints.get("strict_asset_set"),
        default=_boolean(raw_spec.get("strict_asset_set"), default=False),
    )
    if primary_asset is not None and primary_count is not None:
        strict_asset_set = True
    interpretation = str(
        raw_spec.get("interpretation")
        or "The request was normalized into ordered template constraints."
    ).strip()[:1000]

    def clean_gradient(raw_gradient: Any, field: str) -> dict[str, Any] | None:
        if not isinstance(raw_gradient, dict) or len(raw_gradient.get("colors") or []) != 2:
            return None
        return {
            "type": "linear",
            "colors": [_safe_color(color, field=field) for color in raw_gradient["colors"]],
            "direction": raw_gradient.get("direction")
            if raw_gradient.get("direction") in {"vertical", "horizontal", "diagonal"}
            else "vertical",
        }

    background_gradient = clean_gradient(
        hard_constraints.get("background_gradient"), "preprocessing.background_gradient"
    )
    if background_gradient:
        background_color = background_gradient["colors"][0]
    primary_gradient = clean_gradient(
        hard_constraints.get("primary_gradient"), "preprocessing.primary_gradient"
    )
    if primary_gradient:
        primary_color = primary_gradient["colors"][0]
    primary_margin = hard_constraints.get("primary_margin")

    spec = {
        "background_asset": background_asset,
        "background_color": background_color,
        "background_gradient": background_gradient,
        "primary_asset": primary_asset,
        "primary_count": primary_count,
        "primary_count_source": count_source,
        "primary_color": primary_color,
        "primary_gradient": primary_gradient,
        "primary_margin": float(primary_margin) if primary_margin is not None else None,
        "primary_fill": bool(hard_constraints.get("primary_fill")),
        "primary_width_fraction": hard_constraints.get("primary_width_fraction"),
        "header_requested": bool(hard_constraints.get("header_requested")),
        "canvas_can_grow": bool(hard_constraints.get("canvas_can_grow")),
        "layout": layout_value,
        "horizontal_alignment": horizontal_alignment,
        "vertical_alignment": vertical_alignment,
        "equal_spacing": equal_spacing,
        "spacing": spacing,
        "grid_columns": grid_columns,
        "additional_assets": additional_assets,
        "asset_placements": [
            placement
            for placement in hard_constraints.get("asset_placements") or []
            if placement.get("asset") in additional_assets
        ],
        "primary_offset": hard_constraints.get("primary_offset"),
        "prompt_chunks": int(hard_constraints.get("prompt_chunks") or 0),
        "request_digest": hard_constraints.get("request_digest"),
        "strict_asset_set": strict_asset_set,
        "interpretation": interpretation,
        "original_prompt": user_prompt,
        "hard_constraints": {
            key: value
            for key, value in hard_constraints.items()
            if value is not None and value != []
        },
    }
    spec["ordered_steps"] = build_ordered_preprocessing_steps(spec)
    return spec


def build_ordered_preprocessing_steps(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Create the human-readable sequence shown before layout generation."""

    return [
        {
            "step": 1,
            "name": "background",
            "value": {
                "asset": spec.get("background_asset"),
                "color": spec.get("background_color"),
                "gradient": spec.get("background_gradient"),
            },
        },
        {
            "step": 2,
            "name": "object_count",
            "value": {
                "asset": spec.get("primary_asset"),
                "count": spec.get("primary_count"),
                "source": spec.get("primary_count_source"),
            },
        },
        {
            "step": 3,
            "name": "position",
            "value": {
                "layout": spec.get("layout"),
                "horizontal_alignment": spec.get("horizontal_alignment"),
                "vertical_alignment": spec.get("vertical_alignment"),
                "equal_spacing": spec.get("equal_spacing"),
                "spacing": spec.get("spacing"),
            },
        },
        {
            "step": 4,
            "name": "object_color",
            "value": spec.get("primary_color"),
        },
        {
            "step": 5,
            "name": "allowed_extra_assets",
            "value": spec.get("additional_assets") or [],
        },
        {
            "step": 6,
            "name": "extra_asset_placement",
            "value": {
                "placements": spec.get("asset_placements") or [],
                "primary_offset": spec.get("primary_offset"),
            },
        },
    ]


def print_preprocessed_steps(spec: dict[str, Any]) -> None:
    """Show the meaning that will be enforced before asking for coordinates."""

    print("[Preprocessor] Prompt converted into binding steps:")
    gradient = spec.get("background_gradient")
    background_fill = (
        f"linear gradient {gradient['colors'][0]} -> {gradient['colors'][1]} ({gradient['direction']})"
        if gradient
        else f"color={spec.get('background_color')}"
    )
    print(
        "  Step 1 - Background: "
        f"asset={spec.get('background_asset') or 'canvas'}, {background_fill}"
    )
    print(
        "  Step 2 - Objects: "
        f"count={spec.get('primary_count')}, "
        f"asset={spec.get('primary_asset') or 'not specified'}, "
        f"source={spec.get('primary_count_source')}"
    )
    print(
        "  Step 3 - Position: "
        f"layout={spec.get('layout')}, "
        f"horizontal={spec.get('horizontal_alignment')}, "
        f"vertical={spec.get('vertical_alignment')}, "
        f"equal_spacing={spec.get('equal_spacing')}"
        + (f", columns={spec['grid_columns']}" if spec.get("layout") == "grid" and spec.get("grid_columns") else "")
        + (f", margin={spec['primary_margin']:g}px" if spec.get("primary_margin") is not None else "")
        + (", stretch to fit" if spec.get("primary_fill") else "")
        + (f", width={spec['primary_width_fraction'] * 100:g}% of canvas" if spec.get("primary_width_fraction") else "")
        + (", space for heading" if spec.get("header_requested") else "")
        + (", canvas may grow taller" if spec.get("canvas_can_grow") else "")
    )
    primary_gradient = spec.get("primary_gradient")
    if primary_gradient:
        print(
            f"  Step 4 - Object color: linear gradient {primary_gradient['colors'][0]} -> "
            f"{primary_gradient['colors'][1]} ({primary_gradient['direction']})"
        )
    else:
        print(f"  Step 4 - Object color: {spec.get('primary_color') or 'original asset color'}")
    placements = spec.get("asset_placements") or []
    if placements:
        print("  Step 5 - Extra assets:")
        for placement in placements:
            width_note = ", full width" if placement.get("full_width") else ""
            where = (f"inside the boxes, justify {placement.get('justify')}"
                     f"{' (in every box)' if placement.get('per_box') == 'each' else ''}"
                     if placement.get("container")
                     else f"{placement.get('vertical')}/{placement.get('horizontal')}{width_note}")
            print(
                f"    - {placement['asset']} x{placement.get('count', 1)}: "
                f"{where} (from {placement.get('source')})"
            )
    else:
        print("  Step 5 - Extra assets: none")
    offset = spec.get("primary_offset")
    if offset:
        print(
            f"  Step 6 - Object offset: {offset['gap']:g}px {offset['relation']} "
            f"{offset['asset']} (from {offset.get('source')})"
        )


def preprocess_user_prompt(
    *,
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
    endpoint: str,
    model: str,
    timeout: float,
    temperature: float,
    attempts: int = 1,
) -> dict[str, Any]:
    """Read explicit constraints locally, then let Ollama pick extra assets by usage."""

    _ = temperature, attempts
    print("[Preprocessor] Reading prompt locally...")
    hard_constraints = extract_hard_constraints(user_prompt, assets)
    hard_constraints.update(
        select_additional_assets(
            user_prompt=user_prompt,
            assets=assets,
            hard_constraints=hard_constraints,
            endpoint=endpoint,
            model=model,
            timeout=timeout,
        )
    )
    spec = normalize_preprocessed_spec(
        {},
        hard_constraints,
        assets,
        user_prompt=user_prompt,
    )
    print_preprocessed_steps(spec)
    return spec


def build_ollama_prompt(
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
    semantic_spec: dict[str, Any],
    validation_feedback: str | None = None,
    previous_response: str | None = None,
) -> str:
    """Build a strict layout-planning prompt for a text-only Ollama model."""

    default_width, default_height = _default_canvas_size(assets)
    catalog_json = json.dumps(
        build_asset_catalog(assets),
        separators=(",", ":"),
    )
    binding_fields = (
        "background_asset",
        "background_color",
        "primary_asset",
        "primary_count",
        "primary_color",
        "layout",
        "horizontal_alignment",
        "vertical_alignment",
        "equal_spacing",
        "spacing",
        "grid_columns",
        "additional_assets",
        "asset_placements",
        "primary_offset",
        "background_gradient",
        "primary_gradient",
        "primary_margin",
        "primary_fill",
        "strict_asset_set",
    )
    compact_spec = {
        field: semantic_spec.get(field)
        for field in binding_fields
    }
    feedback_section = ""
    if validation_feedback:
        feedback_section = (
            "\nYour previous answer failed validation. Correct it using this exact feedback:\n"
            f"{validation_feedback}\n"
        )
    if previous_response:
        feedback_section += (
            "\nPrevious answer (do not repeat its mistakes):\n"
            f"{previous_response[:4000]}\n"
        )

    # Colours the user wrote are fixed; every other size and colour is Ollama's choice.
    hard = semantic_spec.get("hard_constraints") or {}
    fixed_colors = {
        "background": semantic_spec.get("background_gradient") or hard.get("background_color"),
        "primary": semantic_spec.get("primary_gradient") or semantic_spec.get("primary_color"),
        **{
            placement["asset"]: placement["color"]
            for placement in semantic_spec.get("asset_placements") or []
            if placement.get("color")
        },
    }
    fixed_colors = {key: value for key, value in fixed_colors.items() if value}

    design = _layout_system(default_width, default_height, semantic_spec)
    header_rule = (
        f" Leave the top {design['header_height']:g}px below any top border free for a title."
        if design["header_height"] else ""
    )

    digest = semantic_spec.get("request_digest")
    if digest:
        request_section = (
            f"The original request was long ({_word_count(user_prompt)} words). It was split at "
            f"commas and full stops and processed in {semantic_spec.get('prompt_chunks')} chunks. "
            "Chunk-by-chunk digest of every instruction:\n"
            f"<user_request_digest>\n{digest}\n</user_request_digest>"
        )
    else:
        request_section = f"<user_request>\n{user_prompt}\n</user_request>"

    return f"""You are a deterministic SVG template layout planner.

You do NOT generate images or invent new visual assets. You create one JSON
layout plan using only filenames from the approved asset catalog below. An
asset may be reused multiple times by adding multiple element objects. The
Python pipeline will copy and restyle the original SVG XML for each element.

APPROVED ASSET CATALOG:
{catalog_json}

CANONICAL PREPROCESSED SPECIFICATION (BINDING):
{json.dumps(compact_spec, separators=(",", ":"))}

USER-FIXED COLOURS (never change these): {json.dumps(fixed_colors)}

USER REQUEST:
{request_section}

Return exactly one JSON object and no Markdown. Use this structure:
{{
  "template_name": "short descriptive name",
  "description": "one sentence describing the layout",
  "canvas": {{
    "width": {default_width:g},
    "height": {default_height:g},
    "background_color": "#FFFFFF"
  }},
  "elements": [
    {{
      "id": "unique-element-id",
      "asset": "exact filename from the catalog",
      "x": 0,
      "y": 0,
      "width": 320,
      "height": 240,
      "z_index": 0,
      "rotation": 0,
      "preserve_aspect_ratio": false,
      "style": {{}}
    }}
  ]
}}

Rules:
1. Use pixel coordinates with origin (0, 0) at the top-left.
2. Use only exact asset filenames from the catalog. Never create a filename.
3. Treat the canonical preprocessed specification as authoritative.
4. List each asset ONCE (one element per distinct filename), sized as ONE copy should
   look. The pipeline creates the copies (primary_count boxes, extra asset counts) and
   their positions; do not repeat elements. Assets placed "inside the boxes" must be
   small enough to sit inside one box next to its text.
5. SIZES ARE YOUR DECISION. First look at every element the page needs (background,
   all primary copies, every additional asset with its placement) and choose width and
   height for each so the whole composition fits the canvas with comfortable margins,
   nothing overlaps, and the proportions look balanced. Resize any asset freely; the
   example numbers above are placeholders, never copy them.
6. COLOURS: keep USER-FIXED COLOURS exactly. For everything else you may choose a
   harmonious "fill" (and "stroke") that contrasts with the background, but only for
   assets whose catalog entry has "recolorable": true. Never give a style fill to an
   asset with "recolorable": false; it is a multi-colour illustration. "style" may hold
   fill, stroke, stroke_width, opacity, corner_radius; leave out anything you keep as is.
7. Calculate centered/aligned positions precisely from canvas and element dimensions.
8. Keep elements within the canvas unless the request explicitly asks for cropping.
9. z_index controls back-to-front order; backgrounds should use the lowest value.
10. Do not add text, shapes, URLs, code, XML, or assets outside this schema.
11. When strict_asset_set is true, use only background_asset, primary_asset, and additional_assets.
12. Counts, positions and fixed colours in the specification are enforced afterwards;
    your sizes and free colours are kept when they fit.
13. INFOGRAPHIC STYLE: design like a clean infographic. Keep outer margins of
    {design['margin']:g}px and gaps of {design['gutter']:g}px between elements, align edges to a
    12-column grid (column {design['column_width']:.0f}px), give repeated objects identical
    sizes, balance empty space evenly, and make the most important illustration
    the largest visual.{header_rule}
{feedback_section}
"""


def normalize_ollama_endpoint(url: str) -> str:
    """Accept either an Ollama base URL or its complete generate endpoint."""

    normalized = url.strip().rstrip("/")
    if not normalized:
        raise PlannerError("The Ollama URL cannot be empty.")
    if normalized.endswith("/api/generate"):
        return normalized
    return f"{normalized}/api/generate"


def request_ollama(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    timeout: float,
    temperature: float,
) -> str:
    """Call the local Ollama generate endpoint using only the standard library."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        # Stay loaded between the selection and layout calls (unloaded after planning),
        # so the second call does not pay the model load time again.
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "num_predict": DEFAULT_MAX_OUTPUT_TOKENS,
        },
    }
    request = urllib.request.Request(
        normalize_ollama_endpoint(endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise PlannerError(
            f"Ollama returned HTTP {exc.code}: {details[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise PlannerError(
                f"Ollama did not answer within {timeout:g} seconds."
            ) from exc
        raise PlannerError(
            "Could not connect to Ollama. Start Ollama and confirm the configured URL. "
            f"Details: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise PlannerError(
            f"Ollama did not answer within {timeout:g} seconds."
        ) from exc

    try:
        envelope = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise PlannerError("Ollama returned an invalid HTTP response body.") from exc

    generated_text = envelope.get("response")
    if not isinstance(generated_text, str) or not generated_text.strip():
        error_text = envelope.get("error")
        if error_text:
            raise PlannerError(f"Ollama error: {error_text}")
        raise PlannerError("Ollama returned no layout response.")
    return generated_text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from an Ollama response."""

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise PlannerError("Ollama did not return a valid JSON object.")


def resolve_asset_filename(
    requested: Any,
    assets: dict[str, dict[str, Any]],
) -> str:
    """Resolve exact/case-insensitive filenames or catalog display names."""

    if not isinstance(requested, str) or not requested.strip():
        raise PlannerError("Each element must include an asset filename.")

    requested_text = requested.strip()
    if requested_text in assets:
        return requested_text

    lowered = requested_text.lower()
    matches = [
        filename
        for filename, asset in assets.items()
        if lowered
        in {
            filename.lower(),
            Path(filename).stem.lower(),
            str(asset["name"]).lower(),
        }
    ]
    if len(matches) == 1:
        return matches[0]

    available = ", ".join(assets)
    raise PlannerError(
        f"Unknown asset '{requested_text}'. Allowed filenames: {available}."
    )


def normalize_style(raw_style: Any, *, element_id: str) -> dict[str, Any]:
    """Keep only styling values that can safely be applied to local SVG XML."""

    if raw_style is None:
        return {}
    if not isinstance(raw_style, dict):
        raise PlannerError(f"elements[{element_id}].style must be an object.")

    aliases = {
        "fill_color": "fill",
        "background_color": "fill",
        "stroke_color": "stroke",
        "border_color": "stroke",
        "border_width": "stroke_width",
        "border_radius": "corner_radius",
    }
    style: dict[str, Any] = {}
    for original_key, value in raw_style.items():
        key = aliases.get(str(original_key), str(original_key))
        if key == "fill":
            fill = _safe_color(
                value,
                field=f"elements[{element_id}].style.fill",
            )
            if fill is not None:
                style["fill"] = fill
        elif key == "stroke":
            stroke = _safe_color(
                value,
                field=f"elements[{element_id}].style.stroke",
            )
            if stroke is not None:
                style["stroke"] = stroke
        elif key == "stroke_width":
            style["stroke_width"] = _finite_number(
                value,
                field=f"elements[{element_id}].style.stroke_width",
                minimum=0,
                maximum=1000,
            )
        elif key == "opacity":
            style["opacity"] = _finite_number(
                value,
                field=f"elements[{element_id}].style.opacity",
                minimum=0,
                maximum=1,
            )
        elif key == "corner_radius":
            style["corner_radius"] = _finite_number(
                value,
                field=f"elements[{element_id}].style.corner_radius",
                minimum=0,
                maximum=MAX_CANVAS_SIZE,
            )
    return style


def normalize_plan(
    raw_plan: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    *,
    user_prompt: str,
    model: str,
    schema_file: Path,
) -> dict[str, Any]:
    """Validate and normalize all untrusted values returned by Ollama."""

    if not isinstance(raw_plan, dict):
        raise PlannerError("The layout plan must be a JSON object.")

    default_width, default_height = _default_canvas_size(assets)
    raw_canvas = raw_plan.get("canvas") or {}
    if not isinstance(raw_canvas, dict):
        raise PlannerError("canvas must be a JSON object.")

    canvas_width = _finite_number(
        raw_canvas.get("width"),
        field="canvas.width",
        default=default_width,
        minimum=1,
        maximum=MAX_CANVAS_SIZE,
    )
    canvas_height = _finite_number(
        raw_canvas.get("height"),
        field="canvas.height",
        default=default_height,
        minimum=1,
        maximum=MAX_CANVAS_SIZE,
    )
    canvas_color = _safe_color(
        raw_canvas.get("background_color"),
        field="canvas.background_color",
        default="#FFFFFF",
    )

    raw_elements = raw_plan.get("elements")
    if not isinstance(raw_elements, list) or not raw_elements:
        raise PlannerError("elements must be a non-empty JSON array.")
    if len(raw_elements) > MAX_ELEMENTS:
        raise PlannerError(f"A plan can contain at most {MAX_ELEMENTS} elements.")

    normalized_elements: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_element in enumerate(raw_elements):
        if not isinstance(raw_element, dict):
            raise PlannerError(f"elements[{index}] must be an object.")

        filename = resolve_asset_filename(raw_element.get("asset"), assets)
        source_dimensions = assets[filename]["dimensions"]
        element_id = _safe_identifier(
            raw_element.get("id"),
            f"{_slugify(Path(filename).stem)}-{index + 1}",
        )
        original_element_id = element_id
        duplicate_number = index + 1
        while element_id in used_ids:
            element_id = f"{original_element_id}-{duplicate_number}"
            duplicate_number += 1
        used_ids.add(element_id)

        width = _finite_number(
            raw_element.get("width"),
            field=f"elements[{index}].width",
            default=float(source_dimensions["width"]),
            minimum=0.1,
            maximum=MAX_CANVAS_SIZE * 4,
        )
        height = _finite_number(
            raw_element.get("height"),
            field=f"elements[{index}].height",
            default=float(source_dimensions["height"]),
            minimum=0.1,
            maximum=MAX_CANVAS_SIZE * 4,
        )
        x = _finite_number(
            raw_element.get("x"),
            field=f"elements[{index}].x",
            default=0,
            minimum=-MAX_CANVAS_SIZE * 4,
            maximum=MAX_CANVAS_SIZE * 4,
        )
        y = _finite_number(
            raw_element.get("y"),
            field=f"elements[{index}].y",
            default=0,
            minimum=-MAX_CANVAS_SIZE * 4,
            maximum=MAX_CANVAS_SIZE * 4,
        )
        z_index = int(
            _finite_number(
                raw_element.get("z_index"),
                field=f"elements[{index}].z_index",
                default=index,
                minimum=-10000,
                maximum=10000,
            )
        )
        rotation = _finite_number(
            raw_element.get("rotation"),
            field=f"elements[{index}].rotation",
            default=0,
            minimum=-36000,
            maximum=36000,
        )
        preserve_aspect_ratio = _boolean(
            raw_element.get("preserve_aspect_ratio"),
            default=False,
        )
        if "stretch" in raw_element:
            preserve_aspect_ratio = not _boolean(raw_element.get("stretch"), default=True)

        normalized_elements.append(
            {
                "id": element_id,
                "asset": filename,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "z_index": z_index,
                "rotation": rotation,
                "preserve_aspect_ratio": preserve_aspect_ratio,
                "style": normalize_style(raw_element.get("style"), element_id=element_id),
            }
        )

    template_name = str(raw_plan.get("template_name") or "generated-template").strip()
    description = str(raw_plan.get("description") or user_prompt).strip()
    return {
        "schema_version": "1.0",
        "template_name": template_name[:120],
        "description": description[:1000],
        "canvas": {
            "width": canvas_width,
            "height": canvas_height,
            "background_color": canvas_color,
        },
        "elements": normalized_elements,
        "metadata": {
            "user_prompt": user_prompt,
            "planner": "ollama",
            "model": model,
            "schema_file": str(schema_file.resolve()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def _new_plan_element(
    asset_filename: str,
    assets: dict[str, dict[str, Any]],
    *,
    element_id: str,
    z_index: int,
) -> dict[str, Any]:
    """Create a safe element from an approved asset's native dimensions."""

    dimensions = assets[asset_filename]["dimensions"]
    return {
        "id": element_id,
        "asset": asset_filename,
        "x": 0.0,
        "y": 0.0,
        "width": float(dimensions["width"]),
        "height": float(dimensions["height"]),
        "z_index": z_index,
        "rotation": 0.0,
        "preserve_aspect_ratio": False,
        "style": {},
    }


def _copy_plan_element(element: dict[str, Any]) -> dict[str, Any]:
    """Copy an element without sharing its mutable style dictionary."""

    copied = dict(element)
    copied["style"] = dict(element.get("style") or {})
    copied.pop("generated_asset", None)
    copied.pop("source_dimensions", None)
    return copied


def _aligned_start(container: float, content: float, alignment: str) -> float:
    """Calculate left/top origin for a requested edge or center alignment."""

    if alignment in {"left", "top"}:
        return 0.0
    if alignment in {"right", "bottom"}:
        return max(0.0, container - content)
    return max(0.0, (container - content) / 2.0)


def _fit_dimensions(
    width: float,
    height: float,
    maximum_width: float,
    maximum_height: float,
) -> tuple[float, float]:
    """Scale dimensions down proportionally to fit a positive region."""

    if width <= 0 or height <= 0:
        return 1.0, 1.0
    scale = min(1.0, maximum_width / width, maximum_height / height)
    return width * max(scale, 0.001), height * max(scale, 0.001)


def _position_primary_elements(
    elements: list[dict[str, Any]],
    spec: dict[str, Any],
    canvas: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    *,
    preferred_size: tuple[float, float] | None = None,
) -> None:
    """Apply exact, deterministic placement after Ollama proposes the layout.

    ``preferred_size`` is Ollama's chosen object size; it replaces the asset's
    native size and is still scaled down when the layout needs it to fit.
    """

    if not elements:
        return
    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    source = assets[elements[0]["asset"]]["dimensions"]
    native_width, native_height = preferred_size or (float(source["width"]), float(source["height"]))
    count = len(elements)
    layout = str(spec.get("layout") or "freeform")
    horizontal_alignment = str(spec.get("horizontal_alignment") or "center")
    vertical_alignment = str(spec.get("vertical_alignment") or "center")
    equal_spacing = bool(spec.get("equal_spacing"))
    explicit_spacing = spec.get("spacing")

    if layout == "vertical_stack" and (spec.get("primary_fill") or spec.get("primary_width_fraction")):
        # Full-width rows ('90% of the background'); with 'fit' they also share the
        # height evenly, otherwise they keep the asset's proportions.
        gap = float(explicit_spacing) if explicit_spacing is not None else _gutter(spec)
        width = canvas_width
        fill_height = max(1.0, (canvas_height - gap * (count - 1)) / count)
        height = fill_height if spec.get("primary_fill") else min(
            fill_height, width * native_height / native_width)
        block = count * height + gap * (count - 1)
        start_y = _aligned_start(canvas_height, block, vertical_alignment)
        for index, element in enumerate(elements):
            element.update({"x": 0.0, "y": start_y + index * (height + gap),
                            "width": width, "height": height, "rotation": 0.0})
        return

    if layout == "vertical_stack":
        minimum_gap = float(explicit_spacing) if explicit_spacing is not None else _gutter(spec)
        maximum_item_height = max(
            1.0,
            (canvas_height - minimum_gap * max(0, count - 1)) / count,
        )
        width, height = _fit_dimensions(
            native_width,
            native_height,
            canvas_width * 0.9,
            maximum_item_height,
        )
        x = _aligned_start(canvas_width, width, horizontal_alignment)
        if equal_spacing:
            gap = max(0.0, (canvas_height - count * height) / (count + 1))
            y_positions = [gap + index * (height + gap) for index in range(count)]
        else:
            gap = minimum_gap
            group_height = count * height + max(0, count - 1) * gap
            start_y = _aligned_start(canvas_height, group_height, vertical_alignment)
            y_positions = [start_y + index * (height + gap) for index in range(count)]
        for index, element in enumerate(elements):
            element.update(
                {
                    "x": x,
                    "y": y_positions[index],
                    "width": width,
                    "height": height,
                    "rotation": 0.0,
                }
            )
        return

    if layout == "horizontal_row":
        minimum_gap = float(explicit_spacing) if explicit_spacing is not None else _gutter(spec)
        maximum_item_width = max(
            1.0,
            (canvas_width - minimum_gap * max(0, count - 1)) / count,
        )
        width, height = _fit_dimensions(
            native_width,
            native_height,
            maximum_item_width,
            canvas_height * 0.9,
        )
        y = _aligned_start(canvas_height, height, vertical_alignment)
        if equal_spacing:
            gap = max(0.0, (canvas_width - count * width) / (count + 1))
            x_positions = [gap + index * (width + gap) for index in range(count)]
        else:
            gap = minimum_gap
            group_width = count * width + max(0, count - 1) * gap
            start_x = _aligned_start(canvas_width, group_width, horizontal_alignment)
            x_positions = [start_x + index * (width + gap) for index in range(count)]
        for index, element in enumerate(elements):
            element.update(
                {
                    "x": x_positions[index],
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0.0,
                }
            )
        return

    if layout == "grid":
        requested_columns = spec.get("grid_columns")
        if requested_columns is None:
            columns = max(1, math.ceil(math.sqrt(count)))
        else:
            columns = max(1, min(count, int(requested_columns)))
        rows = math.ceil(count / columns)
        if spec.get("primary_fill"):
            # 'Adjust width and height so it fits': boxes fill the grid area,
            # separated by the spacing (or a default gutter).
            gutter = float(explicit_spacing) if explicit_spacing is not None else _gutter(spec)
            width = max(1.0, (canvas_width - gutter * (columns - 1)) / columns)
            height = max(1.0, (canvas_height - gutter * (rows - 1)) / rows)
            for index, element in enumerate(elements):
                element.update(
                    {
                        "x": (index % columns) * (width + gutter),
                        "y": (index // columns) * (height + gutter),
                        "width": width,
                        "height": height,
                        "rotation": 0.0,
                    }
                )
            return
        # Layout bias: columns span the area with one consistent gutter, objects keep
        # their proportions, and the whole grid block is aligned as requested.
        gutter = float(explicit_spacing) if explicit_spacing is not None else _gutter(spec)
        width = max(1.0, (canvas_width - gutter * (columns - 1)) / columns)
        height = width * native_height / native_width
        if rows * height + gutter * (rows - 1) > canvas_height:
            height = max(1.0, (canvas_height - gutter * (rows - 1)) / rows)
            width = height * native_width / native_height
        block_width = columns * width + gutter * (columns - 1)
        block_height = rows * height + gutter * (rows - 1)
        start_x = _aligned_start(canvas_width, block_width, horizontal_alignment)
        start_y = _aligned_start(canvas_height, block_height, vertical_alignment)
        for index, element in enumerate(elements):
            row = index // columns
            column = index % columns
            element.update(
                {
                    "x": start_x + column * (width + gutter),
                    "y": start_y + row * (height + gutter),
                    "width": width,
                    "height": height,
                    "rotation": 0.0,
                }
            )
        return

    if layout == "overlap":
        width, height = _fit_dimensions(
            native_width,
            native_height,
            canvas_width * 0.9,
            canvas_height * 0.9,
        )
        x = _aligned_start(canvas_width, width, horizontal_alignment)
        y = _aligned_start(canvas_height, height, vertical_alignment)
        for element in elements:
            element.update(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0.0,
                }
            )
        return

    # Several freeform copies without distinct, valid spots (e.g. the offline fallback)
    # would pile up in one place; stack them instead.
    if count > 1:
        spots = {
            (round(float(element.get("x", -1))), round(float(element.get("y", -1))))
            for element in elements
        }
        if len(spots) < count or any(x < 0 or y < 0 for x, y in spots):
            _position_primary_elements(
                elements, {**spec, "layout": "vertical_stack"}, canvas, assets,
                preferred_size=preferred_size,
            )
            return

    # Freeform keeps valid Ollama coordinates, but newly copied objects are
    # placed in the center rather than left at (0, 0).
    for element in elements:
        width, height = _fit_dimensions(
            float(element.get("width") or native_width),
            float(element.get("height") or native_height),
            canvas_width,
            canvas_height,
        )
        element["width"] = width
        element["height"] = height
        # Alignment the user actually wrote wins over Ollama's coordinates.
        # (Only for a single object; several copies keep Ollama's distinct spots.)
        stated = (spec.get("hard_constraints") or {}) if len(elements) == 1 else {}
        if stated.get("horizontal_alignment") or not (
            0 <= float(element.get("x", -1)) <= canvas_width - width
        ):
            element["x"] = _aligned_start(canvas_width, width, horizontal_alignment)
        if stated.get("vertical_alignment") or not (
            0 <= float(element.get("y", -1)) <= canvas_height - height
        ):
            element["y"] = _aligned_start(canvas_height, height, vertical_alignment)


def _enforce_additional_assets(
    elements: list[dict[str, Any]],
    spec: dict[str, Any],
    canvas: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    proposals: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Guarantee each selected extra asset exists and sits where it was requested.

    Sizes follow Ollama's proposal (aspect ratio kept); colours follow the user's
    words, else Ollama's choice, and only for single-colour assets.
    """

    proposals = proposals or {}

    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    gap = _gutter(spec)
    default_margin = float((spec.get("layout_system") or {}).get("margin")
                           or round(0.04 * min(canvas_width, canvas_height)))

    # Every copy of every selected asset, tagged with its requested anchor.
    items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for placement_index, placement in enumerate(spec.get("asset_placements") or []):
        filename = placement.get("asset")
        if filename not in assets:
            continue
        count = int(placement.get("count") or 1)
        if placement.get("container") and placement.get("per_box") == "each":
            count = int(spec.get("primary_count") or count)  # One in every box.
        copies = [
            _copy_plan_element(element)
            for element in elements
            if element.get("asset") == filename
        ][:count]
        while len(copies) < count:
            copies.append(_new_plan_element(filename, assets, element_id="extra", z_index=0))
        # Friendly ids ('gavel-mallet-1') so text can be placed relative to them.
        readable = re.sub(r"(?:^|[-_\s])(?:\d+|[0-9a-fA-F]{8,})(?=$|[-_\s])", "-", Path(filename).stem)
        if not re.search(r"[A-Za-z]", readable):  # e.g. '1529054331.svg': name it from its usage
            usage_words = [
                word for word in re.findall(r"[a-z]{3,}", str(assets[filename].get("usage") or "").lower())
                if word not in GENERIC_ASSET_WORDS
            ]
            readable = "-".join(usage_words[:3])
        stem = _slugify(readable, "asset")[:40]
        full_width = bool(placement.get("full_width"))
        margin = placement.get("margin")
        margin = float(margin) if margin is not None else (0.0 if full_width else default_margin)

        native = assets[filename]["dimensions"]
        native_width, native_height = float(native["width"]), float(native["height"])
        proposal = proposals.get(filename) or {}
        dimensions = {"width": native_width, "height": native_height}
        if proposal.get("size"):
            proposed_width, proposed_height = proposal["size"]
            if full_width:  # Bars keep the full width; Ollama may change their thickness.
                dimensions["height"] = min(max(proposed_height, 0.5 * native_height), 3 * native_height)
            else:  # Illustrations keep their proportions at Ollama's chosen width,
                # but never shrink below a clearly visible size (15% of the canvas height).
                floor_height = min(native_height, 0.15 * canvas_height)
                proposed_width = max(proposed_width, floor_height * native_width / native_height)
                dimensions = {"width": proposed_width,
                              "height": proposed_width * native_height / native_width}

        style: dict[str, Any] = {}
        recolorable = bool(assets[filename].get("recolorable"))
        if placement.get("color"):
            if recolorable:
                style["fill"] = str(placement["color"])
            else:
                print(f"[Planner] {filename} is a multi-colour illustration; "
                      "keeping its own colours instead of one flat fill.")
        elif recolorable:
            for key in ("fill", "stroke"):
                try:
                    value = _safe_color(proposal.get(key), field=f"{filename}.{key}")
                except PlannerError:
                    value = None
                if value:
                    style[key] = value

        for index, element in enumerate(copies):
            element.update(
                {
                    "id": f"{stem}-{index + 1}",
                    "asset": filename,
                    "z_index": 500 + placement_index * 10 + index,
                    "rotation": 0.0,
                    "preserve_aspect_ratio": not full_width,
                    "style": dict(style),
                    "placement": {
                        "vertical": str(placement.get("vertical") or "center"),
                        "horizontal": str(placement.get("horizontal") or "center"),
                        "margin": margin,
                        "full_width": full_width,
                        **({"container": "primary", "justify": placement.get("justify", "left"),
                            "per_box": placement.get("per_box", "spread")}
                           if placement.get("container") else {}),
                    },
                }
            )
            items.append((element, dimensions))

    def anchored_y(vertical: str, height: float, margin: float, edge_offset: float) -> float:
        if vertical == "top":
            return margin + edge_offset
        if vertical == "bottom":
            return canvas_height - margin - height - edge_offset
        return (canvas_height - height) / 2

    # Full-width bars stack along their edge instead of covering each other.
    edge_used = {"top": 0.0, "bottom": 0.0, "center": 0.0}
    for element, native in items:
        where = element["placement"]
        if not where["full_width"]:
            continue
        width = max(1.0, canvas_width - 2 * where["margin"])
        height = float(native["height"])
        element.update(
            {
                "x": where["margin"],
                "y": anchored_y(where["vertical"], height, where["margin"], edge_used[where["vertical"]]),
                "width": width,
                "height": height,
            }
        )
        edge_used[where["vertical"]] += height

    # Other assets sharing an anchor (e.g. 'gavel and chart at the bottom right')
    # form one row there, so they sit side by side rather than on top of each other.
    groups: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for element, native in items:
        where = element["placement"]
        if where.get("container"):
            # Placed inside the boxes once they are positioned; keep the native size for now.
            element.update({"x": 0.0, "y": 0.0, "width": float(native["width"]),
                            "height": float(native["height"])})
            continue
        if not where["full_width"]:
            groups.setdefault((where["vertical"], where["horizontal"]), []).append((element, native))
    for (vertical, horizontal), members in groups.items():
        margin = max(member[0]["placement"]["margin"] for member in members)
        slot_width = max(1.0, (canvas_width - 2 * margin - gap * (len(members) - 1)) / len(members))
        sizes = [
            _fit_dimensions(float(native["width"]), float(native["height"]), slot_width, canvas_height * 0.45)
            for _, native in members
        ]
        row_width = sum(width for width, _ in sizes) + gap * (len(members) - 1)
        x = margin + _aligned_start(canvas_width - 2 * margin, row_width, horizontal)
        for (element, _), (width, height) in zip(members, sizes):
            element.update(
                {
                    "x": x,
                    "y": anchored_y(vertical, height, margin, edge_used.get(vertical, 0.0)),
                    "width": width,
                    "height": height,
                }
            )
            x += width + gap
    return [element for element, _ in items]


def _snap8(value: float) -> float:
    return float(max(8, round(value / 8) * 8))


def _gutter(spec: dict[str, Any]) -> float:
    return float((spec.get("layout_system") or {}).get("gutter") or 24.0)


def _layout_system(canvas_width: float, canvas_height: float, spec: dict[str, Any]) -> dict[str, Any]:
    """Infographic layout bias: one spacing scale and a 12-column grid for everything.

    Outer margins (~5% of the width) and gutters (~3%) are multiples of 8px, so
    objects, extra assets and text all share the same edges and rhythm. A header
    band is kept free for a title when the user did not fix the vertical position.
    """

    margin = _snap8(0.05 * canvas_width)
    gutter = _snap8(0.03 * min(canvas_width, canvas_height))
    if spec.get("spacing") is not None:
        gutter = float(spec["spacing"])
    columns = 12
    content_width = canvas_width - 2 * margin
    stated = spec.get("hard_constraints") or {}
    wants_header = bool(spec.get("primary_asset") and (
        spec.get("header_requested")  # 'leave space at the top for heading'
        or (not spec.get("primary_offset")
            and spec.get("primary_margin") is None
            and not stated.get("vertical_alignment")
            and spec.get("layout") != "overlap")
    ))
    return {
        "margin": margin,
        "gutter": gutter,
        "columns": columns,
        "column_width": (content_width - gutter * (columns - 1)) / columns,
        "header_height": _snap8(0.10 * canvas_height) if wants_header else 0.0,
    }


def _snap_width_to_columns(width: float, system: dict[str, Any]) -> float:
    """Nearest width that spans whole columns of the layout grid (at least 2)."""

    column, gutter = float(system["column_width"]), float(system["gutter"])
    spans = [n * column + (n - 1) * gutter for n in range(2, int(system["columns"]) + 1)]
    return min(spans, key=lambda span: abs(span - width))


def _place_contained_extras(
    primary_elements: list[dict[str, Any]],
    extra_elements: list[dict[str, Any]],
    spec: dict[str, Any],
) -> None:
    """Put 'inside the box' assets into the boxes like a CSS flex row.

    Each box is a flex container: items are vertically centred, sized to the box
    height minus padding (proportions kept, at most 40% of the box width in total),
    and justified to the start (left), center or end (right). With as many assets
    as boxes, box N gets asset N; 'in each box' puts every asset in every box.
    """

    contained = [e for e in extra_elements if (e.get("placement") or {}).get("container")]
    if not contained or not primary_elements:
        return
    boxes = sorted(primary_elements, key=lambda e: (round(float(e["y"])), float(e["x"])))
    per_box: list[list[dict[str, Any]]] = [[] for _ in boxes]
    by_asset: dict[str, list[dict[str, Any]]] = {}
    for element in contained:
        by_asset.setdefault(element["asset"], []).append(element)
    if all(len(copies) == len(boxes) for copies in by_asset.values()):
        for copies in by_asset.values():  # Every asset in every box.
            for index, element in enumerate(copies):
                per_box[index].append(element)
    else:
        for index, element in enumerate(contained):  # Box N gets the Nth asset.
            per_box[index % len(boxes)].append(element)

    gap = _gutter(spec) / 2
    for box, items in zip(boxes, per_box):
        if not items:
            continue
        bx, by, bw, bh = (float(box[key]) for key in ("x", "y", "width", "height"))
        padding = max(gap, 0.12 * min(bw, bh))
        target_height = max(1.0, bh - 2 * padding)
        sizes = []
        for element in items:
            aspect = float(element["width"]) / float(element["height"])
            sizes.append((target_height * aspect, target_height))
        row_width = sum(w for w, _ in sizes) + gap * (len(items) - 1)
        limit = 0.4 * bw
        if row_width > limit:  # Keep most of the box free for its text.
            scale = limit / row_width
            sizes = [(w * scale, h * scale) for w, h in sizes]
            row_width = limit
        justify = str(items[0]["placement"].get("justify") or "left")
        x = (bx + padding if justify == "left" else bx + bw - padding - row_width
             if justify == "right" else bx + (bw - row_width) / 2)
        for element, (width, height) in zip(items, sizes):
            element.update({"x": x, "y": by + (bh - height) / 2, "width": width, "height": height,
                            "z_index": int(box.get("z_index", 1)) + 100})
            element["placement"]["container_id"] = box["id"]
            x += width + gap


def _resolve_extra_overlaps(
    primary_elements: list[dict[str, Any]],
    extra_elements: list[dict[str, Any]],
    spec: dict[str, Any],
    canvas: dict[str, Any],
) -> None:
    """Move and, if necessary, resize extras that still cover the main objects.

    An overlapping asset goes to the larger free band above or below the objects
    (between any edge assets), keeps its horizontal alignment, and is scaled down
    with its proportions kept when the band is smaller than the asset.
    """

    if not primary_elements:
        return
    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    gap = _gutter(spec)
    margin = float((spec.get("layout_system") or {}).get("margin") or 0.0)
    boxes = [_element_box(e) for e in primary_elements]
    g_left, g_top = min(b[0] for b in boxes), min(b[1] for b in boxes)
    g_right, g_bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)

    def overlaps(box: tuple[float, float, float, float]) -> bool:
        return box[0] < g_right and g_left < box[2] and box[1] < g_bottom and g_top < box[3]

    for element in extra_elements:
        placement = element.get("placement") or {}
        if placement.get("full_width") or placement.get("container") or not overlaps(_element_box(element)):
            continue
        others = [_element_box(e) for e in extra_elements
                  if e is not element and not (e.get("placement") or {}).get("container")]
        # The header band stays free for the title.
        header = float((spec.get("layout_system") or {}).get("header_height") or 0.0)
        above_limit = max([b[3] for b in others if b[3] <= g_top] + [margin]) + gap + header
        below_limit = min([b[1] for b in others if b[1] >= g_bottom] + [canvas_height - margin]) - gap
        bands = [
            (g_top - gap - above_limit, above_limit, g_top - gap),
            (below_limit - (g_bottom + gap), g_bottom + gap, below_limit),
        ]
        room, band_top, band_bottom = max(bands, key=lambda band: band[0])
        width, height = float(element["width"]), float(element["height"])
        scale = min(1.0, room / height, (canvas_width - 2 * margin) / width)
        if scale < 0.3:
            continue  # No sensible spot; the text stage can still shrink it slightly.
        width, height = width * scale, height * scale
        element["width"], element["height"] = width, height
        element["x"] = margin + _aligned_start(
            canvas_width - 2 * margin, width, str(placement.get("horizontal") or "center"))
        element["y"] = band_top + (band_bottom - band_top - height) / 2
        placement["vertical"] = "top" if band_bottom <= g_top else "bottom"
        print(f"[Planner] Placed {element['id']} in the free space "
              f"{'above' if band_bottom <= g_top else 'below'} the objects "
              f"({width:.0f}x{height:.0f}px).")


def _clamp_to_canvas(elements: list[dict[str, Any]], canvas: dict[str, Any]) -> None:
    """Scale down anything larger than the canvas and pull it fully inside."""

    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    for element in elements:
        width, height = float(element["width"]), float(element["height"])
        scale = min(1.0, canvas_width / width, canvas_height / height)
        if scale < 1.0:
            element["width"], element["height"] = width * scale, height * scale
        element["x"] = min(max(float(element["x"]), 0.0), canvas_width - float(element["width"]))
        element["y"] = min(max(float(element["y"]), 0.0), canvas_height - float(element["height"]))


def _relayout_row(
    members: list[dict[str, Any]],
    horizontal: str,
    canvas_width: float,
    gap: float = 24.0,
) -> None:
    """Re-align one row of same-anchor extras after their sizes changed."""

    margin = max(float((m.get("placement") or {}).get("margin") or 0) for m in members)
    row_width = sum(float(m["width"]) for m in members) + gap * (len(members) - 1)
    x = margin + _aligned_start(canvas_width - 2 * margin, row_width, horizontal)
    for member in members:
        member["x"] = x
        x += float(member["width"]) + gap


def _primary_region(
    extra_elements: list[dict[str, Any]],
    spec: dict[str, Any],
    canvas: dict[str, Any],
    primary_count: int,
) -> tuple[float, float, float, float]:
    """The area left for the main objects: canvas minus margins and edge bands.

    Extras anchored to the top or bottom edge claim a band there. A band may use at
    most half the canvas height for a single object, or a quarter when there are
    several (grids and stacks need the room); oversized extras such as a large
    illustration are scaled down, keeping their proportions, to fit their band.
    """

    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    gap = _gutter(spec)
    system = spec.get("layout_system") or {}
    # A margin the user wrote wins; otherwise the layout system's outer margin.
    margin = float(spec["primary_margin"]) if spec.get("primary_margin") is not None else float(
        system.get("margin") or 0.0)
    cap = (0.5 if primary_count <= 1 else 0.25) * canvas_height
    bands = {"top": 0.0, "bottom": 0.0}
    if primary_count:
        # The middle belongs to the main objects; a centre-anchored illustration gets
        # its own band below them instead of being drawn over them.
        for element in extra_elements:
            placement = element.get("placement") or {}
            if placement.get("vertical") == "center" and not placement.get("full_width"):
                placement["vertical"] = "bottom"
                element["y"] = canvas_height - float(placement.get("margin") or margin) - float(element["height"])
                print(f"[Planner] {element['id']}: the middle holds the main objects; "
                      "placing it in its own band below them.")
        for edge in bands:
            members = [e for e in extra_elements if (e.get("placement") or {}).get("vertical") == edge]
            if not members:
                continue

            def band() -> float:
                if edge == "top":
                    return max(float(e["y"]) + float(e["height"]) for e in members) + gap
                return canvas_height - min(float(e["y"]) for e in members) + gap

            if band() > cap:
                shrinkable = [e for e in members if not (e.get("placement") or {}).get("full_width")]
                for element in shrinkable:
                    # Keep the edge side fixed; shrink towards the edge.
                    if edge == "top":
                        allowed = cap - gap - float(element["y"])
                    else:
                        allowed = cap - gap - (canvas_height - float(element["y"]) - float(element["height"]))
                    factor = allowed / float(element["height"])
                    if 0.2 <= factor < 1:
                        bottom = float(element["y"]) + float(element["height"])
                        element["width"] = float(element["width"]) * factor
                        element["height"] = float(element["height"]) * factor
                        if edge == "bottom":
                            element["y"] = bottom - float(element["height"])
                rows: dict[str, list[dict[str, Any]]] = {}
                for element in shrinkable:
                    rows.setdefault(str(element["placement"].get("horizontal") or "center"), []).append(element)
                for horizontal, row in rows.items():
                    _relayout_row(row, horizontal, canvas_width, gap)
            bands[edge] = band()
    top = max(margin, bands["top"])
    # Header band for a title (layout bias), below any top-edge assets.
    header = float(system.get("header_height") or 0.0)
    if header and primary_count:
        top = (bands["top"] if bands["top"] else margin) + header
    bottom = max(margin, bands["bottom"])
    return (
        margin,
        top,
        max(1.0, canvas_width - 2 * margin),
        max(1.0, canvas_height - top - bottom),
    )


def _element_box(element: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y = float(element["x"]), float(element["y"])
    return x, y, x + float(element["width"]), y + float(element["height"])


def _fit_primary_and_extras(
    primary_elements: list[dict[str, Any]],
    extra_elements: list[dict[str, Any]],
    spec: dict[str, Any],
    canvas: dict[str, Any],
) -> None:
    """Apply 'N px below <asset>' and keep edge-anchored extras clear of the objects."""

    if not primary_elements:
        return
    canvas_width = float(canvas["width"])
    canvas_height = float(canvas["height"])
    gap = _gutter(spec)

    def group() -> tuple[float, float, float, float]:
        boxes = [_element_box(element) for element in primary_elements]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def shift(dy: float) -> None:
        _, top, _, bottom = group()
        dy = max(-top, min(dy, canvas_height - bottom))
        for element in primary_elements:
            element["y"] = float(element["y"]) + dy

    offset = spec.get("primary_offset")
    targets = [
        element for element in extra_elements
        if offset and element.get("asset") == offset.get("asset")
    ]
    if targets:
        _, target_top, _, target_bottom = _element_box(targets[0])
        _, top, _, bottom = group()
        if offset["relation"] == "below":
            shift(target_bottom + float(offset["gap"]) - top)
        else:
            shift(target_top - float(offset["gap"]) - bottom)

    def overlaps_group(element: dict[str, Any]) -> bool:
        left, top, right, bottom = _element_box(element)
        g_left, g_top, g_right, g_bottom = group()
        return left < g_right and g_left < right and top < g_bottom and g_top < bottom

    # Move the objects below a top border they would otherwise sit under.
    for element in extra_elements:
        placement = element.get("placement") or {}
        if (placement.get("vertical") == "top" and placement.get("full_width")
                and not targets and overlaps_group(element)):
            shift(_element_box(element)[3] + gap - group()[1])

    # Shrink a bottom row into the space left under the objects, keeping its alignment.
    rows: dict[str, list[dict[str, Any]]] = {}
    for element in extra_elements:
        placement = element.get("placement") or {}
        if placement.get("vertical") == "bottom" and not placement.get("full_width"):
            rows.setdefault(str(placement.get("horizontal") or "center"), []).append(element)
    for horizontal, members in rows.items():
        if not any(overlaps_group(element) for element in members):
            continue
        row_bottom = max(_element_box(element)[3] for element in members)
        tallest = max(float(element["height"]) for element in members)
        available = row_bottom - (group()[3] + gap)
        if available < 0.2 * tallest:
            continue
        scale = available / tallest
        margin = max(float((element.get("placement") or {}).get("margin") or 0) for element in members)
        for element in members:
            element["width"] = float(element["width"]) * scale
            element["height"] = float(element["height"]) * scale
        row_width = sum(float(element["width"]) for element in members) + gap * (len(members) - 1)
        x = margin + _aligned_start(canvas_width - 2 * margin, row_width, horizontal)
        for element in members:
            element["x"] = x
            element["y"] = row_bottom - float(element["height"])
            x += float(element["width"]) + gap


def _color_rgb(color: Any) -> tuple[float, float, float] | None:
    value = str(color or "").strip().lower()
    value = NAMED_COLOR_HEX.get(value, value).lstrip("#")
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    if not re.fullmatch(r"[0-9a-f]{6}", value):
        return None
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]


def _color_distance(first: Any, second: Any) -> float:
    """Luminance difference 0..1; unknown colours count as clearly different."""

    a, b = _color_rgb(first), _color_rgb(second)
    if a is None or b is None:
        return 1.0
    luminance = lambda rgb: 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]  # noqa: E731
    return abs(luminance(a) - luminance(b))


def _ollama_proposals(
    plan: dict[str, Any],
    spec: dict[str, Any],
    assets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Sizes and colours Ollama chose per asset, keeping only plausible ones.

    Ollama sizes the whole composition; implausible proposals (copied example
    numbers, slivers, objects covering the canvas) fall back to native sizes.
    """

    canvas_width = float(plan["canvas"]["width"])
    canvas_height = float(plan["canvas"]["height"])
    smallest = 0.05 * min(canvas_width, canvas_height)
    full_width_assets = {
        placement["asset"]
        for placement in spec.get("asset_placements") or []
        if placement.get("full_width")
    }
    proposals: dict[str, dict[str, Any]] = {}
    for element in plan.get("elements") or []:
        filename = element.get("asset")
        if filename in proposals or filename not in assets:
            continue
        proposal: dict[str, Any] = {"size": None, "fill": None, "stroke": None}
        width, height = float(element.get("width") or 0), float(element.get("height") or 0)
        copied_example = (round(width), round(height)) in {(320, 240), (100, 100)}
        plausible = (
            smallest <= width <= canvas_width
            and smallest * (0.2 if filename in full_width_assets else 1) <= height <= canvas_height
            and width * height <= 0.8 * canvas_width * canvas_height
        )
        if filename != spec.get("background_asset") and plausible and not copied_example:
            proposal["size"] = (width, height)
        style = element.get("style") or {}
        if assets[filename].get("recolorable"):
            proposal["fill"] = style.get("fill")
            proposal["stroke"] = style.get("stroke")
        proposals[filename] = proposal

    # A proposed fill that nearly matches the background would hide the object.
    background = spec.get("background_color") or plan["canvas"].get("background_color")
    for filename, proposal in proposals.items():
        if filename == spec.get("background_asset"):
            continue
        if proposal.get("fill") and _color_distance(proposal["fill"], background) < 0.12:
            proposal["fill"] = None
    canvas_color = plan["canvas"].get("background_color")
    background_asset = spec.get("background_asset")
    if background_asset:
        proposals.setdefault(background_asset, {"size": None, "fill": None, "stroke": None})
        proposals[background_asset]["fill"] = (
            proposals[background_asset].get("fill") or canvas_color
        )
    else:
        proposals["__canvas__"] = {"size": None, "fill": canvas_color, "stroke": None}
    return proposals


def enforce_preprocessed_spec(
    plan: dict[str, Any],
    spec: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    *,
    use_ollama_proposals: bool = True,
    _canvas_grown: bool = False,
) -> dict[str, Any]:
    """Correct count, colors, assets, and placement after the Ollama layout pass.

    Counts, positions and user-written colours are binding. Sizes and all other
    colours come from Ollama's composition when ``use_ollama_proposals`` is set.
    """

    original_spec = spec
    original_elements = json.loads(json.dumps(plan["elements"]))
    spec = {
        **spec,
        "layout_system": _layout_system(
            float(plan["canvas"]["width"]), float(plan["canvas"]["height"]), spec
        ),
    }
    proposals = _ollama_proposals(plan, spec, assets) if use_ollama_proposals else {}
    elements = [_copy_plan_element(element) for element in plan["elements"]]
    background_asset = spec.get("background_asset")
    primary_asset = spec.get("primary_asset")
    additional_assets = set(spec.get("additional_assets") or [])

    # Background colour is Ollama's choice unless the user named one (or a gradient).
    user_background = (spec.get("hard_constraints") or {}).get("background_color")
    if use_ollama_proposals and not user_background and not spec.get("background_gradient"):
        proposed = (proposals.get(background_asset or "__canvas__") or {}).get("fill")
        try:
            proposed = _safe_color(proposed, field="proposed background")
        except PlannerError:
            proposed = None
        if proposed and proposed not in {"none", "transparent", "currentcolor"}:
            spec = {**spec, "background_color": proposed}

    if spec.get("strict_asset_set"):
        allowed_assets = {
            value
            for value in (background_asset, primary_asset, *additional_assets)
            if value
        }
        elements = [
            element for element in elements if element.get("asset") in allowed_assets
        ]

    background_color = str(spec.get("background_color") or "#FFFFFF")
    plan["canvas"]["background_color"] = background_color
    background_element: dict[str, Any] | None = None
    if background_asset:
        background_candidates = [
            element for element in elements if element.get("asset") == background_asset
        ]
        non_background = [
            element for element in elements if element.get("asset") != background_asset
        ]
        background_element = (
            _copy_plan_element(background_candidates[0])
            if background_candidates
            else _new_plan_element(
                background_asset,
                assets,
                element_id="background",
                z_index=-1000,
            )
        )
        background_element.update(
            {
                "id": "background",
                "x": 0.0,
                "y": 0.0,
                "width": float(plan["canvas"]["width"]),
                "height": float(plan["canvas"]["height"]),
                "z_index": -1000,
                "rotation": 0.0,
                "preserve_aspect_ratio": False,
            }
        )
        background_element["style"].update(
            {"fill": background_color, "stroke": "none", "opacity": 1.0}
        )
        background_element["style"].pop("gradient", None)
        if spec.get("background_gradient"):
            background_element["style"]["gradient"] = dict(spec["background_gradient"])
        elements = [background_element, *non_background]

    primary_elements: list[dict[str, Any]] = []
    requested_count = spec.get("primary_count")
    if primary_asset and requested_count is not None:
        requested_count = int(requested_count)
        primary_elements = [
            element for element in elements if element.get("asset") == primary_asset
        ][:requested_count]
        prototype = (
            _copy_plan_element(primary_elements[0])
            if primary_elements
            else _new_plan_element(
                primary_asset,
                assets,
                element_id="primary-1",
                z_index=1,
            )
        )
        while len(primary_elements) < requested_count:
            primary_elements.append(_copy_plan_element(prototype))

        primary_stem = _slugify(Path(primary_asset).stem, "object")
        for index, element in enumerate(primary_elements, start=1):
            element["id"] = f"{primary_stem}-{index}"
            element["z_index"] = index
            element["asset"] = primary_asset
            element["preserve_aspect_ratio"] = False
            element["style"].pop("gradient", None)
            if spec.get("primary_gradient"):
                element["style"]["gradient"] = dict(spec["primary_gradient"])
                element["style"]["fill"] = str(spec["primary_gradient"]["colors"][0])
            elif spec.get("primary_color"):
                element["style"]["fill"] = str(spec["primary_color"])
            elif use_ollama_proposals:
                proposed_fill = (proposals.get(primary_asset) or {}).get("fill")
                if proposed_fill and assets[primary_asset].get("recolorable"):
                    element["style"]["fill"] = proposed_fill
                else:
                    element["style"].pop("fill", None)
            element["style"]["opacity"] = 1.0
            # One size for every copy: Ollama's composition size, else the native size.
            size = (proposals.get(primary_asset) or {}).get("size") or (
                float(assets[primary_asset]["dimensions"]["width"]),
                float(assets[primary_asset]["dimensions"]["height"]),
            )
            element["width"], element["height"] = size

    # Extra assets first: their top/bottom bands stay free, and the objects are laid
    # out in the remaining area (inset by the requested margin).
    extra_elements = _enforce_additional_assets(
        elements, spec, plan["canvas"], assets, proposals
    )
    region_x, region_y, region_width, region_height = _primary_region(
        extra_elements, spec, plan["canvas"], len(primary_elements)
    )
    if spec.get("primary_width_fraction"):
        # 'width is 90% of the background': the objects span exactly that, centred.
        region_width = float(spec["primary_width_fraction"]) * float(plan["canvas"]["width"])
        region_x = (float(plan["canvas"]["width"]) - region_width) / 2
    if primary_elements:
        for element in primary_elements:  # Ollama's coordinates, relative to the region.
            element["x"] = float(element.get("x", -1)) - region_x
            element["y"] = float(element.get("y", -1)) - region_y
        _position_primary_elements(
            primary_elements,
            spec,
            {"width": region_width, "height": region_height},
            assets,
            preferred_size=(proposals.get(primary_asset) or {}).get("size"),
        )
        for element in primary_elements:
            element["x"] = float(element["x"]) + region_x
            element["y"] = float(element["y"]) + region_y
        # Layout bias: a lone object spans whole grid columns, so its edges line up
        # with the text and the other assets (unless the user fixed its width).
        if len(primary_elements) == 1 and not spec.get("primary_fill") \
                and not spec.get("primary_width_fraction"):
            element = primary_elements[0]
            center_x = float(element["x"]) + float(element["width"]) / 2
            element["width"] = min(
                _snap_width_to_columns(float(element["width"]), spec["layout_system"]), region_width)
            element["x"] = min(max(center_x - element["width"] / 2, region_x),
                               region_x + region_width - element["width"])
        elements = [
            element for element in elements if element.get("asset") != primary_asset
        ]
        if background_element is not None and primary_asset == background_asset:
            elements = []
        elements.extend(primary_elements)

    if extra_elements:
        placed_assets = {element["asset"] for element in extra_elements}
        elements = [
            element for element in elements if element.get("asset") not in placed_assets
        ] + extra_elements
        _fit_primary_and_extras(primary_elements, extra_elements, spec, plan["canvas"])
        _place_contained_extras(primary_elements, extra_elements, spec)
        _resolve_extra_overlaps(primary_elements, extra_elements, spec, plan["canvas"])
    _clamp_to_canvas([e for e in elements if e is not background_element], plan["canvas"])

    # Shared layout grid, so the text stage uses the same margins, gutter and zones.
    system = spec["layout_system"]
    header_height = float(system.get("header_height") or 0.0)
    top_assets = [e for e in extra_elements if (e.get("placement") or {}).get("vertical") == "top"]
    header_top = (max(_element_box(e)[3] for e in top_assets) if top_assets else 0.0) + (
        0.0 if top_assets else float(system["margin"]))
    plan["layout_grid"] = {
        "margin": system["margin"],
        "gutter": system["gutter"],
        "columns": system["columns"],
        "column_width": round(system["column_width"], 2),
        "content": [round(v, 2) for v in (region_x, region_y, region_width, region_height)],
        "header": [round(v, 2) for v in (system["margin"], header_top,
                                         float(plan["canvas"]["width"]) - 2 * system["margin"],
                                         max(0.0, region_y - header_top))]
        if header_height and primary_elements else None,
    }
    # 'Adjust the height of the background if necessary': when the objects came out
    # too short to hold their content, grow the canvas by the shortfall and lay out again.
    canvas_height = float(plan["canvas"]["height"])
    minimum_object_height = 0.1 * canvas_height
    if (spec.get("canvas_can_grow") and not _canvas_grown and len(primary_elements) > 1
            and min(float(e["height"]) for e in primary_elements) < minimum_object_height - 0.5):
        rows = len({round(float(e["y"])) for e in primary_elements})
        shortfall = minimum_object_height - min(float(e["height"]) for e in primary_elements)
        new_height = min(MAX_CANVAS_SIZE, canvas_height + rows * shortfall)
        print(f"[Planner] Extending the canvas from {canvas_height:.0f}px to {new_height:.0f}px "
              "tall so everything fits.")
        plan["canvas"]["height"] = new_height
        plan["elements"] = original_elements
        return enforce_preprocessed_spec(
            plan, original_spec, assets,
            use_ollama_proposals=use_ollama_proposals, _canvas_grown=True,
        )

    plan["metadata"]["enforced_additional_assets"] = [
        element["asset"] for element in extra_elements
    ]
    plan["metadata"]["ollama_sizing"] = {
        filename: {"size": list(proposal["size"]) if proposal.get("size") else None,
                   "fill": proposal.get("fill")}
        for filename, proposal in proposals.items()
        if filename != "__canvas__"
    }

    plan["elements"] = sorted(
        elements,
        key=lambda element: (int(element.get("z_index", 0)), str(element.get("id", ""))),
    )
    if primary_asset and requested_count is not None:
        plan["template_name"] = (
            f"{requested_count}-{_slugify(Path(primary_asset).stem, 'objects')}-"
            f"{str(spec.get('layout') or 'layout').replace('_', '-')}"
        )
        plan["description"] = (
            f"Exactly {requested_count} {primary_asset} element(s) in a "
            f"{str(spec.get('layout') or 'freeform').replace('_', ' ')} layout, "
            f"using {spec.get('primary_color') or 'the original asset color'} on a "
            f"{background_color} background."
        )
    plan["preprocessing"] = {
        "original_prompt": spec.get("original_prompt"),
        "ordered_steps": spec.get("ordered_steps") or [],
        "canonical_spec": {
            key: value
            for key, value in spec.items()
            if key not in {"ordered_steps", "original_prompt"}
        },
    }
    plan["metadata"]["preprocessing_enforced"] = True
    plan["metadata"]["enforced_primary_count"] = requested_count
    plan["metadata"]["enforced_background_color"] = background_color
    plan["metadata"]["enforced_primary_color"] = spec.get("primary_color")
    return plan


def build_deterministic_fallback_plan(
    *,
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
    semantic_spec: dict[str, Any],
    schema_file: Path,
    model: str,
    reason: str,
) -> dict[str, Any]:
    """Build from the same binding steps when Ollama times out or returns bad JSON."""

    canvas_width, canvas_height = _default_canvas_size(assets)
    raw_elements: list[dict[str, Any]] = []
    background_asset = semantic_spec.get("background_asset")
    if background_asset:
        raw_elements.append(
            {
                "id": "background",
                "asset": background_asset,
                "x": 0,
                "y": 0,
                "width": canvas_width,
                "height": canvas_height,
                "z_index": -1000,
                "style": {"fill": semantic_spec.get("background_color") or "#FFFFFF"},
            }
        )

    primary_asset = semantic_spec.get("primary_asset")
    primary_count = int(semantic_spec.get("primary_count") or 0)
    if primary_asset:
        dimensions = assets[primary_asset]["dimensions"]
        for index in range(primary_count):
            raw_elements.append(
                {
                    "id": f"fallback-object-{index + 1}",
                    "asset": primary_asset,
                    # Off-canvas placeholder: freeform placement then aligns it as requested.
                    "x": -1,
                    "y": -1,
                    "width": dimensions["width"],
                    "height": dimensions["height"],
                    "z_index": index + 1,
                    "style": {
                        "fill": semantic_spec.get("primary_color")
                        or "#FFFFFF"
                    },
                }
            )

    for index, asset_filename in enumerate(
        semantic_spec.get("additional_assets") or [],
        start=len(raw_elements) + 1,
    ):
        dimensions = assets[asset_filename]["dimensions"]
        raw_elements.append(
            {
                "id": f"additional-{index}",
                "asset": asset_filename,
                "x": 0,
                "y": 0,
                "width": dimensions["width"],
                "height": dimensions["height"],
                "z_index": index,
                "style": {},
            }
        )

    if not raw_elements:
        first_filename = next(iter(assets))
        dimensions = assets[first_filename]["dimensions"]
        raw_elements.append(
            {
                "id": "fallback-asset",
                "asset": first_filename,
                "x": 0,
                "y": 0,
                "width": dimensions["width"],
                "height": dimensions["height"],
                "z_index": 0,
                "style": {},
            }
        )

    raw_plan = {
        "template_name": "deterministic-template",
        "description": "Template generated from locally preprocessed constraints.",
        "canvas": {
            "width": canvas_width,
            "height": canvas_height,
            "background_color": semantic_spec.get("background_color") or "#FFFFFF",
        },
        "elements": raw_elements,
    }
    normalized = normalize_plan(
        raw_plan,
        assets,
        user_prompt=user_prompt,
        model=model,
        schema_file=schema_file,
    )
    plan = enforce_preprocessed_spec(
        normalized, semantic_spec, assets, use_ollama_proposals=False
    )
    plan["metadata"]["layout_source"] = "deterministic_fallback"
    plan["metadata"]["fallback_reason"] = reason[:1000]
    return plan


def generate_plan(
    *,
    user_prompt: str,
    assets: dict[str, dict[str, Any]],
    semantic_spec: dict[str, Any],
    schema_file: Path,
    endpoint: str,
    model: str,
    timeout: float,
    temperature: float,
    attempts: int = 1,
) -> dict[str, Any]:
    """Ask Ollama once, then enforce or locally complete the binding steps."""

    feedback: str | None = None
    previous_response: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        print(f"[Planner] Requesting layout from Ollama ({attempt}/{attempts})...")
        prompt = build_ollama_prompt(
            user_prompt,
            assets,
            semantic_spec,
            validation_feedback=feedback,
            previous_response=previous_response,
        )
        try:
            response_text = request_ollama(
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                timeout=timeout,
                temperature=temperature,
            )
        except PlannerError as exc:
            error_text = str(exc).lower()
            if any(
                marker in error_text
                for marker in ("timed out", "did not answer", "could not connect")
            ):
                print(
                    "[Planner] Ollama timed out or is unreachable; completing the image "
                    "immediately from the preprocessed steps."
                )
                return build_deterministic_fallback_plan(
                    user_prompt=user_prompt,
                    assets=assets,
                    semantic_spec=semantic_spec,
                    schema_file=schema_file,
                    model=model,
                    reason=str(exc),
                )
            raise
        try:
            raw_plan = parse_json_object(response_text)
            normalized_plan = normalize_plan(
                raw_plan,
                assets,
                user_prompt=user_prompt,
                model=model,
                schema_file=schema_file,
            )
            return enforce_preprocessed_spec(
                normalized_plan,
                semantic_spec,
                assets,
            )
        except PlannerError as exc:
            last_error = exc
            feedback = str(exc)
            previous_response = response_text
            if attempt < attempts:
                print(f"[Planner] Invalid plan: {exc}")
                print("[Planner] Asking Ollama to correct the JSON plan...")

    print(
        "[Planner] Ollama did not return valid layout JSON; completing the image "
        "from the preprocessed steps."
    )
    return build_deterministic_fallback_plan(
        user_prompt=user_prompt,
        assets=assets,
        semantic_spec=semantic_spec,
        schema_file=schema_file,
        model=model,
        reason=str(last_error),
    )


def _parse_inline_style(value: str | None) -> dict[str, str]:
    """Parse the simple declaration list used by an SVG style attribute."""

    declarations: dict[str, str] = {}
    for declaration in (value or "").split(";"):
        if ":" not in declaration:
            continue
        key, item_value = declaration.split(":", 1)
        key = key.strip().lower()
        item_value = item_value.strip()
        if key and item_value:
            declarations[key] = item_value
    return declarations


def _serialize_inline_style(style: dict[str, str]) -> str:
    """Serialize inline CSS with stable ordering."""

    return ";".join(f"{key}:{value}" for key, value in sorted(style.items()))


def _ensure_local_svg_only(root: ET.Element) -> None:
    """Reject remote references so rendering remains fully local."""

    for element in root.iter():
        for attribute_name, attribute_value in element.attrib.items():
            lowered_name = attribute_name.lower()
            lowered_value = str(attribute_value).strip().lower()
            if lowered_name.endswith("href") and lowered_value.startswith(
                ("http:", "https:", "file:", "//")
            ):
                raise PlannerError("An SVG asset contains a non-local reference.")
            if "url(" in lowered_value:
                references = re.findall(r"url\((.*?)\)", lowered_value, flags=re.IGNORECASE)
                if any(not reference.strip(" \"'").startswith("#") for reference in references):
                    raise PlannerError("An SVG asset contains a non-local CSS reference.")


def apply_style_to_svg(
    xml_text: str,
    style: dict[str, Any],
    *,
    element_id: str,
) -> str:
    """Apply validated style values to a copied SVG's drawable elements."""

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PlannerError(f"Asset XML for '{element_id}' is invalid: {exc}") from exc

    if _local_tag(root.tag) != "svg":
        raise PlannerError(f"Asset XML for '{element_id}' does not have an SVG root.")
    _ensure_local_svg_only(root)

    root.set("data-generated-element-id", element_id)
    if "opacity" in style:
        root.set("opacity", _format_number(float(style["opacity"])))

    fill_value = str(style["fill"]) if "fill" in style else None
    gradient = style.get("gradient")
    drawables = [element for element in root.iter() if _local_tag(element.tag) in DRAWABLE_TAGS]
    if isinstance(gradient, dict) and len(gradient.get("colors") or []) == 2:
        colors = [
            _safe_color(color, field=f"{element_id}.gradient") for color in gradient["colors"]
        ]
        vectors = {
            "vertical": ("0", "0", "0", "1"),
            "horizontal": ("0", "0", "1", "0"),
            "diagonal": ("0", "0", "1", "1"),
        }
        x1, y1, x2, y2 = vectors.get(str(gradient.get("direction")), vectors["vertical"])
        gradient_id = f"gradient-{_safe_identifier(element_id, 'element')}"
        defs = ET.Element(f"{{{SVG_NAMESPACE}}}defs")
        linear = ET.SubElement(
            defs,
            f"{{{SVG_NAMESPACE}}}linearGradient",
            {"id": gradient_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )
        for offset, color in (("0", colors[0]), ("1", colors[1])):
            ET.SubElement(
                linear,
                f"{{{SVG_NAMESPACE}}}stop",
                {"offset": offset, "stop-color": str(color)},
            )
        root.insert(0, defs)
        fill_value = f"url(#{gradient_id})"

    for element in drawables:
        tag_name = _local_tag(element.tag)
        inline_style = _parse_inline_style(element.get("style"))
        if fill_value is not None:
            inline_style["fill"] = fill_value
            element.set("fill", fill_value)
        if "stroke" in style:
            inline_style["stroke"] = str(style["stroke"])
            element.set("stroke", str(style["stroke"]))
        if "stroke_width" in style:
            stroke_width = _format_number(float(style["stroke_width"]))
            inline_style["stroke-width"] = stroke_width
            element.set("stroke-width", stroke_width)
        if inline_style:
            element.set("style", _serialize_inline_style(inline_style))

        if tag_name == "rect" and "corner_radius" in style:
            radius = _format_number(float(style["corner_radius"]))
            element.set("rx", radius)
            element.set("ry", radius)

    return ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n"


def _keep_corners_round(xml_text: str, width: float, height: float) -> str:
    """Stretched boxes would get oval corners; rescale rx/ry per axis so they stay round."""

    try:
        root = ET.fromstring(xml_text)
        view_box = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip())]
    except (ET.ParseError, ValueError):
        return xml_text
    if len(view_box) != 4 or view_box[2] <= 0 or view_box[3] <= 0 or width <= 0 or height <= 0:
        return xml_text
    scale_x, scale_y = width / view_box[2], height / view_box[3]
    if abs(scale_x - scale_y) / max(scale_x, scale_y) < 0.02:
        return xml_text
    changed = False
    for element in root.iter():
        if _local_tag(element.tag) != "rect":
            continue
        radius = element.get("rx") or element.get("ry")
        try:
            radius_units = float(radius) if radius is not None else 0.0
        except ValueError:
            continue
        if radius_units <= 0:
            continue
        radius_px = radius_units * min(scale_x, scale_y)  # As drawn on the shorter axis.
        element.set("rx", _format_number(round(radius_px / scale_x, 3)))
        element.set("ry", _format_number(round(radius_px / scale_y, 3)))
        changed = True
    return ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n" if changed else xml_text


def create_run_directory(output_root: Path, template_name: str) -> Path:
    """Create a non-destructive output folder for one prompt run."""

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{_slugify(template_name)}_{timestamp}"
    candidate = output_root / base_name
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{base_name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    (candidate / "assets").mkdir(parents=False, exist_ok=False)
    return candidate


def materialize_plan_assets(
    plan: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    run_directory: Path,
) -> Path:
    """Write styled SVG copies and the final JSON render manifest."""

    assets_directory = run_directory / "assets"
    preprocessing = plan.get("preprocessing")
    if isinstance(preprocessing, dict):
        (run_directory / "preprocessed_prompt.json").write_text(
            json.dumps(preprocessing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for index, element in enumerate(plan["elements"]):
        source_asset = assets[element["asset"]]
        styled_xml = apply_style_to_svg(
            source_asset["xml"],
            element["style"],
            element_id=element["id"],
        )
        if not element.get("preserve_aspect_ratio"):
            styled_xml = _keep_corners_round(
                styled_xml, float(element["width"]), float(element["height"])
            )
        output_filename = (
            f"{index:03d}-{_slugify(element['id'], 'element')}-"
            f"{_slugify(Path(element['asset']).stem, 'asset')}.svg"
        )
        output_path = assets_directory / output_filename
        output_path.write_text(styled_xml, encoding="utf-8")
        element["generated_asset"] = output_path.relative_to(run_directory).as_posix()
        element["source_dimensions"] = dict(source_asset["dimensions"])

    plan_path = run_directory / "template_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plan_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Use a local Ollama text model to plan a template from approved SVG assets, "
            "then render it without an image-generation model."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Template request. If omitted, the program asks interactively.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local Ollama model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama base URL or full /api/generate endpoint.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Path to the Python SVG dictionary.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will contain per-run generated assets.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout per Ollama request in seconds (default: 180).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Ollama planning temperature from 0 to 2 (default: 0.1).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Create the plan and styled assets but do not run the renderer.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also create a PNG. This optional output requires CairoSVG.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the interactive/local template-planning pipeline."""

    arguments = parse_arguments(argv)
    user_prompt = (arguments.prompt or "").strip()
    if not user_prompt:
        print("Describe the image template you want to create:")
        try:
            user_prompt = input("> ").strip()
        except EOFError:
            user_prompt = ""
    if not user_prompt:
        print("ERROR: A non-empty template prompt is required.", file=sys.stderr)
        return 2

    if arguments.timeout <= 0:
        print("ERROR: --timeout must be greater than zero.", file=sys.stderr)
        return 2
    if not 0 <= arguments.temperature <= 2:
        print("ERROR: --temperature must be between 0 and 2.", file=sys.stderr)
        return 2

    try:
        print(f"[Planner] Loading assets from: {arguments.schema.resolve()}")
        assets = load_svg_dictionary(arguments.schema)
        print(f"[Planner] Loaded {len(assets)} approved SVG asset(s).")
        semantic_spec = preprocess_user_prompt(
            user_prompt=user_prompt,
            assets=assets,
            endpoint=arguments.ollama_url,
            model=arguments.model,
            timeout=arguments.timeout,
            temperature=arguments.temperature,
        )
        plan = generate_plan(
            user_prompt=user_prompt,
            assets=assets,
            semantic_spec=semantic_spec,
            schema_file=arguments.schema,
            endpoint=arguments.ollama_url,
            model=arguments.model,
            timeout=arguments.timeout,
            temperature=arguments.temperature,
        )
        run_directory = create_run_directory(arguments.output_root, plan["template_name"])
        plan_path = materialize_plan_assets(plan, assets, run_directory)
        print(f"[Planner] Styled assets: {run_directory / 'assets'}")
        print(
            f"[Preprocessor] Specification saved: "
            f"{run_directory / 'preprocessed_prompt.json'}"
        )
        print(f"[Planner] Plan saved: {plan_path}")

        from template_text_layer import TextLayerError, offer_text_layer, unload_ollama_model

        if unload_ollama_model(arguments.ollama_url, arguments.model):
            print("[Planner] Ollama model unloaded after planning.")
        else:
            print("[Planner] Could not confirm Ollama model unload; continuing.")
        try:
            offer_text_layer(
                plan_path,
                original_prompt=user_prompt,
                endpoint=arguments.ollama_url,
                model=arguments.model,
                timeout=arguments.timeout,
            )
        except TextLayerError as exc:
            raise PlannerError(str(exc)) from exc

        if not arguments.plan_only:
            from template_image_renderer import render_template

            png_path = run_directory / "generated_image.png" if arguments.png else None
            svg_path, rendered_png = render_template(plan_path, png_path=png_path)
            print(f"[Renderer] Image saved: {svg_path}")
            if rendered_png is not None:
                print(f"[Renderer] PNG saved: {rendered_png}")
        else:
            print(
                "[Planner] Plan-only mode complete. Render later with:\n"
                f'python "{Path(__file__).with_name("template_image_renderer.py")}" '
                f'"{plan_path}"'
            )
        return 0
    except (PlannerError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
