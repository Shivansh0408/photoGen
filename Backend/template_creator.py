#!/usr/bin/env python3
"""Create reusable page templates and store them in the template dictionary.

    python Backend/template_creator.py

The page is divided into a grid of 1-8 cells (numbered left to right, top to
bottom), with an optional heading band across the top. Each field is asked for
separately, like the manual text stage; press Enter to accept a [default]:

    Template name, usage, background colour / gradient
    Grid: number of cells (max 8), rows x columns, margin, gutter, heading band
    Heading and subheading: area, alignment, colour, font
    Boxes: how many, then per box its area, colour, padding and where its text goes
    Images: how many, then per image the SVG asset, area, position and size
    Additional text blocks: area and alignment

An *area* is written as:
    3          grid cell 3
    1-2        cells 1 to 2 (must form a rectangle)
    1+3        cells 1 and 3 (must form a rectangle)
    band       the heading band at the top of the page
    box 2      inside box 2 (images only)
    x,y,w,h    exact pixels

The background asset and canvas size are chosen the same way as in the Core
pipeline (the SVG dictionary asset described as a background). The result is
saved to Schema/template_dictonary/template_dictionary.py as TEMPLATE_DICTIONARY,
with a wireframe preview in Schema/template_dictonary/previews/<name>.svg.
"""

from __future__ import annotations

import ast
import pprint
import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent / "Core"))

from ollama_template_planner import (  # noqa: E402
    DEFAULT_SCHEMA_FILE,
    PlannerError,
    _default_canvas_size,
    _find_asset_by_keywords,
    load_svg_dictionary,
)
from template_text_layer import load_font_dictionary, parse_color, parse_font  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIRECTORY = PROJECT_ROOT / "Schema" / "template_dictonary"
TEMPLATE_FILE = TEMPLATE_DIRECTORY / "template_dictionary.py"
PREVIEW_DIRECTORY = TEMPLATE_DIRECTORY / "previews"

MAX_GRIDS = 8
MAX_ITEMS = 8
Rect = tuple[float, float, float, float]  # x, y, width, height


class TemplateError(RuntimeError):
    """The template cannot be created or saved."""


# --------------------------------------------------------------------------- input (as in the text stage)


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or (default or "")


def _ask_valid(prompt: str, default: str, check: Callable[[str], Any]) -> Any:
    """Ask until ``check`` accepts the answer; returns what ``check`` returns."""

    while True:
        answer = _ask(prompt, default)
        try:
            return check(answer)
        except ValueError as exc:
            print(f"  ! {exc}")
            if not sys.stdin.isatty():
                raise TemplateError(str(exc)) from exc


def _int_in(low: int, high: int) -> Callable[[str], int]:
    def check(raw: str) -> int:
        if not re.fullmatch(r"\d+", raw) or not low <= int(raw) <= high:
            raise ValueError(f"Enter a whole number from {low} to {high}.")
        return int(raw)
    return check


def _choice(*options: str) -> Callable[[str], str]:
    def check(raw: str) -> str:
        if raw.lower() not in options:
            raise ValueError(f"Use {', '.join(options[:-1])} or {options[-1]}.")
        return raw.lower()
    return check


def _yes_no(raw: str) -> bool:
    if raw.lower() in ("y", "yes"):
        return True
    if raw.lower() in ("n", "no"):
        return False
    raise ValueError("Please enter Y or N.")


def _hex(color: str) -> str:
    """parse_color from the text stage, stored as-is (#RRGGBB, rgb() or a CSS name)."""

    return parse_color(color)


def _optional_color(raw: str) -> str | None:
    return None if raw.lower() in ("", "default", "none") else _hex(raw)


def _background(raw: str) -> dict[str, Any]:
    """'white' or 'navy to #6666B3' (a vertical gradient)."""

    parts = re.split(r"\s+to\s+|\s*->\s*", raw.strip(), flags=re.I)
    if len(parts) == 2:
        return {"color": _hex(parts[0]), "gradient": {"colors": [_hex(parts[0]), _hex(parts[1])],
                                                      "direction": "vertical"}}
    if len(parts) == 1:
        return {"color": _hex(parts[0]), "gradient": None}
    raise ValueError("Use one colour, or two colours as 'navy to #6666B3'.")


# --------------------------------------------------------------------------- grid and areas


def _rect(x: float, y: float, w: float, h: float) -> Rect:
    return (round(x, 1), round(y, 1), round(w, 1), round(h, 1))


def _rect_dict(rect: Rect) -> dict[str, float]:
    return {"x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3]}


class Grid:
    """The page split into ``rows`` x ``columns`` equal cells below an optional heading band."""

    def __init__(self, width: float, height: float, rows: int, columns: int,
                 margin: float, gutter: float, band_percent: float) -> None:
        self.width, self.height = width, height
        self.rows, self.columns = rows, columns
        self.margin, self.gutter = margin, gutter
        band_height = height * band_percent / 100
        self.band: Rect | None = (_rect(margin, margin, width - 2 * margin, band_height)
                                  if band_height > 0 else None)
        top = margin + band_height + (gutter if band_height > 0 else 0)
        cell_w = (width - 2 * margin - (columns - 1) * gutter) / columns
        cell_h = (height - top - margin - (rows - 1) * gutter) / rows
        if cell_w < 40 or cell_h < 40:
            raise ValueError("The margin, gutter or heading band leave cells smaller than 40px.")
        self.cells: dict[int, Rect] = {}
        for number in range(1, rows * columns + 1):
            row, column = divmod(number - 1, columns)
            self.cells[number] = _rect(margin + column * (cell_w + gutter),
                                       top + row * (cell_h + gutter), cell_w, cell_h)

    def print_map(self) -> None:
        width = 6
        line = "+" + "+".join("-" * width for _ in range(self.columns)) + "+"
        if self.band:
            print("    +" + "-" * (self.columns * (width + 1) - 1) + "+")
            print("    |" + "band".center(self.columns * (width + 1) - 1) + "|")
        print("    " + line)
        for row in range(self.rows):
            cells = [str(row * self.columns + column + 1).center(width) for column in range(self.columns)]
            print("    |" + "|".join(cells) + "|")
            print("    " + line)
        x, y, w, h = self.cells[1]
        print(f"    Each cell is {w:.0f}x{h:.0f}px (cell 1 at {x:.0f},{y:.0f}); page "
              f"{self.width:.0f}x{self.height:.0f}px.")

    def cells_rect(self, numbers: list[int]) -> Rect:
        for number in numbers:
            if number not in self.cells:
                raise ValueError(f"There is no cell {number}; cells are 1-{len(self.cells)}.")
        rows = [(n - 1) // self.columns for n in numbers]
        columns = [(n - 1) % self.columns for n in numbers]
        covered = {r * self.columns + c + 1 for r in range(min(rows), max(rows) + 1)
                   for c in range(min(columns), max(columns) + 1)}
        if covered != set(numbers):
            raise ValueError(f"Cells {'+'.join(map(str, sorted(numbers)))} do not form a rectangle; "
                             f"the rectangle would be cells {'+'.join(map(str, sorted(covered)))}.")
        first, last = self.cells[min(numbers)], self.cells[max(numbers)]
        return _rect(first[0], first[1], last[0] + last[2] - first[0], last[1] + last[3] - first[1])


def parse_area(raw: str, grid: Grid, boxes: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """An area: cell(s), the heading band, a box (images only) or exact pixels."""

    text = re.sub(r"\s+", " ", raw.strip().lower())
    if not text:
        raise ValueError("An area is required (e.g. 3, 1-2, band, x,y,w,h).")
    if text in ("band", "heading band", "header", "top band"):
        if grid.band is None:
            raise ValueError("This template has no heading band; use a cell number instead.")
        return {"type": "band", "rect": _rect_dict(grid.band)}
    if match := re.fullmatch(r"(\d+) ?- ?(\d+)", text):
        first, last = sorted((int(match.group(1)), int(match.group(2))))
        numbers = list(range(first, last + 1))
        return {"type": "cells", "cells": numbers, "rect": _rect_dict(grid.cells_rect(numbers))}
    if re.fullmatch(r"\d+(?: ?\+ ?\d+)*", text):
        numbers = sorted({int(n) for n in re.findall(r"\d+", text)})
        return {"type": "cells", "cells": numbers, "rect": _rect_dict(grid.cells_rect(numbers))}
    if match := re.fullmatch(r"(?:inside |in )?box ?-? ?(\d+)", text):
        box_id = f"box-{match.group(1)}"
        if boxes is None:
            raise ValueError("Only images can be placed inside a box.")
        if box_id not in boxes:
            raise ValueError(f"There is no {box_id}; boxes: {', '.join(boxes) or 'none'}.")
        box = boxes[box_id]
        pad = box["padding"]
        r = box["rect"]
        return {"type": "box", "box": box_id,
                "rect": _rect_dict(_rect(r["x"] + pad, r["y"] + pad, r["width"] - 2 * pad, r["height"] - 2 * pad))}
    numbers = re.fullmatch(r"(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)", text)
    if numbers:
        x, y, w, h = (float(v) for v in numbers.groups())
        if w < 10 or h < 10:
            raise ValueError("Width and height must be at least 10px.")
        if x + w > grid.width + 0.5 or y + h > grid.height + 0.5:
            raise ValueError(f"The area leaves the {grid.width:.0f}x{grid.height:.0f}px page.")
        return {"type": "pixels", "rect": _rect_dict(_rect(x, y, w, h))}
    raise ValueError(f"Could not read area '{raw}'. Examples: 3, 1-2, 1+3, band, "
                     f"{'box 1, ' if boxes is not None else ''}40,200,500,300")


ANCHOR_WORDS = {"top", "bottom", "left", "right", "center", "centre", "middle"}


def parse_anchor(raw: str) -> dict[str, str]:
    """'top left', 'center', 'bottom right' -> vertical and horizontal alignment."""

    words = re.findall(r"[a-z]+", raw.lower().replace("-", " "))
    if not words or any(word not in ANCHOR_WORDS for word in words):
        raise ValueError("Use position words: top/center/bottom and left/center/right "
                         "(e.g. 'top left', 'center').")
    vertical = next((w for w in words if w in ("top", "bottom")), "center")
    horizontal = next((w for w in words if w in ("left", "right")), "center")
    return {"vertical": vertical, "horizontal": horizontal}


def _percent(raw: str) -> float:
    value = raw.strip().rstrip("%")
    if not re.fullmatch(r"\d+(?:\.\d+)?", value) or not 5 <= float(value) <= 100:
        raise ValueError("Enter a size from 5 to 100 (% of the area).")
    return float(value)


def place_in_area(area: dict[str, float], aspect: float, size_percent: float,
                  anchor: dict[str, str]) -> Rect:
    """Fit an asset of ``aspect`` (w/h) into ``size_percent`` of the area, aligned by ``anchor``."""

    max_w, max_h = area["width"] * size_percent / 100, area["height"] * size_percent / 100
    width = min(max_w, max_h * aspect)
    height = width / aspect
    x = {"left": area["x"], "center": area["x"] + (area["width"] - width) / 2,
         "right": area["x"] + area["width"] - width}[anchor["horizontal"]]
    y = {"top": area["y"], "center": area["y"] + (area["height"] - height) / 2,
         "bottom": area["y"] + area["height"] - height}[anchor["vertical"]]
    return _rect(x, y, width, height)


def _overlaps(a: dict[str, float], b: dict[str, float]) -> bool:
    return (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
            and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"])


# --------------------------------------------------------------------------- dictionary file


def load_templates(path: Path = TEMPLATE_FILE) -> dict[str, Any]:
    """Read TEMPLATE_DICTIONARY without executing the file."""

    if not path.is_file():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise TemplateError(f"Cannot read {path}: {exc}. The file was not changed.") from exc
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TEMPLATE_DICTIONARY" for t in node.targets):
            try:
                data = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise TemplateError(f"{path} is not plain literal data. The file was not changed.") from exc
            if isinstance(data, dict):
                return data
    raise TemplateError(f"{path} has no TEMPLATE_DICTIONARY. The file was not changed.")


FILE_HEADER = '''"""Page templates made with Backend/template_creator.py.

The file is read with ``ast.literal_eval`` (never executed); keep every value a
plain literal. Rectangles are pixels on the canvas: {"x", "y", "width", "height"}.

Per template
    name, usage          name and when to use it
    canvas               width, height, background_asset (chosen as in the Core
                         pipeline) and background {"color", "gradient"}
    grid                 cell_count (max 8), rows, columns, margin, gutter,
                         heading_band (rect or None) and cells {"1": rect, ...}
    heading, subheading  area {"type", "rect", "cells"}, align, vertical, color, font (or None)
    box_count, boxes     per box: id, asset, area, rect, color, gradient, padding and
                         text {"vertical", "horizontal", "color", "font"} (or None = no text)
    image_count, images  per image: id, asset, area (cells, band, box or pixels),
                         anchor, size_percent and the final rect
    text_count, text_blocks  per block: id, area, align, vertical, color, font
"""

'''


def save_templates(templates: dict[str, Any], path: Path = TEMPLATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = pprint.pformat(templates, indent=1, width=100, sort_dicts=False)
    text = FILE_HEADER + "TEMPLATE_DICTIONARY = " + body + "\n"
    ast.literal_eval(body)  # Never write a file the pipelines could not read back.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


# --------------------------------------------------------------------------- preview


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_preview(template: dict[str, Any], path: Path) -> Path:
    """A wireframe SVG: grid cells, heading band, boxes, images and text areas."""

    canvas, grid = template["canvas"], template["grid"]
    width, height = canvas["width"], canvas["height"]
    background = canvas["background"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" '
             f'viewBox="0 0 {width:g} {height:g}" font-family="Segoe UI, Arial, sans-serif">']
    if background["gradient"]:
        first, second = background["gradient"]["colors"]
        parts.append(f'<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
                     f'<stop offset="0" stop-color="{first}"/><stop offset="1" stop-color="{second}"/>'
                     f'</linearGradient></defs><rect width="100%" height="100%" fill="url(#bg)"/>')
    else:
        parts.append(f'<rect width="100%" height="100%" fill="{background["color"]}"/>')

    def rect(r: dict[str, float], stroke: str, fill: str = "none", dash: str = "", label: str = "",
             size: int = 26, label_at_top: bool = False) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<rect x="{r["x"]}" y="{r["y"]}" width="{r["width"]}" height="{r["height"]}" '
                     f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="3"{dash_attr}/>')
        if label:
            # A white outline keeps labels readable on any box or background colour.
            y = r["y"] + size if label_at_top else r["y"] + r["height"] / 2
            parts.append(f'<text x="{r["x"] + r["width"] / 2}" y="{y}" font-size="{size}" '
                         f'text-anchor="middle" dominant-baseline="middle" fill="{stroke}" '
                         f'stroke="#FFFFFF" stroke-width="6" paint-order="stroke">{_esc(label)}</text>')

    for number, cell in grid["cells"].items():
        rect(cell, "#9AA0A6", dash="10 8", label=f"cell {number}", size=40)
        parts[-1] = parts[-1].replace('fill="#9AA0A6"', 'fill="#9AA0A6" fill-opacity="0.35"')
    if grid["heading_band"]:
        rect(grid["heading_band"], "#9AA0A6", dash="10 8")
    for box in template["boxes"]:
        rect(box["rect"], "#1A1A1A", fill=box["color"],
             label=f'{box["id"]}' + (f' - text {box["text"]["vertical"]} {box["text"]["horizontal"]}'
                                     if box["text"] else " - no text"), label_at_top=True)
    for image in template["images"]:
        rect(image["rect"], "#D93025", dash="6 6", label=f'{image["id"]}: {Path(image["asset"]).stem}', size=22)
    for name in ("heading", "subheading"):
        if template[name]:
            rect(template[name]["area"]["rect"], "#1A73E8", dash="4 6",
                 label=f'{name.upper()} ({template[name]["align"]})', size=34 if name == "heading" else 26)
    for block in template["text_blocks"]:
        rect(block["area"]["rect"], "#188038", dash="4 6", label=f'{block["id"]} ({block["align"]})')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the questions


def _grid_shapes(count: int) -> list[tuple[int, int]]:
    """rows x columns that divide the page into exactly ``count`` cells (portrait first)."""

    shapes = [(rows, count // rows) for rows in range(1, count + 1) if count % rows == 0]
    return sorted(shapes, key=lambda s: (s[0] < s[1], abs(s[0] - s[1] * 1.5)))


def ask_text_area(label: str, grid: Grid, default_area: str, fonts: dict[str, Any],
                  default_align: str = "center") -> dict[str, Any]:
    area = _ask_valid(f"  {label} area", default_area, lambda raw: parse_area(raw, grid))
    align = _ask_valid(f"  {label} alignment (left/center/right)", default_align,
                       _choice("left", "center", "right"))
    vertical = _ask_valid(f"  {label} vertical position (top/center/bottom)", "center",
                          _choice("top", "center", "bottom"))
    color = _ask_valid(f"  {label} colour", "default", _optional_color)
    font = (_ask_valid(f"  {label} font (number or name)", "default",
                       lambda raw: None if raw.lower() == "default" else parse_font(raw, fonts))
            if fonts else None)
    return {"area": area, "align": align, "vertical": vertical, "color": color, "font": font}


def collect_template(assets: dict[str, dict[str, Any]], existing: dict[str, Any]) -> dict[str, Any]:
    background_asset = _find_asset_by_keywords(assets, ("background", "canvas", "backdrop"))
    box_asset = _find_asset_by_keywords(assets, ("box", "card", "panel"))
    width, height = _default_canvas_size(assets)
    if background_asset:
        dims = assets[background_asset]["dimensions"]
        width, height = float(dims["width"]), float(dims["height"])
    fonts = load_font_dictionary()
    images = {f: a for f, a in assets.items() if f not in (background_asset, box_asset)}

    print("\n[Template] Press Enter to accept a [default].")
    while True:
        name = _ask("Template name")
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not slug:
            print("  ! A name is required.")
            continue
        if slug in existing and not _ask_valid(f"  '{slug}' exists. Replace it? [Y/N]", "n", _yes_no):
            continue
        break
    usage = _ask("Usage (when to use this template)", "General infographic layout")
    print(f"[Template] Background asset: {background_asset or 'none'} ({width:.0f}x{height:.0f}px), "
          "chosen as in the Core pipeline.")
    background = _ask_valid("Background colour (or 'colour to colour' for a gradient)", "white", _background)

    print(f"\n[Template] Grid: the page is divided into equal cells (max {MAX_GRIDS}).")
    count = _ask_valid(f"Number of grid cells (1-{MAX_GRIDS})", "4", _int_in(1, MAX_GRIDS))
    shapes = _grid_shapes(count)
    choices = ", ".join(f"{r}x{c}" for r, c in shapes)

    def shape(raw: str) -> tuple[int, int]:
        match = re.fullmatch(r"(\d+)\s*[x*]\s*(\d+)", raw.strip().lower())
        if not match or (int(match.group(1)), int(match.group(2))) not in shapes:
            raise ValueError(f"Choose one of: {choices} (rows x columns).")
        return int(match.group(1)), int(match.group(2))

    rows, columns = _ask_valid(f"Rows x columns ({choices})", f"{shapes[0][0]}x{shapes[0][1]}", shape)
    margin = _ask_valid("Page margin in px", "40", _int_in(0, int(min(width, height) // 4)))
    gutter = _ask_valid("Gap between cells in px", "24", _int_in(0, 200))
    band = _ask_valid("Heading band height in % of the page (0 = none)", "12", _int_in(0, 40))
    grid = _ask_valid_grid(width, height, rows, columns, margin, gutter, band)
    grid.print_map()

    print("\n[Template] Areas: 3 | 1-2 | 1+3 | band | x,y,w,h")
    heading = subheading = None
    if _ask_valid("Add a heading? [Y/N]", "y", _yes_no):
        heading = ask_text_area("Heading", grid, "band" if grid.band else "1", fonts)
    if _ask_valid("Add a subheading? [Y/N]", "n", _yes_no):
        subheading = ask_text_area("Subheading", grid, "band" if grid.band else "1", fonts)

    boxes: dict[str, dict[str, Any]] = {}
    box_count = _ask_valid(f"\nHow many boxes (0-{MAX_ITEMS})", str(min(count, MAX_ITEMS)), _int_in(0, MAX_ITEMS))
    if box_count and not box_asset:
        print("  ! The SVG dictionary has no box asset; boxes are skipped.")
        box_count = 0
    previous_color = "#FFFFFF"
    for number in range(1, box_count + 1):
        box_id = f"box-{number}"
        print(f"[Template] {box_id}")
        area = _ask_valid(f"  {box_id} area", str(number) if number <= count else "1",
                          lambda raw: parse_area(raw, grid))
        fill = _ask_valid(f"  {box_id} colour (or 'colour to colour')", previous_color, _background)
        previous_color = fill["color"] if not fill["gradient"] else " to ".join(fill["gradient"]["colors"])
        padding = _ask_valid(f"  {box_id} inner padding in px", "16", _int_in(0, 200))
        text = None
        text_position = _ask_valid(f"  {box_id} text position (e.g. center, top left; 'none' = no text)",
                                   "center", lambda raw: None if raw.lower() == "none" else parse_anchor(raw))
        if text_position:
            text = {**text_position,
                    "color": _ask_valid(f"  {box_id} text colour", "default", _optional_color),
                    "font": (_ask_valid(f"  {box_id} text font (number or name)", "default",
                                        lambda raw: None if raw.lower() == "default" else parse_font(raw, fonts))
                             if fonts else None)}
        boxes[box_id] = {"id": box_id, "asset": box_asset, "area": area, "rect": area["rect"],
                         "color": fill["color"], "gradient": fill["gradient"], "padding": padding,
                         "text": text}

    image_list: list[dict[str, Any]] = []
    names = list(images)
    image_count = _ask_valid(f"\nHow many images (0-{MAX_ITEMS})", "0", _int_in(0, MAX_ITEMS))
    if image_count:
        print("[Template] Images you can use (number or name):")
        for index, filename in enumerate(names, 1):
            print(f"    {index:>2}. {Path(filename).stem[:32]:<32} {images[filename]['usage'][:50]}")

    def pick_asset(raw: str) -> str:
        value = raw.strip().lower()
        if value.isdigit() and 1 <= int(value) <= len(names):
            return names[int(value) - 1]
        found = [f for f in names if value in (f.lower(), Path(f).stem.lower())]
        if not found and value:
            found = [f for f in names if Path(f).stem.lower().startswith(value)]
        if len(found) == 1:
            return found[0]
        raise ValueError(f"Pick 1-{len(names)} or an asset name.")

    for number in range(1, image_count + 1):
        image_id = f"image-{number}"
        print(f"[Template] {image_id}")
        asset = _ask_valid(f"  {image_id} asset", "1", pick_asset)
        area = _ask_valid(f"  {image_id} area (cell, band, box N or x,y,w,h)",
                          "box 1" if boxes else "1", lambda raw: parse_area(raw, grid, boxes))
        anchor = _ask_valid(f"  {image_id} position in the area", "left" if area["type"] == "box" else "center",
                            parse_anchor)
        size = _ask_valid(f"  {image_id} size (% of the area)", "100" if area["type"] == "box" else "80", _percent)
        dims = assets[asset]["dimensions"]
        placed = place_in_area(area["rect"], dims["width"] / dims["height"], size, anchor)
        image_list.append({"id": image_id, "asset": asset, "area": area, "anchor": anchor,
                           "size_percent": size, "rect": _rect_dict(placed)})

    blocks: list[dict[str, Any]] = []
    block_count = _ask_valid(f"\nHow many additional text blocks (0-{MAX_ITEMS})", "0", _int_in(0, MAX_ITEMS))
    for number in range(1, block_count + 1):
        block = ask_text_area(f"Text {number}", grid, str(count), fonts, default_align="left")
        blocks.append({"id": f"text-{number}", **block})

    template = {
        "name": slug,
        "usage": usage,
        "canvas": {"width": width, "height": height, "background_asset": background_asset,
                   "background": background},
        "grid": {"cell_count": count, "rows": rows, "columns": columns, "margin": margin,
                 "gutter": gutter, "heading_band": _rect_dict(grid.band) if grid.band else None,
                 "cells": {str(n): _rect_dict(r) for n, r in grid.cells.items()}},
        "heading": heading,
        "subheading": subheading,
        "box_count": len(boxes),
        "boxes": list(boxes.values()),
        "image_count": len(image_list),
        "images": image_list,
        "text_count": len(blocks),
        "text_blocks": blocks,
    }
    _warn_overlaps(template)
    return template


def _ask_valid_grid(width: float, height: float, rows: int, columns: int,
                    margin: int, gutter: int, band: int) -> Grid:
    try:
        return Grid(width, height, rows, columns, margin, gutter, band)
    except ValueError as exc:
        raise TemplateError(f"{exc} Use a smaller margin, gap or band.") from exc


def _warn_overlaps(template: dict[str, Any]) -> None:
    boxes = template["boxes"]
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            if _overlaps(first["rect"], second["rect"]):
                print(f"[Template] Note: {first['id']} and {second['id']} overlap.")
    for image in template["images"]:
        if image["area"]["type"] != "box" and any(_overlaps(image["rect"], b["rect"]) for b in boxes):
            print(f"[Template] Note: {image['id']} overlaps a box without being placed inside it.")


def print_summary(template: dict[str, Any]) -> None:
    grid = template["grid"]
    print(f"\n[Template] '{template['name']}': {grid['rows']}x{grid['columns']} grid "
          f"({grid['cell_count']} cells), {template['box_count']} box(es), "
          f"{template['image_count']} image(s), {template['text_count']} text block(s).")
    for name in ("heading", "subheading"):
        if template[name]:
            r = template[name]["area"]["rect"]
            print(f"    {name:<12} {template[name]['align']:<6} at {r['x']:.0f},{r['y']:.0f} "
                  f"{r['width']:.0f}x{r['height']:.0f}")
    for box in template["boxes"]:
        r = box["rect"]
        text = f"text {box['text']['vertical']} {box['text']['horizontal']}" if box["text"] else "no text"
        print(f"    {box['id']:<12} {r['x']:.0f},{r['y']:.0f} {r['width']:.0f}x{r['height']:.0f}  "
              f"{box['color']}  {text}")
    for image in template["images"]:
        r = image["rect"]
        print(f"    {image['id']:<12} {Path(image['asset']).stem[:24]:<24} {r['x']:.0f},{r['y']:.0f} "
              f"{r['width']:.0f}x{r['height']:.0f}")


def main() -> int:
    try:
        assets = load_svg_dictionary(DEFAULT_SCHEMA_FILE)
        templates = load_templates()
        template = collect_template(assets, templates)
        print_summary(template)
        preview = write_preview(template, PREVIEW_DIRECTORY / f"{template['name']}.svg")
        print(f"[Template] Preview: {preview}")
        if not _ask_valid("Save this template? [Y/N]", "y", _yes_no):
            print("[Template] Not saved.")
            return 0
        templates[template["name"]] = template
        save_templates(templates)
        print(f"[Template] Saved '{template['name']}' to {TEMPLATE_FILE} "
              f"({len(templates)} template(s) in total).")
        return 0
    except (TemplateError, PlannerError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
