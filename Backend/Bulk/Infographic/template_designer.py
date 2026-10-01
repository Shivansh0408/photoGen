"""Turn a saved page template into a finished infographic plan, without the planner.

Used by user_request.py. For each image the bulk pipeline picks a template from
Schema/template_dictonary/template_dictionary.py (made with Backend/template_creator.py)
and this module:

* swaps the template's image slots for assets relevant to the topic, refitting
  each to its own aspect ratio, and edits the layout for better visuals
  (an icon in every text box, an illustration in empty decorative boxes, image
  slots dropped when nothing fits);
* colours everything from a palette derived from the chosen asset's usage text
  (and its own SVG colours for contrast);
* builds the template_plan elements and the text spec directly, so the only
  Ollama call per image is the one that writes the text.
"""

from __future__ import annotations

import ast
import colorsys
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ollama_template_planner import PROJECT_ROOT, _find_asset_by_keywords
from photogen import _hex_color, _luminance, _readable_on

TEMPLATE_FILE = PROJECT_ROOT / "Schema" / "template_dictonary" / "template_dictionary.py"

# Usage words -> colour mood (hue in degrees, saturation). First match wins.
USAGE_MOODS = (
    (r"moon|lunar|night|space|star|planet|galaxy|sky", 228, 0.55),
    (r"leaf|maple|mapple|autumn|fall", 18, 0.70),
    (r"tree|forest|plant|nature|eco|green|garden", 125, 0.45),
    (r"water|waterfall|ocean|sea|river|rain|wave", 195, 0.60),
    (r"judge|gavel|mallet|law|legal|justice|court", 350, 0.50),
    (r"profit|money|finance|bank|growth|graph|chart|revenue", 155, 0.50),
    (r"stress|deadline|urgent|warning|pressure", 14, 0.70),
    (r"equal|fair|gender|divers", 275, 0.45),
    (r"success|succesful|sucseful|award|win", 42, 0.70),
    (r"business|bussines|meeting|office|corporate|presentation|mangment|management|contract|tech",
     212, 0.50),
)
DEFAULT_MOOD = (212, 0.45)


# --------------------------------------------------------------------------- templates


def load_templates(path: Path = TEMPLATE_FILE) -> dict[str, dict[str, Any]]:
    """Read TEMPLATE_DICTIONARY without executing the file; {} when there is none."""

    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TEMPLATE_DICTIONARY" for t in node.targets):
            data = ast.literal_eval(node.value)
            return {str(k): v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}
    return {}


# --------------------------------------------------------------------------- colour


def _hls_hex(hue: float, lightness: float, saturation: float) -> str:
    red, green, blue = colorsys.hls_to_rgb((hue % 360) / 360, min(max(lightness, 0), 1),
                                           min(max(saturation, 0), 1))
    return "#{:02X}{:02X}{:02X}".format(round(red * 255), round(green * 255), round(blue * 255))


def svg_colors(xml: str) -> list[str]:
    """The asset's own colours, most used first (hex, rgb() and rgb(%) fills)."""

    found: list[str] = []
    for match in re.finditer(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", xml):
        if color := _hex_color(match.group(0)):
            found.append(color)
    for match in re.finditer(r"rgb\(\s*([\d.]+)(%?)\s*,\s*([\d.]+)%?\s*,\s*([\d.]+)%?\s*\)", xml):
        scale = 2.55 if match.group(2) else 1.0
        values = [min(255, round(float(match.group(i)) * scale)) for i in (1, 3, 4)]
        found.append("#{:02X}{:02X}{:02X}".format(*values))
    return [color for color, _ in Counter(found).most_common()]


def _hue_of(colors: list[str]) -> float | None:
    """Hue of the most used clearly coloured (not grey, white or black) colour."""

    for color in colors:
        red, green, blue = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
        if saturation > 0.25 and 0.15 < lightness < 0.85:
            return hue * 360
    return None


class PaletteMaker:
    """Colours per asset, decided from its usage text (Ollama only for unknown usages)."""

    def __init__(self, engine: Any, assets: dict[str, dict[str, Any]]) -> None:
        self.engine, self.assets = engine, assets
        self.moods: dict[str, tuple[float, float, str]] = {}

    def mood(self, filename: str | None, topic: str) -> tuple[float, float, str]:
        """(hue, saturation, reason) for an asset, or for the topic when there is no asset."""

        key = filename or f"topic:{topic}"
        if key in self.moods:
            return self.moods[key]
        asset = self.assets.get(filename or "", {})
        text = f"{asset.get('usage', '')} {Path(filename or '').stem} {'' if filename else topic}".lower()
        mood = next(((h, s, f"usage '{asset.get('usage', topic)}' -> {pattern.split('|')[0]} colours")
                     for pattern, h, s in USAGE_MOODS if re.search(pattern, text)), None)
        if mood is None:
            answer = self.engine._ask_ollama(
                "Pick ONE hex colour that best represents this subject in an infographic.\n"
                f"Subject: {asset.get('usage') or topic}\n"
                'Return ONLY this JSON object: {"color": "#RRGGBB"}', "Palette") or {}
            color = _hex_color(answer.get("color"))
            hue = _hue_of([color]) if color else None
            if hue is not None:
                mood = (hue, 0.55, f"Ollama colour {color} for '{asset.get('usage') or topic}'")
        if mood is None:
            hue = _hue_of(svg_colors(asset.get("xml", "")))
            mood = ((hue, 0.5, "the asset's own colours") if hue is not None
                    else (*DEFAULT_MOOD, "default corporate colours"))
        self.moods[key] = mood
        return mood

    def palette(self, filename: str | None, topic: str, rng: Any, icon: str | None) -> dict[str, Any]:
        hue, saturation, reason = self.mood(filename, topic)
        hue += rng.uniform(-12, 12)
        dark = rng.random() < 0.35
        if dark:
            background = [_hls_hex(hue, 0.13, 0.45), _hls_hex(hue + 10, 0.24, 0.45)]
            boxes = [_hls_hex(hue, 0.64, saturation), _hls_hex(hue + 18, 0.74, saturation)]
            heading = _hls_hex(hue, 0.92, 0.35)
        else:
            background = [_hls_hex(hue, 0.97, 0.35), _hls_hex(hue, 0.89, 0.40)]
            boxes = [_hls_hex(hue, 0.36, saturation), _hls_hex(hue + 18, 0.50, saturation)]
            heading = _hls_hex(hue, 0.22, saturation)
        if rng.random() < 0.35:
            background = background[:1]
        if rng.random() < 0.3:
            boxes = boxes[:1]
        # An icon inside the boxes must stand out from them: flip the boxes' lightness.
        if icon:
            icon_colors = svg_colors(self.assets[icon]["xml"])[:6]
            if icon_colors:
                icon_light = sum(_luminance(c) for c in icon_colors) / len(icon_colors)
                box_light = sum(_luminance(c) for c in boxes) / len(boxes)
                if abs(icon_light - box_light) < 0.25:
                    target = 0.30 if icon_light > 0.5 else 0.85
                    boxes = [_hls_hex(hue, target, saturation), _hls_hex(hue + 18, target + 0.08, saturation)]
                    reason += "; boxes re-toned so the icon stands out"
        heading = _readable_on(heading, background)
        return {
            "background": background,
            "boxes": boxes,
            "border": _hls_hex(hue, 0.25 if not dark else 0.80, saturation),
            "heading": heading,
            "subheading": _readable_on(_hls_hex(hue, 0.35 if not dark else 0.80, saturation * 0.8),
                                       background, heading),
            "box_text": _readable_on(None, boxes),
            "footer": _readable_on(_hls_hex(hue, 0.30 if not dark else 0.85, saturation * 0.6),
                                   background, heading),
            "accent": _hls_hex(hue + 180, 0.50, saturation),
            "reason": reason,
        }


# --------------------------------------------------------------------------- template editing


def _fit(area: dict[str, float], aspect: float, size_percent: float, anchor: dict[str, str]) -> dict[str, float]:
    """The same fitting as template_creator.place_in_area, for a new asset's aspect ratio."""

    width = min(area["width"] * size_percent / 100, area["height"] * size_percent / 100 * aspect)
    height = width / aspect
    x = {"left": area["x"], "center": area["x"] + (area["width"] - width) / 2,
         "right": area["x"] + area["width"] - width}.get(anchor.get("horizontal"), area["x"])
    y = {"top": area["y"], "center": area["y"] + (area["height"] - height) / 2,
         "bottom": area["y"] + area["height"] - height}.get(anchor.get("vertical"), area["y"])
    return {"x": round(x, 1), "y": round(y, 1), "width": round(width, 1), "height": round(height, 1)}


FRAME_PATTERN = r"border|frame|bracket|divider"


def _overlaps(a: dict[str, float], b: dict[str, float]) -> bool:
    return (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
            and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"])


def usable(template: dict[str, Any]) -> bool:
    """A template needs somewhere to put points: a box with text or a text block."""

    return any(b.get("text") for b in template.get("boxes") or []) or bool(template.get("text_blocks"))


def _inner(box: dict[str, Any]) -> dict[str, float]:
    r, pad = box["rect"], float(box.get("padding") or 0)
    return {"x": r["x"] + pad, "y": r["y"] + pad, "width": r["width"] - 2 * pad, "height": r["height"] - 2 * pad}


def _aspect(assets: dict[str, dict[str, Any]], filename: str) -> float:
    dims = assets[filename]["dimensions"]
    return float(dims["width"]) / float(dims["height"])


def edit_template(template: dict[str, Any], icons: list[str], illustrations: list[str],
                  assets: dict[str, dict[str, Any]], rng: Any) -> dict[str, Any]:
    """Fill the template's image slots with topic assets and tidy it for visuals.

    Returns {"boxes", "images", "notes", "icon"}: final boxes (sorted like the text
    stage labels them), final image placements and what was changed.
    """

    notes: list[str] = []
    boxes = sorted((dict(b) for b in template.get("boxes") or []),
                   key=lambda b: (b["rect"]["y"], b["rect"]["x"]))
    text_boxes = [b for b in boxes if b.get("text")]
    slots = list(template.get("images") or [])
    in_box = [s for s in slots if (s.get("area") or {}).get("type") == "box"]
    free = [s for s in slots if (s.get("area") or {}).get("type") != "box"]
    images: list[dict[str, Any]] = []
    pool = icons or illustrations

    # Icons in the boxes: one style for every text box, so the points read alike.
    icon_style = in_box[0] if in_box else None
    if pool and (icon_style or len(text_boxes) >= 2 and rng.random() < 0.6):
        style = icon_style or {"anchor": {"vertical": "center", "horizontal": "left"}, "size_percent": 100}
        distinct = len(pool) >= len(text_boxes) and rng.random() < 0.6
        order = list(pool)  # The caller rotates the pool so each image leads with another asset.
        for index, box in enumerate(text_boxes):
            asset = order[index % len(order)] if distinct else order[0]
            inner = _inner(box)
            if inner["height"] > 1.4 * inner["width"]:
                # Tall, narrow box: the icon sits on top and the text uses the full width below.
                side = min(inner["width"] * 0.7, inner["height"] * 0.3)
                area = {"x": inner["x"], "y": inner["y"], "width": inner["width"], "height": side}
                anchor = {"vertical": "top", "horizontal": "center"}
            else:
                # Icons take at most ~35% of the box width so the text keeps room.
                area = {**inner, "width": min(inner["width"], inner["height"] * 1.2, inner["width"] * 0.35)}
                if style["anchor"].get("horizontal") == "right":
                    area["x"] = inner["x"] + inner["width"] - area["width"]
                anchor = {"vertical": "center", "horizontal": style["anchor"].get("horizontal", "left")}
            images.append({"id": f"icon-{index + 1}", "asset": asset, "rect": _fit(
                area, _aspect(assets, asset), float(style.get("size_percent") or 100), anchor)})
        notes.append(f"icon{'s' if distinct else ''} {', '.join(sorted({Path(i['asset']).stem for i in images}))} "
                     f"in all {len(text_boxes)} text box(es)")
    elif in_box:
        notes.append("no relevant asset: in-box image slots removed")

    # Free slots and empty decorative boxes get the larger illustrations.
    used = {i["asset"] for i in images}
    spare = [a for a in illustrations + icons if a not in used] or illustrations or icons
    empty_boxes = [b for b in boxes if not b.get("text")
                   and not any(i["rect"]["x"] >= b["rect"]["x"] and i["rect"]["y"] >= b["rect"]["y"]
                               and i["rect"]["x"] < b["rect"]["x"] + b["rect"]["width"]
                               and i["rect"]["y"] < b["rect"]["y"] + b["rect"]["height"] for i in images)]
    targets = []
    for slot in free:
        if re.search(FRAME_PATTERN, assets.get(slot["asset"], {}).get("usage", ""), re.I):
            # A border or divider stays itself (recoloured later); it is layout, not content.
            # Inside a text box it would split the text, so it is left out there.
            if any(_overlaps(slot["rect"], b["rect"]) for b in text_boxes):
                notes.append(f"divider {slot['id']} removed: it split a text box")
            else:
                images.append({"id": f"frame-{len(images) + 1}", "asset": slot["asset"], "rect": slot["rect"]})
            continue
        if any(_overlaps(slot["area"]["rect"], b["rect"]) for b in text_boxes):
            notes.append(f"image slot {slot['id']} skipped: it would cover a text box")
            continue
        targets.append((slot["area"]["rect"], slot.get("anchor") or {"vertical": "center", "horizontal": "center"},
                        float(slot.get("size_percent") or 80)))
    targets += [(_inner(b), {"vertical": "center", "horizontal": "center"}, 85.0) for b in empty_boxes]
    for index, (area, anchor, size) in enumerate(targets):
        if not spare:
            notes.append(f"no relevant asset: {len(targets)} image slot(s) left empty")
            break
        asset = spare[index % len(spare)]
        images.append({"id": f"image-{index + 1}", "asset": asset,
                       "rect": _fit(area, _aspect(assets, asset), size, anchor)})
    if empty_boxes and spare:
        notes.append(f"illustration added to {len(empty_boxes)} empty box(es)")
    return {"boxes": boxes, "text_boxes": text_boxes, "images": images, "notes": notes}


def block_lines(template: dict[str, Any]) -> list[int]:
    """Lines of text each text block holds (about one per 90px of height, 1-4)."""

    return [max(1, min(4, int(block["area"]["rect"]["height"] // 90)))
            for block in template.get("text_blocks") or []]


def box_word_limit(box: dict[str, Any], has_icon: bool) -> int:
    """How many words of detail fit the box at a comfortable size."""

    inner = _inner(box)
    width = inner["width"] * (0.62 if has_icon else 1.0)
    capacity = (width / 17) * (inner["height"] / 44) / 6.5  # chars per line * lines / chars per word
    return int(min(max(capacity * 0.55 - 3, 4), 16))


# --------------------------------------------------------------------------- plan and text spec


def build_plan(template: dict[str, Any], edited: dict[str, Any], palette: dict[str, Any],
               assets: dict[str, dict[str, Any]], topic: str) -> dict[str, Any]:
    canvas = template["canvas"]
    width, height = float(canvas["width"]), float(canvas["height"])
    background_asset = canvas.get("background_asset") or _find_asset_by_keywords(
        assets, ("background", "canvas", "backdrop"))
    background, boxes = palette["background"], palette["boxes"]

    def fill(colors: list[str]) -> dict[str, Any]:
        style: dict[str, Any] = {"fill": colors[0], "opacity": 1.0}
        if len(colors) > 1:
            style["gradient"] = {"type": "linear", "colors": colors[:2], "direction": "vertical"}
        return style

    elements: list[dict[str, Any]] = []
    if background_asset in assets:
        elements.append({"id": "background", "asset": background_asset, "x": 0.0, "y": 0.0,
                         "width": width, "height": height, "z_index": -1000, "rotation": 0.0,
                         "preserve_aspect_ratio": False, "style": {**fill(background), "stroke": "none"}})
    for index, box in enumerate(edited["boxes"], 1):
        r = box["rect"]
        elements.append({"id": f"box-{index}", "asset": box["asset"], "x": r["x"], "y": r["y"],
                         "width": r["width"], "height": r["height"], "z_index": index, "rotation": 0.0,
                         "preserve_aspect_ratio": False,
                         "style": {**fill(boxes), "stroke": palette["border"], "stroke_width": 3.0}})
    for index, image in enumerate(edited["images"], 1):
        r = image["rect"]
        style = {"fill": palette["accent"]} if assets[image["asset"]].get("recolorable") else {}
        elements.append({"id": image["id"], "asset": image["asset"], "x": r["x"], "y": r["y"],
                         "width": r["width"], "height": r["height"], "z_index": 100 + index,
                         "rotation": 0.0, "preserve_aspect_ratio": True, "style": style})

    grid = template.get("grid") or {}
    band = grid.get("heading_band")
    return {
        "schema_version": "1.0",
        "template_name": template["name"],
        "description": f"Template '{template['name']}' for {topic}",
        "canvas": {"width": width, "height": height, "background_color": background[0]},
        "layout_grid": {"margin": float(grid.get("margin") or 40), "gutter": float(grid.get("gutter") or 24),
                        "header": [band["x"], band["y"], band["width"], band["height"]] if band else None},
        "elements": elements,
        "metadata": {"user_prompt": topic, "planner": "template", "template": template["name"],
                     "palette": palette, "edits": edited["notes"]},
    }


def _area_placement(area: dict[str, Any] | None, default: str) -> str:
    """The text stage's placement for a template area: the band is 'top', others 'x,y,width'."""

    if not area:
        return default
    if area.get("type") == "band":
        return "top"
    r = area["rect"]
    return f"{r['x']:.0f},{r['y']:.0f},{r['width']:.0f}"


def _anchor_words(text: dict[str, Any]) -> str:
    vertical, horizontal = text.get("vertical", "center"), text.get("horizontal", "center")
    words = [w for w in (vertical, horizontal) if w != "center"]
    return " ".join(words) or "center"


def build_text_spec(template: dict[str, Any], edited: dict[str, Any], content: dict[str, Any],
                    palette: dict[str, Any], fonts: dict[str, str]) -> dict[str, Any]:
    heading_t, sub_t = template.get("heading"), template.get("subheading")
    items: list[dict[str, Any]] = []

    def style_for(entry: dict[str, Any] | None) -> str:
        align = (entry or {}).get("align")
        return align if align in ("left", "right") else ""

    items.append({"id": "heading", "role": "heading", "label": "Heading", "content": content["heading"],
                  "placement": _area_placement((heading_t or {}).get("area"), "top"),
                  "color": palette["heading"], "font": (heading_t or {}).get("font") or fonts["heading"],
                  "style": style_for(heading_t)})
    if content.get("subheading"):
        sub_area = (sub_t or {}).get("area")
        if sub_area and heading_t and sub_area.get("rect") == heading_t["area"].get("rect"):
            sub_area = None  # Sharing the heading's area: stack it under the heading instead.
        items.append({"id": "subheading", "role": "subheading", "label": "Subheading",
                      "content": content["subheading"],
                      "placement": _area_placement(sub_area, "below heading"),
                      "color": palette["subheading"],
                      "font": (sub_t or {}).get("font") or fonts["heading"], "style": style_for(sub_t)})
    # Box order matches the text stage's labels (box-1 = top-left-most).
    for index, box in enumerate(edited["boxes"], 1):
        if not box.get("text"):
            continue
        position = sum(1 for b in edited["text_boxes"] if (b["rect"]["y"], b["rect"]["x"]) <
                       (box["rect"]["y"], box["rect"]["x"]))
        if position >= len(content["points"]):
            continue
        point = content["points"][position]
        label = f"box-{index}"
        items.append({"id": f"{label}-text", "role": "box", "label": f"Text in {label}",
                      "content": " | ".join(p for p in (point["title"], point["detail"]) if p),
                      "placement": _anchor_words(box["text"]), "color": palette["box_text"],
                      "font": box["text"].get("font") or fonts["body"], "style": "", "box": label})
    # The footer and notes fill the text blocks in order, each block as many lines as it holds.
    lines = [line for line in [content.get("footer") or ""] + list(content.get("notes") or []) if line]
    start = 0
    for index, (block, count) in enumerate(zip(template.get("text_blocks") or [], block_lines(template)), 1):
        text = " | ".join(lines[start:start + count])
        start += count
        if not text:
            continue
        items.append({"id": f"text-{index}", "role": "text", "label": f"Text {index}", "content": text,
                      "placement": _area_placement(block.get("area"), "bottom"),
                      "color": palette["footer"], "font": block.get("font") or fonts["body"],
                      "style": style_for(block)})
    return {"default_color": palette["box_text"], "default_font": fonts["body"], "items": items}
