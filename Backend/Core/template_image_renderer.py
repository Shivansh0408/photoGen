#!/usr/bin/env python3
"""Render a validated template plan using only its generated local SVG assets.

The renderer is deliberately deterministic. It does not connect to Ollama and
does not use an AI image-generation model. It composes the styled SVG copies
listed in ``template_plan.json`` into one self-contained SVG image.

When no plan path is supplied, the newest plan under ``gen_assets`` is used::

    python Backend/Core/template_image_renderer.py

An explicit plan can also be rendered::

    python Backend/Core/template_image_renderer.py \
        gen_assets/my-template_20260101_120000/template_plan.json

Use ``--png`` to additionally rasterize the SVG with optional CairoSVG.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "gen_assets"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NAMESPACE)
ET.register_namespace("xlink", XLINK_NAMESPACE)


class RendererError(RuntimeError):
    """Raised when a template plan cannot be rendered safely."""


def _svg_tag(name: str) -> str:
    return f"{{{SVG_NAMESPACE}}}{name}"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RendererError(f"{field} must be a number.") from exc
    if not math.isfinite(number):
        raise RendererError(f"{field} must be finite.")
    if minimum is not None and number < minimum:
        raise RendererError(f"{field} must be at least {minimum}.")
    return number


def _safe_color(value: Any) -> str:
    color = str(value or "#FFFFFF").strip()
    if color.lower() in {"none", "transparent", "currentcolor"}:
        return color.lower()
    if re.fullmatch(
        r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})",
        color,
    ):
        return color
    if re.fullmatch(r"[A-Za-z]+", color):
        return color
    if re.fullmatch(r"(?i)(?:rgb|rgba|hsl|hsla)\([0-9.,%+\-\s]+\)", color):
        return color
    raise RendererError("canvas.background_color is not a safe CSS color.")


def _safe_identifier(value: Any, fallback: str) -> str:
    identifier = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "")).strip("-")
    return identifier[:80] or fallback


def _ensure_asset_is_local_svg(svg_bytes: bytes, asset_path: Path) -> None:
    """Validate SVG syntax and reject references to external resources."""

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise RendererError(f"Generated asset is not valid SVG: {asset_path}") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise RendererError(f"Generated asset has no SVG root: {asset_path}")

    for element in root.iter():
        for attribute_name, attribute_value in element.attrib.items():
            lowered_name = attribute_name.lower()
            lowered_value = str(attribute_value).strip().lower()
            if lowered_name.endswith("href") and lowered_value.startswith(
                ("http:", "https:", "file:", "//")
            ):
                raise RendererError(f"Asset contains an external reference: {asset_path}")
            if "url(" in lowered_value:
                references = re.findall(r"url\((.*?)\)", lowered_value, flags=re.IGNORECASE)
                if any(not reference.strip(" \"'").startswith("#") for reference in references):
                    raise RendererError(
                        f"Asset contains an external CSS reference: {asset_path}"
                    )


def find_latest_plan(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Find the most recently modified generated template plan."""

    output_root = output_root.expanduser().resolve()
    plans = [path for path in output_root.glob("*/template_plan.json") if path.is_file()]
    if not plans:
        raise RendererError(f"No template plans were found under: {output_root}")
    return max(plans, key=lambda path: path.stat().st_mtime)


def load_plan(plan_path: Path) -> dict[str, Any]:
    """Load a JSON render plan from disk."""

    plan_path = plan_path.expanduser().resolve()
    if not plan_path.is_file():
        raise RendererError(f"Template plan was not found: {plan_path}")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RendererError(f"Could not read the template plan: {exc}") from exc
    if not isinstance(plan, dict):
        raise RendererError("The template plan root must be a JSON object.")
    return plan


def _resolve_generated_asset(run_directory: Path, relative_path: Any) -> Path:
    """Resolve an asset while preventing path traversal outside its run folder."""

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise RendererError("An element is missing generated_asset.")
    candidate = (run_directory / relative_path).resolve()
    try:
        candidate.relative_to(run_directory.resolve())
    except ValueError as exc:
        raise RendererError("A generated_asset path leaves its run directory.") from exc
    if not candidate.is_file() or candidate.suffix.lower() != ".svg":
        raise RendererError(f"Generated SVG asset was not found: {candidate}")
    return candidate


FONT_DIRECTORY = PROJECT_ROOT / "Schema" / "font_dictnoary"
DEFAULT_FONT_FAMILY = "Arial, DejaVu Sans, sans-serif"
DEFAULT_LINE_HEIGHT = 1.28
FONT_MIME_TYPES = {".ttf": "font/ttf", ".otf": "font/otf", ".woff": "font/woff", ".woff2": "font/woff2"}
MAX_FONT_BYTES = 5 * 1024 * 1024


def _add_text_defs(root: ET.Element, *, wants_shadow: bool, embedded_fonts: dict[str, str]) -> None:
    """Shared text effects and embedded font files, placed before the drawing."""

    defs = ET.Element(_svg_tag("defs"))
    if wants_shadow:
        text_filter = ET.SubElement(defs, _svg_tag("filter"), {
            "id": "text-shadow", "x": "-10%", "y": "-10%", "width": "120%", "height": "140%",
        })
        ET.SubElement(text_filter, _svg_tag("feDropShadow"), {
            "dx": "0", "dy": "3", "stdDeviation": "3",
            "flood-color": "#000000", "flood-opacity": "0.35",
        })
    rules = []
    for relative, family in embedded_fonts.items():
        # Only files inside the font dictionary folder are embedded, keeping rendering local.
        candidate = (FONT_DIRECTORY / "fonts" / relative).resolve()
        try:
            candidate.relative_to(FONT_DIRECTORY.resolve())
        except ValueError:
            print(f"[Renderer] Skipping font outside the font folder: {relative}", file=sys.stderr)
            continue
        mime = FONT_MIME_TYPES.get(candidate.suffix.lower())
        if not mime or not candidate.is_file() or candidate.stat().st_size > MAX_FONT_BYTES:
            print(f"[Renderer] Font file not usable, using installed fonts: {relative}", file=sys.stderr)
            continue
        name = family.split(",")[0].strip().strip("'\"")
        data = base64.b64encode(candidate.read_bytes()).decode("ascii")
        rules.append(f"@font-face{{font-family:'{name}';src:url(data:{mime};base64,{data});}}")
    if rules:
        ET.SubElement(defs, _svg_tag("style")).text = "".join(rules)
    if len(defs):
        root.insert(3, defs)  # After title, desc and metadata.


def compose_svg(plan: dict[str, Any], plan_path: Path) -> bytes:
    """Compose a self-contained SVG from local generated assets."""

    canvas = plan.get("canvas")
    if not isinstance(canvas, dict):
        raise RendererError("The plan is missing its canvas object.")
    canvas_width = _finite_number(canvas.get("width"), field="canvas.width", minimum=1)
    canvas_height = _finite_number(canvas.get("height"), field="canvas.height", minimum=1)
    background_color = _safe_color(canvas.get("background_color"))

    elements = plan.get("elements")
    if not isinstance(elements, list) or not elements:
        raise RendererError("The plan must contain at least one element.")
    run_directory = plan_path.resolve().parent

    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": _format_number(canvas_width),
            "height": _format_number(canvas_height),
            "viewBox": f"0 0 {_format_number(canvas_width)} {_format_number(canvas_height)}",
            "role": "img",
            "aria-labelledby": "template-title template-description",
        },
    )
    title = ET.SubElement(root, _svg_tag("title"), {"id": "template-title"})
    title.text = str(plan.get("template_name") or "Generated template")
    description = ET.SubElement(
        root,
        _svg_tag("desc"),
        {"id": "template-description"},
    )
    description.text = str(plan.get("description") or "Template built from local SVG assets")
    metadata = ET.SubElement(root, _svg_tag("metadata"))
    metadata.text = json.dumps(
        {
            "schema_version": plan.get("schema_version"),
            "renderer": "template_image_renderer.py",
            "source_plan": plan_path.name,
            "asset_policy": "local-generated-assets-only",
        },
        separators=(",", ":"),
    )

    ET.SubElement(
        root,
        _svg_tag("rect"),
        {
            "id": "canvas-background",
            "x": "0",
            "y": "0",
            "width": _format_number(canvas_width),
            "height": _format_number(canvas_height),
            "fill": background_color,
        },
    )

    indexed_elements = list(enumerate(elements))
    try:
        indexed_elements.sort(
            key=lambda pair: (float(pair[1].get("z_index", pair[0])), pair[0])
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RendererError("Each plan element must be an object with a numeric z_index.") from exc

    for original_index, element in indexed_elements:
        if not isinstance(element, dict):
            raise RendererError(f"elements[{original_index}] must be an object.")
        element_id = _safe_identifier(element.get("id"), f"element-{original_index + 1}")
        x = _finite_number(element.get("x"), field=f"{element_id}.x")
        y = _finite_number(element.get("y"), field=f"{element_id}.y")
        width = _finite_number(element.get("width"), field=f"{element_id}.width", minimum=0.1)
        height = _finite_number(
            element.get("height"), field=f"{element_id}.height", minimum=0.1
        )
        rotation = _finite_number(element.get("rotation", 0), field=f"{element_id}.rotation")
        preserve_aspect_ratio = bool(element.get("preserve_aspect_ratio", False))

        asset_path = _resolve_generated_asset(run_directory, element.get("generated_asset"))
        try:
            asset_bytes = asset_path.read_bytes()
        except OSError as exc:
            raise RendererError(f"Could not read generated asset: {asset_path}") from exc
        _ensure_asset_is_local_svg(asset_bytes, asset_path)
        encoded_asset = base64.b64encode(asset_bytes).decode("ascii")

        image_attributes = {
            "id": element_id,
            "x": _format_number(x),
            "y": _format_number(y),
            "width": _format_number(width),
            "height": _format_number(height),
            "preserveAspectRatio": "xMidYMid meet" if preserve_aspect_ratio else "none",
            "href": f"data:image/svg+xml;base64,{encoded_asset}",
            f"{{{XLINK_NAMESPACE}}}href": f"data:image/svg+xml;base64,{encoded_asset}",
            "data-source-asset": str(element.get("asset") or ""),
        }
        if rotation % 360:
            center_x = x + (width / 2)
            center_y = y + (height / 2)
            image_attributes["transform"] = (
                f"rotate({_format_number(rotation)} "
                f"{_format_number(center_x)} {_format_number(center_y)})"
            )
        ET.SubElement(root, _svg_tag("image"), image_attributes)

    # Optional text is positioned by template_text_layer.py after planning.
    # Validate it again here because a saved plan can also be rendered directly.
    overlays = plan.get("text_overlays", [])
    if not isinstance(overlays, list) or len(overlays) > 101:
        raise RendererError("text_overlays must be a short array.")
    wants_shadow = False
    embedded_fonts: dict[str, str] = {}  # font file -> CSS family stack
    for index, overlay in enumerate(overlays):
        if not isinstance(overlay, dict):
            raise RendererError(f"text_overlays[{index}] must be an object.")
        overlay_id = _safe_identifier(overlay.get("id"), f"text-{index + 1}")
        x = _finite_number(overlay.get("x"), field=f"{overlay_id}.x")
        y = _finite_number(overlay.get("y"), field=f"{overlay_id}.y")
        width = _finite_number(overlay.get("width"), field=f"{overlay_id}.width", minimum=1)
        height = _finite_number(overlay.get("height"), field=f"{overlay_id}.height", minimum=1)
        font_size = _finite_number(
            overlay.get("font_size"), field=f"{overlay_id}.font_size", minimum=8
        )
        rotation = _finite_number(overlay.get("rotation", 0), field=f"{overlay_id}.rotation")
        if x < 0 or y < 0 or x + width > canvas_width + 0.01 or y + height > canvas_height + 0.01:
            raise RendererError(f"Text {overlay_id} must fit within the canvas.")
        lines = overlay.get("lines")
        if (not isinstance(lines, list) or not 1 <= len(lines) <= 64
                or any(not isinstance(line, str) or len(line) > 500 for line in lines)):
            raise RendererError(f"Text {overlay_id} must have short lines.")
        alignment = overlay.get("align", "center")
        if alignment not in ("left", "center", "right"):
            raise RendererError(f"Text {overlay_id} has an invalid alignment.")
        vertical_alignment = overlay.get("vertical_align", "center")
        if vertical_alignment not in ("top", "center", "bottom"):
            raise RendererError(f"Text {overlay_id} has an invalid vertical alignment.")
        color = _safe_color(overlay.get("color", "#000000"))
        text_x = x if alignment == "left" else x + width if alignment == "right" else x + width / 2
        anchor = {"left": "start", "center": "middle", "right": "end"}[alignment]
        raw_weight = str(overlay.get("weight") or "normal")
        weight = raw_weight if raw_weight in ("bold", "normal") or re.fullmatch(
            r"[1-9]00", raw_weight) else "normal"
        # Font styling from the font dictionary; older plans fall back to Arial.
        family = str(overlay.get("font_family") or DEFAULT_FONT_FAMILY)
        if not re.fullmatch(r"[A-Za-z0-9 ,'\"-]{1,200}", family):
            raise RendererError(f"Text {overlay_id} has an unsafe font family.")
        line_height = _finite_number(overlay.get("line_height", DEFAULT_LINE_HEIGHT),
                                     field=f"{overlay_id}.line_height", minimum=0.8)
        line_height = min(line_height, 3.0)
        letter_spacing = _finite_number(overlay.get("letter_spacing", 0),
                                        field=f"{overlay_id}.letter_spacing")
        if overlay.get("font_file"):
            embedded_fonts.setdefault(str(overlay["font_file"]), family)
        line_block_height = len(lines) * font_size * line_height
        top_offset = (0 if vertical_alignment == "top" else
                      height - line_block_height if vertical_alignment == "bottom" else
                      (height - line_block_height) / 2)
        group = ET.SubElement(root, _svg_tag("g"), {"id": overlay_id})
        if rotation % 360:
            group.set("transform", f"rotate({_format_number(rotation)} "
                      f"{_format_number(x + width / 2)} {_format_number(y + height / 2)})")
        if overlay.get("shadow"):
            group.set("filter", "url(#text-shadow)")
            wants_shadow = True
        text_attributes = {
            "x": _format_number(text_x),
            # Baseline: half the extra leading plus the ascent (0.84 at line height 1.28).
            "y": _format_number(y + top_offset + font_size * (0.84 + (line_height - 1.28) / 2)),
            "fill": color,
            "font-family": family,
            "font-size": _format_number(font_size),
            "font-weight": weight,
            "text-anchor": anchor,
            "xml:space": "preserve",
        }
        if abs(letter_spacing) > 0.01:
            text_attributes["letter-spacing"] = _format_number(round(letter_spacing, 2))
        text_element = ET.SubElement(group, _svg_tag("text"), text_attributes)
        for line_index, line in enumerate(lines):
            tspan = ET.SubElement(text_element, _svg_tag("tspan"), {
                "x": _format_number(text_x),
                "dy": "0" if line_index == 0 else _format_number(font_size * line_height),
            })
            tspan.text = line

    if wants_shadow or embedded_fonts:
        _add_text_defs(root, wants_shadow=wants_shadow, embedded_fonts=embedded_fonts)

    xml_body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_body + b"\n"


def render_png(svg_bytes: bytes, png_path: Path) -> Path:
    """Optionally rasterize with CairoSVG; this remains a non-AI render step."""

    try:
        import cairosvg  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise RendererError(
            "PNG output needs CairoSVG and its native Cairo library. "
            "Without Cairo, omit --png and use the generated SVG image."
        ) from exc

    png_path = png_path.expanduser().resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cairosvg.svg2png(bytestring=svg_bytes, write_to=str(png_path))
    except Exception as exc:
        raise RendererError(f"CairoSVG could not create the PNG: {exc}") from exc
    return png_path


def render_template(
    plan_path: Path,
    *,
    output_path: Path | None = None,
    png_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Render one plan and return the SVG path plus optional PNG path."""

    plan_path = plan_path.expanduser().resolve()
    plan = load_plan(plan_path)
    svg_bytes = compose_svg(plan, plan_path)

    final_svg_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else plan_path.parent / "generated_image.svg"
    )
    final_svg_path.parent.mkdir(parents=True, exist_ok=True)
    final_svg_path.write_bytes(svg_bytes)

    rendered_png: Path | None = None
    if png_path is not None:
        rendered_png = render_png(svg_bytes, png_path)
    save_timestamped_png(final_svg_path, svg_bytes, plan)
    return final_svg_path, rendered_png


# --------------------------------------------------------------------------- timestamped PNG copy

BROWSER_CANDIDATES = (
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
)


def timestamped_png_path(directory: Path, moment: datetime | None = None) -> Path:
    """HHMMDDMM.png: an image made at 17:10 on 29 September is 17102909.png."""

    stem = (moment or datetime.now()).strftime("%H%M%d%m")
    path = directory / f"{stem}.png"
    number = 2
    while path.exists():  # Two images in the same minute: 17102909_2.png.
        path = directory / f"{stem}_{number}.png"
        number += 1
    return path


def _find_browser() -> Path | None:
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return next((path for path in BROWSER_CANDIDATES if path.is_file()), None)


def _browser_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    """Rasterize with a headless Edge/Chrome screenshot (used when Cairo is missing)."""

    browser = _find_browser()
    if browser is None:
        raise RendererError("no CairoSVG and no Edge/Chrome browser found")
    with tempfile.TemporaryDirectory() as profile:  # Never touches the user's browser profile.
        subprocess.run(
            [str(browser), "--headless", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=1", f"--user-data-dir={profile}",
             f"--window-size={width},{height}", f"--screenshot={png_path}", svg_path.as_uri()],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90, check=False,
        )
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RendererError(f"{browser.name} did not produce a PNG")


def save_timestamped_png(svg_path: Path, svg_bytes: bytes, plan: dict[str, Any]) -> Path | None:
    """Store a PNG of the final SVG beside it, named by generation time.

    Best effort: a failure is reported and the SVG result is kept unchanged.
    """

    png_path = timestamped_png_path(svg_path.parent)
    try:
        try:
            render_png(svg_bytes, png_path)
        except RendererError:
            canvas = plan.get("canvas") or {}
            width = math.ceil(_finite_number(canvas.get("width"), field="canvas.width", minimum=1))
            height = math.ceil(_finite_number(canvas.get("height"), field="canvas.height", minimum=1))
            _browser_png(svg_path, png_path, width, height)
    except (RendererError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[Renderer] Timestamped PNG not saved ({exc}); the SVG is unaffected.")
        return None
    print(f"[Renderer] Timestamped PNG saved: {png_path}")
    return png_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse renderer command-line options."""

    parser = argparse.ArgumentParser(
        description="Compose a final image from a generated template plan and local SVG assets."
    )
    parser.add_argument(
        "plan",
        nargs="?",
        type=Path,
        help="Path to template_plan.json. Defaults to the newest generated plan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Final SVG path. Defaults to generated_image.svg beside the plan.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also generate generated_image.png using optional CairoSVG.",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        help="Custom PNG path; implies --png.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root searched when no explicit plan path is supplied.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the standalone deterministic renderer."""

    arguments = parse_arguments(argv)
    try:
        plan_path = arguments.plan or find_latest_plan(arguments.output_root)
        png_path: Path | None = None
        if arguments.png_output is not None:
            png_path = arguments.png_output
        elif arguments.png:
            png_path = plan_path.expanduser().resolve().parent / "generated_image.png"

        svg_path, rendered_png = render_template(
            plan_path,
            output_path=arguments.output,
            png_path=png_path,
        )
        print(f"[Renderer] Source plan: {plan_path.expanduser().resolve()}")
        print(f"[Renderer] SVG image: {svg_path}")
        if rendered_png is not None:
            print(f"[Renderer] PNG image: {rendered_png}")
        return 0
    except (RendererError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
