"""Motivational vector pipeline for the bulk generator (choice 2 in ../user_request.py).

Each image is a scenic background from Assets/backgrounds (category 'backgrounds'
in svg_dictionary.py), one person from Assets/Humans (category 'humans') and a
short quote. For a topic and a number of images:

1. Scenes and people are ranked once per batch by how well their usage text fits
   the topic (plus Ollama's related words, asked without showing the catalog).
   Each person is paired with the background that suits them best and has not been
   used recently, so the images vary.
2. Every background is analysed from its SVG: the focal sun or moon (a <circle>),
   the sky colours of the open upper area and the ground line. The person stands
   on the ground on the side away from the focal point, sized by pose (standing
   figures taller, seated ones wider); the quote goes in the upper area, clear of
   the focal point, in a colour that contrasts with the sky there.
3. The person's clothing is recoloured at random (deep tint, vivid, pastel or a
   monotone vector look) in a hue that harmonises with the background; skin and
   hair keep their colours except in the monotone style.
4. ONE Ollama call writes the text: an original quote of at most 12 words and a
   short tagline, different from earlier images.

The rendering engine (plan assets, text stage, renderer) is shared with the
Infographic pipeline and imported from ../Infographic.
"""

from __future__ import annotations

import argparse
import ast
import colorsys
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Infographic"))

from ollama_template_planner import (  # noqa: E402
    GENERIC_ASSET_WORDS,
    PlannerError,
    _asset_words,
    _word_root,
    create_run_directory,
    load_svg_dictionary,
    materialize_plan_assets,
)
from photogen import ThinkingEngine, _clean_words, _luminance  # noqa: E402
from template_image_renderer import RendererError, render_template  # noqa: E402
from template_text_layer import TextLayerError, apply_text_spec, load_font_dictionary  # noqa: E402

QUOTE_WORDS, TAGLINE_WORDS = 12, 5
DANGLING_WORDS = {"a", "an", "the", "your", "my", "our", "of", "to", "on", "in", "for", "and", "or", "with", "at"}
# Display fonts for the quote; the tagline uses a calmer body font.
QUOTE_FONTS = ("Bold Impact", "Condensed Display", "Heavy Black", "Elegant Serif", "Classic Serif",
               "Warm Serif", "Light Minimal", "Geometric Modern", "Newsroom Gothic")
TAGLINE_FONTS = ("Light Minimal", "Corporate Sans", "Soft Sans", "Humanist Rounded", "Geometric Modern")
RECOLOR_STYLES = ("deep", "vivid", "pastel", "monotone")
# Skin, lips, hair and leather sit between deep red (340 degrees, wrapping round) and yellow (50).
SKIN_HUE_MIN, SKIN_HUE_MAX = 340 / 360, 50 / 360
# General words in a theme that say little about who or where ('the power of', 'study hard').
FILLER_THEME_WORDS = {_word_root(w) for w in (
    "power", "hard", "never", "give", "keep", "make", "life", "day", "days", "time", "best", "great",
    "way", "ways", "things", "always", "every", "good", "better", "new", "your", "own", "real")}
# Usage words of scenes that suit any motivational theme.
GENERAL_SCENE_WORDS = {_word_root(w) for w in ("motivation", "goals", "success", "progress", "growth",
                                                "ambition", "perseverance", "aspiration", "achievement",
                                                "journeys")}
IMAGE_ERRORS = (PlannerError, TextLayerError, RendererError, OSError, ValueError, KeyError)


# --------------------------------------------------------------------------- colour helpers


def _rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def _hls(color: str) -> tuple[float, float, float]:
    return colorsys.rgb_to_hls(*_rgb(color))


def _hex(hue: float, lightness: float, saturation: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue % 1.0, min(max(lightness, 0), 1), min(max(saturation, 0), 1))
    return "#{:02X}{:02X}{:02X}".format(round(red * 255), round(green * 255), round(blue * 255))


# --------------------------------------------------------------------------- background analysis


def analyse_background(xml: str, width: float, height: float) -> dict[str, Any]:
    """Focal point, sky colours of the upper area and ground line, read from the SVG."""

    stops = [(float(offset), color.upper()) for offset, color in re.findall(
        r'<stop[^>]*offset="([\d.]+)"[^>]*stop-color="(#[0-9A-Fa-f]{6})"', xml)]
    # The first gradient is the sky; its stops over the top half colour the quote area.
    sky = []
    for offset, color in stops:
        if sky and offset <= sky[-1][0]:
            break  # The next gradient starts.
        sky.append((offset, color))
    upper = [color for offset, color in sky if offset <= 0.55] or [c for _, c in sky[:1]] or ["#808080"]
    focal = None
    for cx, cy, r, fill in re.findall(
            r'<circle[^>]*cx="([\d.]+)"[^>]*cy="([\d.]+)"[^>]*r="([\d.]+)"[^>]*fill="(#[0-9A-Fa-f]{6})"', xml):
        if float(r) >= 30 and (focal is None or float(r) > focal["r"]):
            focal = {"x": float(cx), "y": float(cy), "r": float(r), "color": fill.upper()}
    ground = height * 0.92
    group = re.search(r'<g id="foreground-ground">\s*<path d="([^"]+)"', xml)
    if group:
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", group.group(1))]
        ys = [y for y in numbers[1::2] if 0.5 * height < y < height]
        if ys:
            ground = sum(ys) / len(ys)
    return {"upper_colors": upper, "upper_light": sum(_luminance(c) for c in upper) / len(upper),
            "sky": [c for _, c in sky], "focal": focal, "ground": ground}


class BackgroundSampler:
    """Measures how light a background really is behind the text.

    The first gradient of an SVG is not always what sits behind the quote (a
    chalkboard or window can cover the wall), so each background is rendered once
    (headless Edge/Chrome, as for the PNG output) and the pixels under the text are
    averaged. Without a browser or Pillow the SVG's gradient estimate is kept.
    """

    def __init__(self, work_root: Path) -> None:
        self.work_root = work_root
        self.images: dict[str, Any] = {}

    def _image(self, filename: str, asset: dict[str, Any]) -> Any:
        if filename not in self.images:
            self.images[filename] = None
            try:
                from PIL import Image  # type: ignore[import-not-found]
                from template_image_renderer import _browser_png

                self.work_root.mkdir(parents=True, exist_ok=True)
                svg = self.work_root / f"sample-{Path(filename).stem}.svg"
                png = svg.with_suffix(".png")
                svg.write_text(asset["xml"], encoding="utf-8")
                dims = asset["dimensions"]
                _browser_png(svg, png, round(float(dims["width"])), round(float(dims["height"])))
                with Image.open(png) as picture:
                    self.images[filename] = picture.convert("L").copy()
                svg.unlink(missing_ok=True)
                png.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - any failure keeps the SVG estimate.
                print(f"[Motivation] Could not sample {Path(filename).stem} ({exc}); "
                      "using its gradient colours.")
        return self.images[filename]

    def region(self, filename: str, asset: dict[str, Any],
               box: tuple[float, float, float, float]) -> tuple[float, float] | None:
        """(mean lightness 0-1, clash 0-0.5) behind text in ``box``, or None.

        Clash is the share of pixels on the minority side of light/dark: the part of
        the area where one text colour cannot be read. Centred text rarely reaches
        the box edges, so only the middle 80% of its width is measured.
        """

        picture = self._image(filename, asset)
        if picture is None:
            return None
        x, y, w, h = box
        x, w = x + w * 0.1, w * 0.8
        crop = picture.crop((max(round(x), 0), max(round(y), 0),
                             min(round(x + w), picture.width), min(round(y + h), picture.height)))
        values = list(crop.resize((max(1, crop.width // 8), max(1, crop.height // 8))).getdata())
        if not values:
            return None
        mean = sum(values) / len(values) / 255
        light_share = sum(1 for v in values if v > 140) / len(values)
        return mean, min(light_share, 1 - light_share)

    def even_span(self, filename: str, asset: dict[str, Any], box: tuple[float, float, float, float],
                  dark: bool) -> tuple[float, float] | None:
        """(x, width) of the widest run of columns in ``box`` that are dark (or light).

        Used when a quote still straddles two areas: it narrows to the part where its
        colour reads, e.g. exactly the chalkboard beside a pale window.
        """

        picture = self._image(filename, asset)
        if picture is None:
            return None
        x, y, w, h = (round(v) for v in box)
        crop = picture.crop((max(x, 0), max(y, 0), min(x + w, picture.width), min(y + h, picture.height)))
        step = 8
        small = crop.resize((max(1, crop.width // step), 1))  # One mean lightness per column strip.
        columns = [(v > 140) != dark for v in small.getdata()]  # True where the text colour reads.
        best, start = (0, 0), None
        for index, ok in enumerate(columns + [False]):
            if ok and start is None:
                start = index
            elif not ok and start is not None:
                if index - start > best[1] - best[0]:
                    best = (start, index)
                start = None
        if best[1] - best[0] < len(columns) * 0.5:
            return None  # No clear area wide enough; keep the full width.
        scale = crop.width / max(1, len(columns))
        return x + best[0] * scale, (best[1] - best[0]) * scale


# --------------------------------------------------------------------------- recolouring people


def recolor_person(xml: str, style: str, hue: float, rng: random.Random) -> str:
    """Recolour clothing (greys and non-skin colours) in ``hue``; skin and hair stay unless monotone."""

    saturation = {"deep": 0.45, "vivid": 0.70, "pastel": 0.40, "monotone": 0.35}[style]
    mapping: dict[str, str] = {}
    for color in set(c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}\b", xml)):
        h, lightness, s = _hls(color)
        if style == "monotone":
            # A one-colour vector look: every tone mapped onto one hue, shading kept.
            mapping[color] = _hex(hue, 0.12 + lightness * 0.8, saturation)
            continue
        if lightness < 0.08 or lightness > 0.92:
            continue  # Deep shadows and highlights keep their colours.
        if s > 0.18 and (h <= SKIN_HUE_MAX or h >= SKIN_HUE_MIN):
            continue  # Skin, hair, leather and wood tones (reds to yellows) stay natural.
        # Neutral greys and clearly coloured clothing (greens, teals, blues, purples) change.
        if style == "deep":
            new_l = lightness
        elif style == "vivid":
            new_l = 0.25 + lightness * 0.6
        else:  # pastel
            new_l = 0.55 + lightness * 0.35
        mapping[color] = _hex(hue + rng.uniform(-0.01, 0.01), new_l, saturation)
    if not mapping:
        return xml
    pattern = re.compile("|".join(re.escape(c) for c in mapping), re.I)
    return pattern.sub(lambda m: mapping[m.group(0).upper()], xml)


# --------------------------------------------------------------------------- scene selection


def _words(text: str) -> set[str]:
    return {_word_root(w) for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in GENERIC_ASSET_WORDS}


# Word endings; 'nes' and 'fulnes' are 'ness' after _word_root drops the final 's'.
WORD_ENDINGS = ("s", "es", "ed", "ing", "ness", "nes", "ful", "fulness", "fulnes", "ity", "ion", "ation",
                "ly", "ment", "al", "en")


def _related(a: str, b: str) -> bool:
    """calm~calmness, peace~peaceful, meditation~meditating; not care~career or busy~business."""

    short, long = sorted((a, b), key=len)
    if len(short) >= 4 and long.startswith(short) and (long == short or long[len(short):] in WORD_ENDINGS):
        return True
    # Two forms of one stem: meditation / meditating, serene / serenity.
    return len(short) >= 6 and short[:6] == long[:6] or (len(short) >= 5 and short[:5] == long[:5]
                                                            and abs(len(a) - len(b)) <= 3)


def _overlap(first: set[str], second: set[str]) -> int:
    """How many words of ``first`` have a related word in ``second``."""

    return sum(1 for word in first if any(_related(word, other) for other in second))


class SceneChooser:
    """Pairs people with backgrounds that fit the topic and each other, varying per image."""

    def __init__(self, assets: dict[str, dict[str, Any]], categories: dict[str, str],
                 topic: str, engine: ThinkingEngine, rng: random.Random) -> None:
        self.assets, self.rng = assets, rng
        self.backgrounds = [f for f, c in categories.items() if c == "backgrounds"]
        self.people = [f for f, c in categories.items() if c == "humans"]
        if not self.backgrounds or not self.people:
            raise ValueError("svg_dictionary.py needs assets with category 'backgrounds' and 'humans' "
                             "(map Assets/backgrounds and Assets/Humans with schema_creator.py).")
        # 'the power of education': 'education' is the subject; 'power' is filler that
        # would otherwise pull in the karate and deadlift figures.
        topic_words = _words(topic) - FILLER_THEME_WORDS
        raw = engine._ask_ollama(
            "List up to 8 single-word nouns or adjectives for scenes, activities and moods that fit "
            f"a motivational poster about this theme.\nTheme: {topic[:300]}\n"
            'Return ONLY this JSON object: {"keywords": ["..."]}', "Theme keywords") or {}
        keywords = [str(k) for k in raw.get("keywords") or [] if isinstance(k, str)][:8]
        self.theme = topic_words | _words(" ".join(keywords))
        # Filler words still count, but only as weak (theme-level) matches.
        self.theme |= _words(topic) & FILLER_THEME_WORDS
        if keywords:
            print(f"[Motivation] Theme keywords: {', '.join(keywords)}")
        self.topic_words = topic_words

        def score(filename: str) -> float:
            words = _asset_words(filename, assets[filename])
            return 2 * _overlap(topic_words, words) + _overlap(self.theme, words)

        self.score = {f: score(f) for f in self.backgrounds + self.people}
        ranked = sorted(self.people, key=lambda f: (-self.score[f], rng.random()))
        relevant = [f for f in ranked if self.score[f] > 0]
        # Only clear matches rotate (at least half the best score); a weak one-word match
        # ('power' for the karate figure) waits until no strong match exists.
        best = max((self.score[f] for f in relevant), default=0)
        strong = [f for f in relevant if self.score[f] >= best / 2]
        self.people_order = strong or ranked
        for f in sorted(self.backgrounds + self.people, key=lambda f: -self.score[f]):
            if self.score[f] > 0:
                print(f"    {Path(f).stem:<30} fits the theme (score {self.score[f]:g})")
        if not relevant:
            print("[Motivation] No person matches the theme directly; rotating through all of them.")
        self.uses = {f: 0 for f in self.backgrounds}
        self.pairs: dict[tuple[str, str], int] = {}  # (person, background) -> times used

    def pick(self, index: int) -> tuple[str, str]:
        person = self.people_order[index % len(self.people_order)]
        person_words = _asset_words(person, self.assets[person])

        def fit(background: str) -> float:
            words = _asset_words(background, self.assets[background])
            return _overlap(person_words, words) + self.score[background]

        # Scenes that suit this person or theme, plus the general motivational scenes;
        # the least-used one wins so a batch walks through them instead of repeating.
        suitable = [b for b in self.backgrounds if fit(b) > 0
                    or _overlap(GENERAL_SCENE_WORDS, _asset_words(b, self.assets[b]))] or self.backgrounds
        # Bias against repeats: this person is never shown on the same scene twice while
        # an unused pairing exists. Once every suitable scene has had this person, the
        # other scenes are tried before any pairing repeats.
        if all(self.pairs.get((person, b)) for b in suitable):
            fresh = [b for b in self.backgrounds if not self.pairs.get((person, b))]
            if fresh:
                suitable = fresh
        background = min(suitable, key=lambda b: (self.pairs.get((person, b), 0), self.uses[b],
                                                   -fit(b), self.rng.random()))
        self.uses[background] += 1
        self.pairs[(person, background)] = self.pairs.get((person, background), 0) + 1
        return background, person


# --------------------------------------------------------------------------- composition


def compose(background: str, person: str, assets: dict[str, dict[str, Any]],
            rng: random.Random, sampler: BackgroundSampler | None = None) -> dict[str, Any]:
    """Where the person and the quote go, from the background's own layout."""

    dims = assets[background]["dimensions"]
    width, height = float(dims["width"]), float(dims["height"])
    scene = analyse_background(assets[background]["xml"], width, height)
    focal = scene["focal"]

    pdims = assets[person]["dimensions"]
    aspect = float(pdims["width"]) / float(pdims["height"])
    # Standing figures are tall; seated or wide ones are drawn smaller.
    person_h = height * (rng.uniform(0.40, 0.46) if aspect > 0.75 else rng.uniform(0.52, 0.62))
    person_w = min(person_h * aspect, width * 0.62)
    person_h = person_w / aspect

    # The person goes to the side away from the focal point (sun, moon, doorway light).
    if focal and abs(focal["x"] - width / 2) > width * 0.08:
        side = "left" if focal["x"] > width / 2 else "right"
    else:
        side = rng.choice(("left", "right", "center")) if aspect > 0.75 else rng.choice(("left", "right"))
    margin = width * 0.06
    x = {"left": margin, "right": width - margin - person_w, "center": (width - person_w) / 2}[side]
    # Feet on the ground line, a little in front of it.
    bottom = min(height - height * 0.02, scene["ground"] + (height - scene["ground"]) * 0.55)
    y = bottom - person_h

    # The quote sits in the open upper area, clear of the focal point.
    text_top = height * 0.07
    text_w = width * 0.84
    text_x = (width - text_w) / 2
    align = "center"
    if focal and focal["y"] - focal["r"] < height * 0.42 and abs(focal["x"] - width / 2) > width * 0.12:
        # A high, off-centre sun: the quote takes the other half and aligns to its edge.
        text_w = width * 0.56
        if focal["x"] > width / 2:
            text_x, align = width * 0.07, "left"
        else:
            text_x, align = width - width * 0.07 - text_w, "right"

    lightness, spread = scene["upper_light"], 0.0
    text_h = height * 0.30  # Room for the quote and the tagline.
    measured = sampler.region(background, assets[background],
                              (text_x, text_top, text_w, text_h)) if sampler else None
    if measured and measured[1] > 0.10:
        # A mixed area (wall above a chalkboard): slide the quote down to where one text
        # colour reads best, staying in the upper area and clear of the person's head.
        lowest = min(height * 0.25, y - text_h * 0.5)  # A quote block is rarely the full 30%.
        best = measured
        for top in (height * f for f in (0.10, 0.13, 0.16, 0.19, 0.22, 0.25)):
            if top > lowest:
                break
            option = sampler.region(background, assets[background], (text_x, top, text_w, text_h))
            if option and option[1] < best[1] - 0.03:
                text_top, best = top, option
        measured = best
        if measured[1] > 0.05 and align == "center":
            # Still straddling (a board beside a window): narrow to the even columns.
            span = sampler.even_span(background, assets[background],
                                     (text_x, text_top, text_w, text_h * 0.6), dark=measured[0] < 0.55)
            if span:
                inset = width * 0.03
                text_x, text_w = span[0] + inset, span[1] - 2 * inset
                measured = sampler.region(background, assets[background],
                                          (text_x, text_top, text_w, text_h)) or measured
    if measured:
        lightness, spread = measured
    light = lightness >= 0.55
    if light:
        text_color = _hex(_hls(scene["sky"][-1])[0], 0.16, 0.35) if scene["sky"] else "#1A1A1A"
    else:
        text_color = "#FFFFFF"
    # The sun or moon colour makes a warm accent, but only where the sky is truly dark.
    dark_sky = lightness < 0.35
    accent = focal["color"] if focal and dark_sky and _luminance(focal["color"]) > 0.6 else text_color
    # Soft shadow when the area is mid-toned or mixed (a board edge, busy scenery).
    shadow = 0.35 < lightness < 0.65 or spread > 0.08  # spread = clash share when measured
    return {"width": width, "height": height, "scene": scene,
            "person": {"x": round(x, 1), "y": round(y, 1), "width": round(person_w, 1),
                       "height": round(person_h, 1), "side": side},
            "text": {"x": round(text_x), "y": round(text_top), "width": round(text_w), "align": align,
                     "color": text_color, "accent": accent, "shadow": shadow,
                     # Moved down to an even spot: one size smaller so it clears the person.
                     "compact": text_top > height * 0.08}}


def write_quote(engine: ThinkingEngine, topic: str, earlier: list[str]) -> dict[str, str]:
    """The one Ollama call per image: an original short quote and tagline."""

    raw = engine._ask_ollama("\n".join(filter(None, [
        "Write ONE original motivational quote for a poster. Plain words, no quotation marks, "
        "no author, no hashtags or emojis.",
        f"Theme: {topic[:300]}",
        f"The quote has at most {QUOTE_WORDS} words; the tagline at most {TAGLINE_WORDS} words.",
        f"Do not repeat or paraphrase these earlier quotes: {' / '.join(earlier)}" if earlier else "",
        'Return ONLY this JSON object: {"quote": "...", "tagline": "..."}',
    ])), "Quote") or {}
    quote = _clean_words(raw.get("quote"), QUOTE_WORDS)
    if not quote or quote.lower() in (q.lower() for q in earlier):
        quote = _clean_words(f"Every step toward {topic} counts", QUOTE_WORDS)
    # A tagline cut mid-phrase ('never lose sight of your') reads broken: drop it instead.
    tagline = _clean_words(raw.get("tagline"), 99)
    words = tagline.rstrip(".!").split()
    if len(words) > TAGLINE_WORDS + 1 or (words and words[-1].lower() in DANGLING_WORDS):
        tagline = ""
    return {"quote": quote, "tagline": tagline}


def build_image(index: int, total: int, background: str, person: str, topic: str,
                engine: ThinkingEngine, assets: dict[str, dict[str, Any]], fonts: dict[str, Any],
                earlier: list[str], used_fonts: list[str], batch_dir: Path, work_root: Path,
                args: argparse.Namespace, rng: random.Random,
                sampler: BackgroundSampler | None = None) -> Path | None:
    print(f"\n[Motivation] ===== Image {index}/{total}: {Path(person).stem} on "
          f"{Path(background).stem} =====")
    layout = compose(background, person, assets, rng, sampler)
    scene, place, text = layout["scene"], layout["person"], layout["text"]

    style = rng.choice(RECOLOR_STYLES)
    sky_hue = _hls(scene["sky"][len(scene["sky"]) // 2])[0] if scene["sky"] else rng.random()
    # Harmony with the scene: its complement, a triad partner, or a close analogue.
    hue = (sky_hue + rng.choice((0.5, 0.33, -0.33, 0.08, -0.08))) % 1.0
    variant = f"{Path(person).stem}-v{index}.svg"
    assets = {**assets, variant: {**assets[person],
                                  "xml": recolor_person(assets[person]["xml"], style, hue, rng)}}
    print(f"[Motivation] Person: {place['side']} side, {place['width']:.0f}x{place['height']:.0f}px, "
          f"clothing recoloured '{style}' (hue {hue * 360:.0f})")

    available = [f for f in QUOTE_FONTS if f in fonts] or list(fonts)
    quote_font = rng.choice([f for f in available if f not in used_fonts[-3:]] or available)
    used_fonts.append(quote_font)
    tagline_font = rng.choice([f for f in TAGLINE_FONTS if f in fonts and f != quote_font] or list(fonts))
    words = write_quote(engine, topic, earlier)
    earlier.append(words["quote"])
    print(f"[Motivation] Quote: {words['quote']!r} | tagline: {words['tagline']!r} | font '{quote_font}'")

    plan = {
        "schema_version": "1.0",
        "template_name": f"motivation-{Path(background).stem}",
        "description": f"Motivational poster about {topic}",
        "canvas": {"width": layout["width"], "height": layout["height"],
                   "background_color": scene["sky"][0] if scene["sky"] else "#FFFFFF"},
        "layout_grid": {"margin": round(layout["width"] * 0.06), "gutter": 24, "header": None},
        "elements": [
            {"id": "background", "asset": background, "x": 0.0, "y": 0.0, "width": layout["width"],
             "height": layout["height"], "z_index": -1000, "rotation": 0.0,
             "preserve_aspect_ratio": False, "style": {}},
            {"id": "person", "asset": variant, "x": place["x"], "y": place["y"], "width": place["width"],
             "height": place["height"], "z_index": 10, "rotation": 0.0,
             "preserve_aspect_ratio": True, "style": {}},
        ],
        "metadata": {"user_prompt": topic, "planner": "motivational", "recolor": style},
    }
    style_words = " ".join(filter(None, [text["align"], "shadow" if text["shadow"] else ""]))
    # A short quote is the whole poster: draw it large in the open sky.
    count = len(words["quote"].split())
    size = "xl" if count <= 6 else "large" if count <= 8 else ""
    if text["compact"]:
        size = {"xl": "large", "large": ""}.get(size, "")
    items = [{"id": "heading", "role": "heading", "label": "Quote", "content": words["quote"],
              "placement": f"{text['x']},{text['y']},{text['width']}", "color": text["color"],
              "font": quote_font, "style": " ".join(filter(None, [style_words, size]))}]
    if words["tagline"]:
        items.append({"id": "subheading", "role": "subheading", "label": "Tagline",
                      "content": words["tagline"], "placement": "below heading", "color": text["accent"],
                      "font": tagline_font, "style": " ".join(filter(None, [style_words, "upper"]))})
    spec = {"default_color": text["color"], "default_font": tagline_font, "items": items}

    run_directory = create_run_directory(work_root, plan["template_name"])
    try:
        plan_path = materialize_plan_assets(plan, assets, run_directory)
        apply_text_spec(plan_path, spec, original_prompt=topic, endpoint=args.ollama_url,
                        model=args.model, timeout=args.timeout, interactive=False)
        before = set(batch_dir.glob("*.png"))
        render_template(plan_path, png_dir=batch_dir)
        created = sorted(set(batch_dir.glob("*.png")) - before)
    finally:
        shutil.rmtree(run_directory, ignore_errors=True)  # Only the PNG is kept.
    return created[0] if created else None


def _categories(schema: Path) -> dict[str, str]:
    """filename -> category from svg_dictionary.py (the shared loader does not keep it)."""

    tree = ast.parse(schema.read_text(encoding="utf-8"), filename=str(schema))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SVG_DICTIONARY" for t in node.targets):
            data = ast.literal_eval(node.value)
            return {str(f): str(a["category"]) for f, a in data.items()
                    if isinstance(a, dict) and a.get("category")}
    return {}


def run(topic: str, total: int, args: argparse.Namespace, batch_dir: Path,
        work_root: Path) -> tuple[list[Path], int]:
    """Create ``total`` motivational posters in ``batch_dir``; returns (saved PNGs, failures)."""

    assets = load_svg_dictionary(args.schema)
    fonts = load_font_dictionary()
    if not fonts:
        raise ValueError("The font dictionary is empty or unreadable.")
    rng = random.Random(args.seed)
    engine = ThinkingEngine(assets=assets, fonts=fonts, endpoint=args.ollama_url, model=args.model,
                            timeout=args.timeout, temperature=args.temperature)
    print(f"\n[Motivation] Matching scenes and people to '{topic}'.")
    chooser = SceneChooser(assets, _categories(args.schema), topic, engine, rng)
    sampler = BackgroundSampler(work_root)

    saved: list[Path] = []
    failed = 0
    earlier: list[str] = []
    used_fonts: list[str] = []
    for index in range(1, total + 1):
        background, person = chooser.pick(index - 1)
        try:
            png = build_image(index, total, background, person, topic, engine, assets, fonts,
                              earlier, used_fonts, batch_dir, work_root, args, rng, sampler)
        except IMAGE_ERRORS as exc:
            print(f"[Motivation] Image {index} failed: {exc}")
            png = None
        if png:
            saved.append(png)
        else:
            failed += 1
    return saved, failed
