"""Font styles offered by the text stage (heading, subheading, box text, additional text).

Each key is the style name the user picks (by name or by its number in this order).
The text layer reads this file with ``ast.literal_eval`` (it is never executed), so
keep every value a plain literal.

Fields
    name            display name
    usage           when the style works best (shown to the user)
    family          CSS font stack; the first font that exists on the machine is used.
                    The stacks start with fonts that ship with Windows.
    weights         font weight per text role (heading, subheading, box, text)
    letter_spacing  tracking in em for headings and for body text
    line_height     line spacing as a multiple of the font size
    width_factor    average glyph width relative to Arial (used to wrap and fit text)
    heading_case    default case for headings: "upper", "title" or None (as typed)
    file            optional font file inside Schema/font_dictnoary/fonts/ (.ttf, .otf,
                    .woff, .woff2). When present it is embedded in the SVG, so the style
                    looks the same on machines that do not have the font installed.
"""

FONT_DICTIONARY = {
    "Corporate Sans": {
        "name": "Corporate Sans",
        "usage": "Clean business look for reports, dashboards and corporate graphics",
        "family": "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
        "weights": {"heading": "700", "subheading": "600", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.0, "body": 0.0},
        "line_height": 1.25,
        "width_factor": 1.0,
        "heading_case": None,
        "file": None
    },
    "Geometric Modern": {
        "name": "Geometric Modern",
        "usage": "Round, modern shapes for tech, product launches and minimal posters",
        "family": "'Century Gothic', 'Futura', 'Avenir Next', 'Segoe UI', sans-serif",
        "weights": {"heading": "700", "subheading": "400", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.03, "body": 0.01},
        "line_height": 1.25,
        "width_factor": 1.15,
        "heading_case": None,
        "file": None
    },
    "Bold Impact": {
        "name": "Bold Impact",
        "usage": "Loud headlines for sales, sports and announcement posters (headlines only)",
        "family": "Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif",
        "weights": {"heading": "400", "subheading": "400", "box": "400", "text": "400"},
        "letter_spacing": {"heading": 0.03, "body": 0.01},
        "line_height": 1.1,
        "width_factor": 0.92,
        "heading_case": "upper",
        "file": None
    },
    "Condensed Display": {
        "name": "Condensed Display",
        "usage": "Tall, space-saving type for infographics, stats and narrow columns",
        "family": "'Bahnschrift SemiCondensed', 'Bahnschrift', 'Arial Narrow', sans-serif",
        "weights": {"heading": "700", "subheading": "600", "box": "600", "text": "400"},
        "letter_spacing": {"heading": 0.05, "body": 0.01},
        "line_height": 1.2,
        "width_factor": 0.95,
        "heading_case": "upper",
        "file": None
    },
    "Classic Serif": {
        "name": "Classic Serif",
        "usage": "Editorial, magazine and newsletter layouts with a trustworthy tone",
        "family": "Georgia, 'Times New Roman', serif",
        "weights": {"heading": "700", "subheading": "400", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.0, "body": 0.0},
        "line_height": 1.3,
        "width_factor": 1.08,
        "heading_case": None,
        "file": None
    },
    "Elegant Serif": {
        "name": "Elegant Serif",
        "usage": "Refined look for invitations, luxury brands, weddings and awards",
        "family": "'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif",
        "weights": {"heading": "400", "subheading": "400", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.08, "body": 0.01},
        "line_height": 1.3,
        "width_factor": 1.05,
        "heading_case": "upper",
        "file": None
    },
    "Formal Serif": {
        "name": "Formal Serif",
        "usage": "Formal, official tone for legal, academic and government material",
        "family": "'Times New Roman', Times, serif",
        "weights": {"heading": "700", "subheading": "400", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.02, "body": 0.0},
        "line_height": 1.3,
        "width_factor": 0.95,
        "heading_case": None,
        "file": None
    },
    "Friendly Humanist": {
        "name": "Friendly Humanist",
        "usage": "Warm, approachable feel for education, community and social posts",
        "family": "'Trebuchet MS', 'Lucida Grande', 'Segoe UI', sans-serif",
        "weights": {"heading": "700", "subheading": "400", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.01, "body": 0.0},
        "line_height": 1.28,
        "width_factor": 1.0,
        "heading_case": None,
        "file": None
    },
    "Wide Readable": {
        "name": "Wide Readable",
        "usage": "Very legible body text for small sizes, captions and long descriptions",
        "family": "Verdana, Tahoma, 'Segoe UI', sans-serif",
        "weights": {"heading": "700", "subheading": "400", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.0, "body": 0.0},
        "line_height": 1.35,
        "width_factor": 1.18,
        "heading_case": None,
        "file": None
    },
    "Typewriter Mono": {
        "name": "Typewriter Mono",
        "usage": "Technical, retro or code-style graphics, receipts and data labels",
        "family": "'Courier New', Consolas, 'Lucida Console', monospace",
        "weights": {"heading": "700", "subheading": "700", "box": "700", "text": "400"},
        "letter_spacing": {"heading": 0.02, "body": 0.0},
        "line_height": 1.3,
        "width_factor": 1.15,
        "heading_case": None,
        "file": None
    }
}
