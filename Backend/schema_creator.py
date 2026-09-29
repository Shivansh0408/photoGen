import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "Assets"
DICTIONARY_DIR = PROJECT_ROOT / "Schema" / "asset_dictonary"
OUTPUT_FILE = DICTIONARY_DIR / "svg_dictionary.py"

NAME_PATTERN = re.compile(
    r"^(?P<name>.+?)\.svg$",
    re.IGNORECASE
)

WIDTH_PATTERN = re.compile(
    r'\bwidth\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE
)

HEIGHT_PATTERN = re.compile(
    r'\bheight\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE
)

VIEWBOX_PATTERN = re.compile(
    r'\bviewBox\s*=\s*["\']\s*'
    r'(-?\d+(?:\.\d+)?)\s+'
    r'(-?\d+(?:\.\d+)?)\s+'
    r'(-?\d+(?:\.\d+)?)\s+'
    r'(-?\d+(?:\.\d+)?)'
    r'\s*["\']',
    re.IGNORECASE
)


def create_directories():
    DICTIONARY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def extract_asset_name(filename):
    match = NAME_PATTERN.search(filename)

    if not match:
        return None

    return match.group("name")


def read_svg_file(file_path):
    try:
        return file_path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        return file_path.read_text(
            encoding="utf-8-sig"
        )


def convert_to_pixels(value):
    value = value.strip().lower()

    match = re.fullmatch(
        r"(-?\d+(?:\.\d+)?)\s*(px|pt|pc|mm|cm|in)?",
        value
    )

    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2) or "px"

    conversions = {
        "px": 1,
        "pt": 96 / 72,
        "pc": 16,
        "mm": 96 / 25.4,
        "cm": 96 / 2.54,
        "in": 96
    }

    pixels = number * conversions[unit]

    if pixels.is_integer():
        return int(pixels)

    return round(pixels, 2)


def extract_dimensions(xml_text):
    width_match = WIDTH_PATTERN.search(xml_text)
    height_match = HEIGHT_PATTERN.search(xml_text)

    width = None
    height = None

    if width_match:
        width = convert_to_pixels(
            width_match.group(1)
        )

    if height_match:
        height = convert_to_pixels(
            height_match.group(1)
        )

    if width is not None and height is not None:
        return {
            "width": width,
            "height": height,
            "unit": "px"
        }

    viewbox_match = VIEWBOX_PATTERN.search(
        xml_text
    )

    if viewbox_match:
        viewbox_width = float(
            viewbox_match.group(3)
        )

        viewbox_height = float(
            viewbox_match.group(4)
        )

        if width is None:
            width = (
                int(viewbox_width)
                if viewbox_width.is_integer()
                else viewbox_width
            )

        if height is None:
            height = (
                int(viewbox_height)
                if viewbox_height.is_integer()
                else viewbox_height
            )

    return {
        "width": width,
        "height": height,
        "unit": "px"
    }


def get_svg_assets():
    if not ASSETS_DIR.exists():
        print(
            f"\nERROR: Assets directory does not exist:\n"
            f"{ASSETS_DIR.resolve()}"
        )
        return []

    return sorted(
        [
            file_path
            for file_path in ASSETS_DIR.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() == ".svg"
        ]
    )


def escape_python_string(value):
    return (
        value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
    )


def format_xml(xml_text):
    xml_text = xml_text.strip()

    xml_text = xml_text.replace(
        '"""',
        '\\"\\"\\"'
    )

    return xml_text


def load_existing_dictionary():
    if not OUTPUT_FILE.exists():
        return {}

    content = OUTPUT_FILE.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        content,
        filename=str(OUTPUT_FILE)
    )

    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue

        if any(
            isinstance(target, ast.Name)
            and target.id == "SVG_DICTIONARY"
            for target in statement.targets
        ):
            data = ast.literal_eval(
                statement.value
            )

            if isinstance(data, dict):
                return data

            break

    raise ValueError(
        f"{OUTPUT_FILE} does not contain a valid SVG_DICTIONARY. "
        "The existing file was not changed."
    )


def format_python_dictionary(data):
    lines = [
        "SVG_DICTIONARY = {"
    ]

    items = list(data.items())

    for item_index, (filename, asset) in enumerate(items):

        lines.append(
            f'    "{escape_python_string(filename)}": {{'
        )

        lines.append(
            f'        "name": "{escape_python_string(asset["name"])}",'
        )

        lines.append(
            f'        "usage": "{escape_python_string(asset["usage"])}",'
        )

        dimensions = asset["dimensions"]

        lines.append(
            '        "dimensions": {'
        )

        lines.append(
            f'            "width": {dimensions["width"]},'
        )

        lines.append(
            f'            "height": {dimensions["height"]},'
        )

        lines.append(
            f'            "unit": "{dimensions["unit"]}"'
        )

        lines.append(
            "        },"
        )

        lines.append(
            '        "xml": """'
        )

        xml = format_xml(
            asset["xml"]
        )

        lines.append(xml)

        lines.append(
            '        """'
        )

        lines.append(
            "    }"
            + (
                ","
                if item_index < len(items) - 1
                else ""
            )
        )

    lines.append("}")

    return "\n".join(lines) + "\n"


def save_dictionary(data):
    content = format_python_dictionary(
        data
    )

    OUTPUT_FILE.write_text(
        content,
        encoding="utf-8"
    )


def main():
    print("=" * 60)
    print("SVG ASSET DICTIONARY CREATOR")
    print("=" * 60)

    create_directories()

    try:
        svg_dictionary = load_existing_dictionary()
    except (OSError, SyntaxError, ValueError) as error:
        print(
            f"\nERROR: Existing dictionary could not be read: {error}"
        )
        print(
            "Saved mappings were left untouched."
        )
        return

    assets = get_svg_assets()

    if not assets:
        print(
            "\nNo SVG files were found in:"
        )

        print(
            ASSETS_DIR.resolve()
        )

        return

    existing_names = {
        str(filename).casefold()
        for filename in svg_dictionary
    }
    new_assets = [
        asset_path
        for asset_path in assets
        if asset_path.name.casefold() not in existing_names
    ]

    print(
        f"\nFound {len(assets)} SVG asset(s)."
    )
    print(
        f"Already mapped: {len(assets) - len(new_assets)}; "
        f"new to map: {len(new_assets)}."
    )

    print(
        f"Output file:\n{OUTPUT_FILE.resolve()}"
    )

    if not new_assets:
        print(
            "\nNo new SVG files to map. Existing mappings were kept."
        )
        return

    for index, asset_path in enumerate(
        assets,
        start=1
    ):
        filename = asset_path.name

        if filename.casefold() in existing_names:
            print(
                f"\n[{index}/{len(assets)}] "
                f"{filename} already processed."
            )
            continue

        asset_name = extract_asset_name(
            filename
        )

        if not asset_name:
            print(
                f"\nCould not extract name from:"
                f"\n{filename}"
            )
            continue

        try:
            xml_text = read_svg_file(
                asset_path
            )

        except OSError as error:
            print(
                f"\nCould not read {filename}: {error}"
            )
            continue

        dimensions = extract_dimensions(
            xml_text
        )

        print("\n")
        print("=" * 60)
        print(
            f"ASSET {index}/{len(assets)}"
        )
        print("=" * 60)

        print(
            f"Filename   : {filename}"
        )

        print(
            f"Name       : {asset_name}"
        )

        print(
            f"Width      : {dimensions['width']} px"
        )

        print(
            f"Height     : {dimensions['height']} px"
        )

        print(
            f"XML Size   : {len(xml_text)} characters"
        )

        print("-" * 60)

        while True:
            usage = input(
                "\nWhat is this asset used for?\n> "
            ).strip()

            if usage:
                break

            print(
                "Please enter a description."
            )

        svg_dictionary[filename] = {
            "name": asset_name,
            "usage": usage,
            "dimensions": {
                "width": dimensions["width"],
                "height": dimensions["height"],
                "unit": "px"
            },
            "xml": xml_text
        }

        save_dictionary(
            svg_dictionary
        )
        existing_names.add(
            filename.casefold()
        )

        print(
            f"\nSaved: {filename}"
        )

        print(
            f"Dictionary: "
            f"{OUTPUT_FILE.resolve()}"
        )

    print("\n")
    print("=" * 60)
    print("PROCESS COMPLETE")
    print("=" * 60)

    print(
        f"\nTotal assets stored: "
        f"{len(svg_dictionary)}"
    )

    print(
        f"\nDictionary file:"
        f"\n{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()