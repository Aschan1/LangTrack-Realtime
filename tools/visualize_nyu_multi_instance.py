#!/usr/bin/env python3
import argparse
import colorsys
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANNOTATIONS = REPO_ROOT / "nyu_dataset" / "filtered_nyu_LM_vg_multi_instance.json"
DEFAULT_IMAGES_DIR = REPO_ROOT / "nyu_dataset" / "images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "nyu_dataset" / "visualizations"


def parse_image_id(value: str) -> int:
    raw = value.strip()
    if raw.lower().endswith(".png"):
        raw = Path(raw).stem
    if not raw.isdigit():
        raise argparse.ArgumentTypeError(
            f"Invalid image id '{value}'. Use a numeric id like 0, 0000, or 0000.png."
        )
    return int(raw)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize NYU multi-instance relation annotations.")
    parser.add_argument(
        "-i",
        "--image-id",
        type=parse_image_id,
        default=0,
        help="First NYU image id to visualize, for example 0, 0000, or 0000.png.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="How many annotated image ids to visualize starting from --image-id. Default: 1.",
    )
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="Path to NYU annotation JSON.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR), help="Directory containing NYU PNG images.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save visualizations.")
    return parser.parse_args()


def color_for_key(key: str) -> tuple[int, int, int]:
    value = abs(hash(key)) % 360
    hue = value / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        path = Path(font_path)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def make_visualization(image_path: Path, entries: list[dict], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    font = try_load_font(16)
    small_font = try_load_font(14)

    legend_width = 560
    header_height = 42
    row_gap = 8
    line_height = 18

    temp_draw = ImageDraw.Draw(image.copy())
    legend_rows = []
    for index, entry in enumerate(entries, start=1):
        phrase = entry["phrase"]
        key = f"{entry['keywords']['target']}|{entry['keywords']['anchor_object']}"
        color = color_for_key(key)
        wrapped = wrap_text(temp_draw, phrase, font=small_font, max_width=legend_width - 70)
        legend_rows.append((index, entry, color, wrapped))

    legend_content_height = 24
    for _, entry, _, wrapped in legend_rows:
        legend_content_height += len(wrapped) * line_height + row_gap

    canvas_height = max(image.height, header_height + legend_content_height + 12)
    canvas_width = image.width + legend_width
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 246, 241))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((image.width, 0, canvas_width, canvas_height), fill=(245, 243, 236))
    draw.rectangle((0, 0, image.width, 34), fill=(20, 20, 20))
    draw.text((12, 8), image_path.name, fill=(255, 255, 255), font=font)

    draw.text((image.width + 18, 10), "NYU Multi-Instance Relations", fill=(28, 28, 28), font=font)

    for index, entry, color, wrapped in legend_rows:
        x1 = int(entry["x"])
        y1 = int(entry["y"])
        x2 = x1 + int(entry["width"]) - 1
        y2 = y1 + int(entry["height"]) - 1

        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

        badge_text = str(index)
        badge_bbox = draw.textbbox((0, 0), badge_text, font=small_font)
        badge_w = badge_bbox[2] - badge_bbox[0] + 12
        badge_h = badge_bbox[3] - badge_bbox[1] + 8
        badge_x = x1
        badge_y = max(36, y1 - badge_h - 4)
        draw.rounded_rectangle((badge_x, badge_y, badge_x + badge_w, badge_y + badge_h), radius=8, fill=color)
        draw.text((badge_x + 6, badge_y + 3), badge_text, fill=(255, 255, 255), font=small_font)

    legend_y = 42
    for index, entry, color, wrapped in legend_rows:
        swatch_x = image.width + 18
        swatch_y = legend_y + 4
        draw.rounded_rectangle((swatch_x, swatch_y, swatch_x + 18, swatch_y + 18), radius=4, fill=color)
        draw.text((swatch_x + 28, legend_y), f"[{index}] {wrapped[0]}", fill=(30, 30, 30), font=small_font)
        current_y = legend_y + line_height
        for extra_line in wrapped[1:]:
            draw.text((swatch_x + 28, current_y), extra_line, fill=(30, 30, 30), font=small_font)
            current_y += line_height
        detail = f"bbox=({entry['x']}, {entry['y']}, {entry['width']}, {entry['height']})"
        draw.text((swatch_x + 28, current_y), detail, fill=(90, 90, 90), font=small_font)
        legend_y = current_y + row_gap + 8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def select_image_ids(data: list[dict], start_image_id: int, num_images: int) -> list[int]:
    if num_images <= 0:
        raise ValueError("--num-images must be at least 1.")

    available_image_ids = sorted({int(item["image_id"]) for item in data})
    selected_image_ids = [image_id for image_id in available_image_ids if image_id >= start_image_id][:num_images]
    if not selected_image_ids:
        raise ValueError(f"No annotations found for image_id>={start_image_id}.")
    return selected_image_ids


def main() -> None:
    args = parse_args()

    annotations_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_ids = select_image_ids(data, args.image_id, args.num_images)
    saved_paths: list[Path] = []
    for image_id in image_ids:
        entries = [item for item in data if int(item["image_id"]) == image_id]
        if not entries:
            continue

        entries.sort(key=lambda item: (item["keywords"]["target"], item["x"], item["y"], item["region_id"]))

        image_path = images_dir / f"{image_id:04d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        output_path = output_dir / f"nyu_multi_instance_{image_id:04d}.png"
        make_visualization(image_path, entries, output_path)
        saved_paths.append(output_path)

    if not saved_paths:
        raise ValueError("No visualizations were generated.")

    if len(saved_paths) == 1:
        print(f"Saved visualization to: {saved_paths[0]}")
    else:
        print(f"Saved {len(saved_paths)} visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
