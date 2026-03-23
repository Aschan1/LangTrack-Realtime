from __future__ import annotations

import argparse
import importlib.util
import random
import re
import site
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# Prefer the active environment's packages over user-site overlays, which can
# otherwise inject an incompatible torchvision build into YOLOE imports.
USER_SITE_PATHS = {
    Path(path).resolve()
    for path in {
        site.getusersitepackages(),
        *(site.getsitepackages() if hasattr(site, "getsitepackages") else []),
    }
    if isinstance(path, str) and ".local" in path
}
sys.path[:] = [
    path
    for path in sys.path
    if not any(Path(path).resolve() == user_site for user_site in USER_SITE_PATHS if path)
]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ULTRALYTICS_PACKAGE_ROOT = REPO_ROOT / "src" / "ultralytics"
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ULTRALYTICS_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PACKAGE_ROOT))

from ultralytics import YOLOE


def load_whatsup_module():
    module_path = REPO_ROOT / "ultralytics" / "run_yoloe_whatsup.py"
    spec = importlib.util.spec_from_file_location("run_yoloe_whatsup_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load whatsup module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_whatsup_module()


@dataclass(frozen=True)
class BoxSelection:
    selected_target_box: Optional[tuple[float, float, float, float]]
    selected_anchor_box: Optional[tuple[float, float, float, float]]
    adjusted_target_box: Optional[tuple[float, float, float, float]]
    score: float
    target_found: bool
    anchor_found: bool
    pair_found: bool


@dataclass(frozen=True)
class SampleVisualization:
    predicted_index: Optional[int]
    display_index: Optional[int]
    correct_index: int
    option_scores: list[float]
    option_available: list[bool]
    box_selection: BoxSelection

    @property
    def is_correct(self) -> bool:
        return self.predicted_index == self.correct_index if self.predicted_index is not None else False

    @property
    def abstained(self) -> bool:
        return self.predicted_index is None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize YOLOE on the whatsup benchmark (for example controlled_images), showing the selected "
            "target/anchor detections, adjusted target box, and option scores."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["controlled_images"],
        choices=list(BENCHMARK.ALL_DATASETS),
        help="Benchmark datasets to sample from.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "yoloe-26l-seg.pt",
        help="Path to YOLOE weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if BENCHMARK.torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="YOLOE confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--max-det", type=int, default=50, help="Maximum detections per image.")
    parser.add_argument(
        "--prompt-mode",
        choices=BENCHMARK.PROMPT_MODES,
        default="object",
        help="How to build YOLOE prompts for each image.",
    )
    parser.add_argument(
        "--query-mode",
        choices=BENCHMARK.QUERY_MODES,
        default="target",
        help="How to turn labels into YOLOE query text.",
    )
    parser.add_argument(
        "--scorer",
        choices=("heuristic", "relation_head"),
        default="relation_head",
        help="Use the built-in heuristic option scorer or a trained relation-head checkpoint.",
    )
    parser.add_argument(
        "--relation-head",
        type=Path,
        default=REPO_ROOT / "outputs" / "relation_head_indoors_6500.pt",
        help="Path to a relation-head checkpoint for --scorer relation_head.",
    )
    parser.add_argument(
        "--relation-head-margin",
        type=float,
        default=0.0,
        help="Optional abstention margin for learned relation-head scores.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of sampled benchmark rows to visualize.",
    )
    parser.add_argument(
        "--max-load-samples",
        type=int,
        default=None,
        help="Optional per-dataset cap when loading benchmark rows before sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling benchmark rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "visualizations_yoloe_whatsup",
        help="Directory to store visualization images.",
    )
    parser.add_argument(
        "--show-all-detections",
        action="store_true",
        help="Draw all target/anchor detections for the displayed option, not just the selected pair.",
    )
    parser.add_argument(
        "--show-adjusted-boxes",
        dest="show_adjusted_boxes",
        action="store_true",
        help="Draw the adjusted target box overlay.",
    )
    parser.add_argument(
        "--hide-adjusted-boxes",
        dest="show_adjusted_boxes",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-abstain",
        action="store_true",
        help="Allow the visualizer to abstain instead of forcing a best-effort prediction.",
    )
    parser.set_defaults(show_adjusted_boxes=False)
    return parser


def sanitize_filename(text: str, max_length: int = 60) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    if not cleaned:
        cleaned = "sample"
    return cleaned[:max_length]


def choose_samples(samples: list, *, num_samples: int, seed: int) -> list:
    if not samples:
        return []
    random.seed(seed)
    if num_samples >= len(samples):
        return list(samples)
    return random.sample(samples, num_samples)


def pick_display_index(predicted_index: Optional[int], option_scores: list[float], option_available: list[bool], correct_index: int) -> Optional[int]:
    if predicted_index is not None:
        return predicted_index
    candidate_indices = [index for index, available in enumerate(option_available) if available]
    if candidate_indices:
        return max(candidate_indices, key=lambda index: option_scores[index])
    return correct_index if option_scores else None


def pick_predicted_index(option_scores: list[float], option_available: list[bool]) -> Optional[int]:
    available_indices = [index for index, available in enumerate(option_available) if available]
    if available_indices:
        return max(available_indices, key=lambda index: option_scores[index])
    if option_scores:
        return max(range(len(option_scores)), key=lambda index: option_scores[index])
    return None


def build_object_to_query_labels(query_to_objects: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    object_to_query_labels: dict[str, list[str]] = defaultdict(list)
    for query_label, object_labels in query_to_objects.items():
        for object_label in object_labels:
            if query_label not in object_to_query_labels[object_label]:
                object_to_query_labels[object_label].append(query_label)
    return {
        object_label: tuple(query_labels)
        for object_label, query_labels in object_to_query_labels.items()
    }


def ensure_required_detections(
    model: YOLOE,
    image: Image.Image,
    detections_by_label: dict[str, list],
    *,
    required_object_labels: set[str],
    object_to_query_labels: dict[str, tuple[str, ...]],
    imgsz: int,
    max_det: int,
) -> dict[str, list]:
    next_det_id = max(
        (detection.det_id for detections in detections_by_label.values() for detection in detections),
        default=-1,
    ) + 1

    for object_label in sorted(required_object_labels):
        if detections_by_label.get(object_label):
            continue

        candidate_query_labels = object_to_query_labels.get(object_label, ())
        if not candidate_query_labels:
            continue

        query_label = candidate_query_labels[0]
        BENCHMARK.configure_classes(model, [query_label])
        result = model.predict(
            image,
            conf=0.0,
            imgsz=imgsz,
            max_det=max(1, max_det),
            verbose=False,
        )[0]

        raw_boxes = result.boxes.xyxy.detach().cpu().tolist()
        raw_confidences = result.boxes.conf.detach().cpu().tolist()
        if not raw_boxes:
            continue

        best_index = max(range(len(raw_boxes)), key=lambda index: raw_confidences[index])
        detections_by_label[object_label].append(
            BENCHMARK.Detection(
                det_id=next_det_id,
                label=object_label,
                conf=float(raw_confidences[best_index]),
                box=tuple(float(value) for value in raw_boxes[best_index]),
            )
        )
        next_det_id += 1

    return detections_by_label


def select_boxes_for_option(
    option,
    detections_by_label: dict[str, list],
    *,
    scorer: str,
    relation_head,
    image_width: int,
    image_height: int,
) -> BoxSelection:
    target_detections = list(detections_by_label.get(option.target, []))
    anchor_detections = list(detections_by_label.get(option.anchor or "", [])) if option.anchor else []

    target_found = bool(target_detections)
    anchor_found = bool(anchor_detections) if option.anchor else False
    pair_found = target_found and anchor_found if option.anchor else target_found

    if not target_detections:
        return BoxSelection(None, None, None, 0.0, target_found, anchor_found, pair_found)

    if not option.is_pairwise:
        best_target = max(
            target_detections,
            key=lambda det: det.conf * BENCHMARK.score_single_relation(det.box, option.relation, image_width, image_height),
        )
        return BoxSelection(
            selected_target_box=best_target.box,
            selected_anchor_box=None,
            adjusted_target_box=best_target.box,
            score=float(best_target.conf * BENCHMARK.score_single_relation(best_target.box, option.relation, image_width, image_height)),
            target_found=target_found,
            anchor_found=anchor_found,
            pair_found=pair_found,
        )

    if not anchor_detections:
        return BoxSelection(None, None, None, 0.0, target_found, anchor_found, pair_found)

    best_target_box = None
    best_anchor_box = None
    best_adjusted_box = None
    best_score = -float("inf")

    for target_det in target_detections:
        for anchor_det in anchor_detections:
            if target_det.label == anchor_det.label and target_det.det_id == anchor_det.det_id:
                continue
            if scorer == "relation_head":
                pair_detections = {
                    option.target: [target_det],
                    option.anchor: [anchor_det],
                }
                score = BENCHMARK.score_options_with_relation_head(
                    [option],
                    pair_detections,
                    image_width,
                    image_height,
                    relation_head,
                )[0]
            else:
                relation_score = BENCHMARK.score_pair_relation(
                    target_det.box,
                    anchor_det.box,
                    option.relation,
                    image_width,
                    image_height,
                )
                score = (max(target_det.conf, 1e-8) * max(anchor_det.conf, 1e-8)) ** 0.5 * relation_score

            if score > best_score:
                best_score = float(score)
                best_target_box = target_det.box
                best_anchor_box = anchor_det.box
                best_adjusted_box = BENCHMARK.transform_box_for_relation(
                    target_det.box,
                    anchor_det.box,
                    option.relation,
                    role="target",
                )

    return BoxSelection(
        selected_target_box=best_target_box,
        selected_anchor_box=best_anchor_box,
        adjusted_target_box=best_adjusted_box,
        score=0.0 if best_score == -float("inf") else best_score,
        target_found=target_found,
        anchor_found=anchor_found,
        pair_found=pair_found,
    )


def build_sample_visualization(
    sample,
    detections_by_label: dict[str, list],
    *,
    scorer: str,
    relation_head,
    relation_head_margin: float,
    image_width: int,
    image_height: int,
    allow_abstain: bool,
) -> SampleVisualization:
    if scorer == "relation_head":
        option_scores = BENCHMARK.score_options_with_relation_head(
            sample.options,
            detections_by_label,
            image_width,
            image_height,
            relation_head,
        )
        option_available = [BENCHMARK.option_has_required_detections(option, detections_by_label) for option in sample.options]
        predicted_index = BENCHMARK.choose_prediction(
            option_scores,
            option_mask=option_available,
            require_positive=False,
            min_margin=relation_head_margin,
        )
    else:
        option_scores = [BENCHMARK.score_option(option, detections_by_label, image_width, image_height) for option in sample.options]
        option_available = [BENCHMARK.option_has_required_detections(option, detections_by_label) for option in sample.options]
        predicted_index = BENCHMARK.choose_prediction(option_scores)

    if predicted_index is None and not allow_abstain:
        predicted_index = pick_predicted_index(option_scores, option_available)

    display_index = pick_display_index(predicted_index, option_scores, option_available, sample.correct_index)
    if display_index is None:
        box_selection = BoxSelection(None, None, None, 0.0, False, False, False)
    else:
        box_selection = select_boxes_for_option(
            sample.options[display_index],
            detections_by_label,
            scorer=scorer,
            relation_head=relation_head,
            image_width=image_width,
            image_height=image_height,
        )

    return SampleVisualization(
        predicted_index=predicted_index,
        display_index=display_index,
        correct_index=sample.correct_index,
        option_scores=[float(score) for score in option_scores],
        option_available=option_available,
        box_selection=box_selection,
    )


def color(name: str) -> tuple[int, int, int]:
    palette = {
        "target_all": (88, 162, 255),
        "anchor_all": (214, 110, 255),
        "target_selected": (0, 204, 255),
        "anchor_selected": (255, 64, 214),
        "adjusted_correct": (0, 220, 80),
        "adjusted_wrong": (255, 60, 60),
        "adjusted_abstain": (255, 170, 0),
    }
    return palette[name]


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], line_color: tuple[int, int, int], width: int, label: Optional[str], font) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=line_color, width=width)
    if not label:
        return
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    label_x = x1
    label_y = max(0, y1 - text_h - 6)
    draw.rectangle((label_x, label_y, label_x + text_w + 8, label_y + text_h + 6), fill=(20, 20, 20))
    draw.text((label_x + 4, label_y + 3), label, fill=line_color, font=font)


def draw_text_block(draw: ImageDraw.ImageDraw, lines: list[str], image_width: int, font) -> None:
    if not lines:
        return
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = min(max(box[2] - box[0] for box in line_boxes), image_width - 20)
    line_height = max(box[3] - box[1] for box in line_boxes)
    block_height = len(lines) * (line_height + 4) + 10
    draw.rectangle((8, 8, 16 + text_width, 8 + block_height), fill=(15, 15, 15))
    y = 14
    for line in lines:
        draw.text((12, y), line, fill=(245, 245, 245), font=font)
        y += line_height + 4


def draw_sample(
    image: Image.Image,
    sample,
    sample_vis: SampleVisualization,
    detections_by_label: dict[str, list],
    *,
    dataset_name: str,
    scorer: str,
    prompt_mode: str,
    query_mode: str,
    show_all_detections: bool,
    show_adjusted_boxes: bool,
) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()

    display_option = sample.options[sample_vis.display_index] if sample_vis.display_index is not None else None
    correct_option = sample.options[sample.correct_index]

    if show_all_detections and display_option is not None:
        for index, det in enumerate(detections_by_label.get(display_option.target, [])[:10], start=1):
            draw_box(draw, det.box, color("target_all"), 1, f"T{index}:{det.conf:.2f}", font)
        if display_option.anchor:
            for index, det in enumerate(detections_by_label.get(display_option.anchor, [])[:10], start=1):
                draw_box(draw, det.box, color("anchor_all"), 1, f"A{index}:{det.conf:.2f}", font)

    if sample_vis.box_selection.selected_target_box is not None:
        draw_box(draw, sample_vis.box_selection.selected_target_box, color("target_selected"), 3, "picked target", font)
    if sample_vis.box_selection.selected_anchor_box is not None:
        draw_box(draw, sample_vis.box_selection.selected_anchor_box, color("anchor_selected"), 3, "picked anchor", font)

    if show_adjusted_boxes and sample_vis.box_selection.adjusted_target_box is not None:
        if sample_vis.abstained:
            adjusted_color = color("adjusted_abstain")
        elif sample_vis.is_correct:
            adjusted_color = color("adjusted_correct")
        else:
            adjusted_color = color("adjusted_wrong")
        draw_box(draw, sample_vis.box_selection.adjusted_target_box, adjusted_color, 4, "adjusted target", font)

    score_lines = []
    for index, (option, score, available) in enumerate(zip(sample.options, sample_vis.option_scores, sample_vis.option_available)):
        tags = []
        if index == sample.correct_index:
            tags.append("gt")
        if index == sample_vis.predicted_index:
            tags.append("pred")
        if index == sample_vis.display_index and index != sample_vis.predicted_index:
            tags.append("shown")
        suffix = f" [{' '.join(tags)}]" if tags else ""
        availability = "ok" if available else "miss"
        score_lines.append(f"{index}: {score:.4f} ({availability}){suffix}")

    predicted_text = sample.options[sample_vis.predicted_index].text if sample_vis.predicted_index is not None else "ABSTAIN"
    shown_text = display_option.text if display_option is not None else "<none>"
    lines = [
        f"dataset: {dataset_name} sample: {sample.sample_id}",
        f"scorer: {scorer} | prompt: {prompt_mode} | query: {query_mode}",
        f"correct: {correct_option.text}",
        f"predicted: {predicted_text}",
        f"displayed: {shown_text}",
        f"status: {'correct' if sample_vis.is_correct else ('abstain' if sample_vis.abstained else 'wrong')}",
        (
            f"target_found: {sample_vis.box_selection.target_found} | "
            f"anchor_found: {sample_vis.box_selection.anchor_found} | "
            f"pair_found: {sample_vis.box_selection.pair_found}"
        ),
        f"display score: {sample_vis.box_selection.score:.4f}",
        "scores:",
        *score_lines,
    ]
    draw_text_block(draw, lines, vis.width, font)
    return vis


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Missing YOLOE weights: {args.weights}")
    if args.scorer == "relation_head" and args.relation_head is None:
        raise ValueError("--relation-head is required when --scorer relation_head")

    relation_head = None
    if args.relation_head is not None:
        if not args.relation_head.exists():
            raise FileNotFoundError(f"Missing relation-head checkpoint: {args.relation_head}")
        relation_head = BENCHMARK.load_relation_head_checkpoint(args.relation_head, device=args.device)

    print(f"Loading YOLOE from {args.weights} on {args.device}...")
    print(f"Using scorer: {args.scorer}")
    print(f"Using prompt mode: {args.prompt_mode}")
    print(f"Using query mode: {args.query_mode}")
    if relation_head is not None:
        print(f"Using relation head: {args.relation_head}")

    model = YOLOE(str(args.weights)).to(args.device)
    resolver = BENCHMARK.ImageResolver(REPO_ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    samples_by_dataset_image: dict[tuple[str, str], list] = defaultdict(list)
    for dataset_name in args.datasets:
        spec = BENCHMARK.DATASET_SPECS[dataset_name]
        dataset_samples = BENCHMARK.load_samples(spec, max_samples=args.max_load_samples)
        all_samples.extend(dataset_samples)
        for sample in dataset_samples:
            samples_by_dataset_image[(dataset_name, sample.image_key)].append(sample)

    selected_samples = choose_samples(all_samples, num_samples=args.num_samples, seed=args.seed)
    if not selected_samples:
        raise RuntimeError("No samples were selected for visualization.")

    print(f"Visualizing {len(selected_samples)} sampled benchmark rows")
    selected_by_dataset_image: dict[tuple[str, str], list] = defaultdict(list)
    for sample in selected_samples:
        selected_by_dataset_image[(sample.dataset_name, sample.image_key)].append(sample)

    saved = 0
    for (dataset_name, image_key), image_selected_samples in selected_by_dataset_image.items():
        spec = BENCHMARK.DATASET_SPECS[dataset_name]
        image_group = samples_by_dataset_image[(dataset_name, image_key)]
        image_hint = image_group[0].image_hint
        image, _ = resolver.load(spec.image_source, image_key, image_hint)
        image_width, image_height = image.size

        query_to_objects = BENCHMARK.build_query_to_objects(
            image_group,
            prompt_mode=args.prompt_mode,
            query_mode=args.query_mode,
        )
        object_to_query_labels = build_object_to_query_labels(query_to_objects)
        detections_by_label = BENCHMARK.run_yoloe(
            model=model,
            image=image,
            query_labels=sorted(query_to_objects),
            query_to_objects=query_to_objects,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
        )

        required_object_labels = {
            label
            for sample in image_selected_samples
            for label in (
                sample.options[sample.correct_index].target,
                sample.options[sample.correct_index].anchor,
            )
            if label
        }
        detections_by_label = ensure_required_detections(
            model,
            image,
            detections_by_label,
            required_object_labels=required_object_labels,
            object_to_query_labels=object_to_query_labels,
            imgsz=args.imgsz,
            max_det=args.max_det,
        )

        for sample in image_selected_samples:
            sample_vis = build_sample_visualization(
                sample,
                detections_by_label,
                scorer=args.scorer,
                relation_head=relation_head,
                relation_head_margin=args.relation_head_margin,
                image_width=image_width,
                image_height=image_height,
                allow_abstain=args.allow_abstain,
            )
            vis = draw_sample(
                image,
                sample,
                sample_vis,
                detections_by_label,
                dataset_name=dataset_name,
                scorer=args.scorer,
                prompt_mode=args.prompt_mode,
                query_mode=args.query_mode,
                show_all_detections=args.show_all_detections,
                show_adjusted_boxes=args.show_adjusted_boxes,
            )
            phrase_slug = sanitize_filename(sample.options[sample.correct_index].text, max_length=50)
            output_path = args.output_dir / (
                f"whatsupvis_{dataset_name}_{sample.sample_id}_{phrase_slug}.png"
            )
            vis.save(output_path)
            print(f"Saved: {output_path}")
            saved += 1

    print(f"Done. Saved {saved} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
