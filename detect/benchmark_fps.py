"""
Unified FPS & GFLOPs benchmark for all detection pipelines.

Measures per-image inference latency (with CUDA synchronization) on the same
set of images so results are directly comparable.

Pipelines benchmarked:
  1. YOLOE-only          (detect/run_yolo.py)
  2. YOLOE + SigLIP       (detect/run_world.py)
  3. GroundingDINO-Base    (detect/run_world_Dino.py)
  4. WeDetect-Ref-Uni      (run_wedetect_ref_uni.py)

Usage:
  python benchmark_fps.py \
      --json_path yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json \
      --images_dir yolo_dataset/indoors_subset/images \
      --num_images 100 --warmup 10 \
      --wedetect_ref_checkpoint <path> \
      --wedetect_uni_checkpoint <path>
"""

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path
from contextlib import contextmanager

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

@contextmanager
def cuda_timer():
    """Context manager that yields a dict; on exit, stores elapsed_ms."""
    result = {}
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        yield result
        end_event.record()
        torch.cuda.synchronize()
        result["ms"] = start_event.elapsed_time(end_event)
    else:
        t0 = time.perf_counter()
        yield result
        result["ms"] = (time.perf_counter() - t0) * 1000.0


def load_image_list(json_path, images_dir, max_images):
    """Return list of (img_id, img_path, anns) tuples."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_to_anns = {}
    for item in data:
        img_id = str(item["image_id"])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)

    entries = []
    for img_id, anns in img_to_anns.items():
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            entries.append((img_id, img_path, anns))
        if max_images > 0 and len(entries) >= max_images:
            break
    return entries


# ---------------------------------------------------------------------------
# GFLOPs estimation helper (best-effort via fvcore or thop)
# ---------------------------------------------------------------------------

def estimate_gflops_fvcore(model, sample_inputs, label=""):
    """Try fvcore FlopCountAnalysis; return GFLOPs string or 'N/A'."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, sample_inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        gflops = flops.total() / 1e9
        return f"{gflops:.2f}"
    except Exception as e:
        return f"N/A ({e})"


def estimate_param_count(model):
    """Return total parameter count in millions."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6

# ---------------------------------------------------------------------------
# Pipeline 1: YOLOE-only
# ---------------------------------------------------------------------------

def bench_yoloe(entries, warmup, device):
    from ultralytics import YOLOE

    yolo_weights = SCRIPT_DIR / "yoloe-26l-seg.pt"
    model = YOLOE(str(yolo_weights)).to(device)

    params_m = estimate_param_count(model.model)

    # We need class names per image — build from first entry to set_classes
    def get_labels(anns):
        labels = set()
        for ann in anns:
            target = ann.get("keywords", {}).get("target", "")
            attrs = ann.get("keywords", {}).get("attributes", [])
            label = " ".join(attrs + [target]).strip() if attrs else target.strip()
            if label:
                labels.add(label)
        return list(labels)

    # Warmup
    for i in range(min(warmup, len(entries))):
        _, img_path, anns = entries[i]
        labels = get_labels(anns)
        if labels:
            model.set_classes(labels)
        model.predict(img_path, conf=0.05, verbose=False)

    # Timed run
    times_ms = []
    for _, img_path, anns in tqdm(entries, desc="[YOLOE] Benchmarking"):
        labels = get_labels(anns)
        if not labels:
            continue
        # here we don't count the encoding time into the total time
        model.set_classes(labels)
        with cuda_timer() as t:
            model.predict(img_path, conf=0.05, verbose=False)
        times_ms.append(t["ms"])

    # GFLOPs — use ultralytics built-in info if available
    gflops_str = "N/A"
    try:
        info = model.info(verbose=False)
        # model.info() returns (layers, params, gradients, gflops)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            gflops_str = f"{info[3]:.2f}"
    except Exception:
        pass

    return times_ms, params_m, gflops_str


# ---------------------------------------------------------------------------
# Pipeline 2: YOLOE + SigLIP  (run_world.py style)
# ---------------------------------------------------------------------------

def bench_yoloe_siglip(entries, warmup, device):
    from ultralytics import YOLOE
    from transformers import AutoProcessor, AutoModel

    # Load YOLOE
    yolo_weights = SCRIPT_DIR / "yoloe-26l-seg.pt"
    yolo_model = YOLOE(str(yolo_weights)).to(device)

    # Load SigLIP
    sys.path.insert(0, str(SCRIPT_DIR / "detect"))
    from run_world import resolve_siglip_source, expand_crop, siglip_verify
    siglip_source = resolve_siglip_source()
    siglip_processor = AutoProcessor.from_pretrained(siglip_source)
    siglip_model = AutoModel.from_pretrained(siglip_source).to(device)
    siglip_model.eval()

    params_m = estimate_param_count(yolo_model.model) + estimate_param_count(siglip_model)

    def get_labels(anns):
        labels = set()
        for ann in anns:
            target = ann.get("keywords", {}).get("target", "")
            attrs = ann.get("keywords", {}).get("attributes", [])
            label = " ".join(attrs + [target]).strip() if attrs else target.strip()
            if label:
                labels.add(label)
        return list(labels)

    # Warmup
    for i in range(min(warmup, len(entries))):
        _, img_path, anns = entries[i]
        labels = get_labels(anns)
        if labels:
            yolo_model.set_classes(labels)
        yolo_model.predict(img_path, conf=0.01, verbose=False)

    # Timed run: YOLOE predict + SigLIP verify on each low-conf box
    CONF_BYPASS = 0.4
    times_ms = []
    for _, img_path, anns in tqdm(entries, desc="[YOLOE+SigLIP] Benchmarking"):
        labels = get_labels(anns)
        if not labels:
            continue

        img_pil = Image.open(img_path).convert("RGB")
        yolo_model.set_classes(labels)

        with cuda_timer() as t:
            results = yolo_model.predict(img_path, conf=0.01, verbose=False)
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            # SigLIP verification on ambiguous boxes (conf < threshold)
            for p_box, p_cls, p_conf in zip(boxes, confs, classes):
                if float(p_conf) < CONF_BYPASS:
                    x1, y1, x2, y2 = map(int, p_box)
                    if x2 > x1 and y2 > y1:
                        crop_img = expand_crop(img_pil, [x1, y1, x2, y2], expansion_factor=1.5)
                        keyword = labels[int(p_cls)]
                        siglip_verify(crop_img, keyword, siglip_processor, siglip_model, device)

        times_ms.append(t["ms"])

    gflops_str = "N/A (composite)"
    return times_ms, params_m, gflops_str


# ---------------------------------------------------------------------------
# Pipeline 3: GroundingDINO-Base
# ---------------------------------------------------------------------------

def bench_grounding_dino(entries, warmup, device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    params_m = estimate_param_count(model)

    def get_prompt(anns):
        targets = set()
        for ann in anns:
            targets.add(ann.get("keywords", {}).get("target", ""))
        return " . ".join(t for t in targets if t)

    # Warmup
    for i in range(min(warmup, len(entries))):
        _, img_path, anns = entries[i]
        img_pil = Image.open(img_path).convert("RGB")
        prompt = get_prompt(anns)
        if not prompt:
            continue
        inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    # Timed run
    times_ms = []
    for _, img_path, anns in tqdm(entries, desc="[GroundingDINO] Benchmarking"):
        img_pil = Image.open(img_path).convert("RGB")
        prompt = get_prompt(anns)
        if not prompt:
            continue

        with cuda_timer() as t:
            inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                text_threshold=0.25,
                target_sizes=[img_pil.size[::-1]]
            )
        times_ms.append(t["ms"])

    # GFLOPs via fvcore
    gflops_str = "N/A"
    try:
        sample_img = Image.open(entries[0][1]).convert("RGB")
        sample_prompt = get_prompt(entries[0][2])
        sample_inputs = processor(images=sample_img, text=sample_prompt, return_tensors="pt").to(device)
        gflops_str = estimate_gflops_fvcore(model, (sample_inputs,), "GroundingDINO")
    except Exception:
        pass

    return times_ms, params_m, gflops_str


# ---------------------------------------------------------------------------
# Pipeline 4: WeDetect-Ref-Uni (proposal + Qwen grounding)
# ---------------------------------------------------------------------------

def bench_wedetect_ref(entries, warmup, device, args):
    from wedetect_ref.models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
    from wedetect_ref.models.vision_process import process_vision_info
    from transformers import AutoProcessor
    from generate_proposal import SimpleYOLOWorldDetector

    # Load detection model
    model_size = "base" if "base" in args.wedetect_uni_checkpoint else "large"
    det_model = SimpleYOLOWorldDetector(
        backbone_size=model_size, prompt_dim=768, num_prompts=256,
        num_proposals=args.num_proposals
    )
    checkpoint = torch.load(args.wedetect_uni_checkpoint, map_location="cpu")
    keys = list(checkpoint.keys())
    for key in keys:
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            checkpoint[new_key] = checkpoint.pop(key)
    keys = list(checkpoint.keys())
    for key in keys:
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            checkpoint[new_key] = checkpoint.pop(key)
    det_model = det_model.cuda().eval()
    det_model.load_state_dict(checkpoint, strict=False)

    # Load Qwen grounding model
    model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    grounding_model = Qwen3VLGroundingForConditionalGeneration.from_pretrained(
        args.wedetect_ref_checkpoint, **model_kwargs
    )
    grounding_processor = AutoProcessor.from_pretrained(args.wedetect_ref_checkpoint)
    object_token_index = grounding_processor.tokenizer.convert_tokens_to_ids("<object>")
    grounding_model.model.object_token_id = object_token_index
    grounding_model = grounding_model.cuda().eval()

    params_m = estimate_param_count(det_model) + estimate_param_count(grounding_model)

    def run_one_image(img_path, anns, image):
        """Full pipeline: proposal generation + grounding for each query."""
        with torch.no_grad():
            det_outputs = det_model([img_path])
        proposals = det_outputs[0]["bboxes"].float().cpu().numpy().tolist()
        if len(proposals) == 0:
            return

        # Run grounding for each annotation query
        for ann in anns:
            query = ann["phrase"]
            num_proposals = len(proposals)
            proposal_str = "<object>" * num_proposals
            ori_shape = [image.size]

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": copy.deepcopy(image)},
                    {"type": "text", "text": 'Please detect the "%s" in the image' % query},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": proposal_str}]},
            ]

            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            texts = [grounding_processor.apply_chat_template(messages, tokenize=False)]
            model_inputs = grounding_processor(
                text=texts, images=image_inputs, videos=video_inputs,
                return_tensors="pt", padding=True, do_resize=False,
            ).to(grounding_model.device)

            proposals_tensor = [torch.tensor(proposals).cuda().to(grounding_model.dtype)]
            with torch.inference_mode():
                grounding_model(
                    **model_inputs,
                    bboxes=copy.deepcopy(proposals_tensor),
                    ori_shapes=ori_shape,
                    bboxes_id=object_token_index,
                    image_inputs=image_inputs,
                )

    # Warmup
    for i in range(min(warmup, len(entries))):
        _, img_path, anns = entries[i]
        image = Image.open(img_path).convert("RGB")
        run_one_image(img_path, anns, image)

    # Timed run
    times_ms = []
    for _, img_path, anns in tqdm(entries, desc="[WeDetect-Ref] Benchmarking"):
        image = Image.open(img_path).convert("RGB")
        with cuda_timer() as t:
            run_one_image(img_path, anns, image)
        times_ms.append(t["ms"])

    gflops_str = "N/A (composite)"
    return times_ms, params_m, gflops_str


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results):
    print("\n" + "=" * 72)
    print(f"{'Pipeline':<25} {'Params(M)':>10} {'GFLOPs':>18} "
          f"{'Avg ms':>8} {'FPS':>8} {'Imgs':>5}")
    print("-" * 72)
    for name, times_ms, params_m, gflops_str in results:
        if len(times_ms) == 0:
            print(f"{name:<25} {'SKIPPED':>10}")
            continue
        arr = np.array(times_ms)
        avg_ms = arr.mean()
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"{name:<25} {params_m:>10.1f} {gflops_str:>18} "
              f"{avg_ms:>8.1f} {fps:>8.1f} {len(times_ms):>5}")
    print("=" * 72)

    # Per-pipeline detailed stats
    for name, times_ms, _, _ in results:
        if len(times_ms) == 0:
            continue
        arr = np.array(times_ms)
        print(f"\n[{name}] Latency distribution (ms):")
        print(f"  Mean: {arr.mean():.1f}  Median: {np.median(arr):.1f}  "
              f"Std: {arr.std():.1f}  Min: {arr.min():.1f}  Max: {arr.max():.1f}  "
              f"P95: {np.percentile(arr, 95):.1f}  P99: {np.percentile(arr, 99):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified FPS & GFLOPs benchmark")
    parser.add_argument("--json_path", type=str,
                        default="yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json")
    parser.add_argument("--images_dir", type=str,
                        default="yolo_dataset/indoors_subset/images")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Max images to benchmark (0 = all)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations before timing")

    # Pipelines to run (comma-separated or 'all')
    parser.add_argument("--pipelines", type=str, default="all",
                        help="Comma-separated list: yoloe,yoloe_siglip,dino,wedetect  or 'all'")

    # WeDetect-specific args
    parser.add_argument("--wedetect_ref_checkpoint", type=str, default="")
    parser.add_argument("--wedetect_uni_checkpoint", type=str, default="")
    parser.add_argument("--num_proposals", type=int, default=100)
    parser.add_argument("--score_thre", type=float, default=0.1)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load shared image list
    entries = load_image_list(args.json_path, args.images_dir, args.num_images)
    print(f"Loaded {len(entries)} images for benchmarking.")

    pipelines = (
        args.pipelines.split(",") if args.pipelines != "all"
        else ["yoloe", "yoloe_siglip", "dino", "wedetect"]
    )

    results = []

    # --- YOLOE ---
    if "yoloe" in pipelines:
        print("\n>>> Benchmarking YOLOE-only ...")
        times_ms, params_m, gflops_str = bench_yoloe(entries, args.warmup, device)
        results.append(("YOLOE-only", times_ms, params_m, gflops_str))

    # --- YOLOE + SigLIP ---
    if "yoloe_siglip" in pipelines:
        print("\n>>> Benchmarking YOLOE + SigLIP ...")
        times_ms, params_m, gflops_str = bench_yoloe_siglip(entries, args.warmup, device)
        results.append(("YOLOE + SigLIP", times_ms, params_m, gflops_str))

    # --- GroundingDINO ---
    if "dino" in pipelines:
        print("\n>>> Benchmarking GroundingDINO-Base ...")
        times_ms, params_m, gflops_str = bench_grounding_dino(entries, args.warmup, device)
        results.append(("GroundingDINO-Base", times_ms, params_m, gflops_str))

    # --- WeDetect-Ref ---
    if "wedetect" in pipelines:
        if not args.wedetect_ref_checkpoint or not args.wedetect_uni_checkpoint:
            print("\n>>> Skipping WeDetect-Ref (no checkpoints provided)")
            results.append(("WeDetect-Ref-Uni", [], 0, "N/A"))
        else:
            print("\n>>> Benchmarking WeDetect-Ref-Uni ...")
            times_ms, params_m, gflops_str = bench_wedetect_ref(
                entries, args.warmup, device, args
            )
            results.append(("WeDetect-Ref-Uni", times_ms, params_m, gflops_str))

    print_report(results)


if __name__ == "__main__":
    main()
