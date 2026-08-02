"""
Evaluate generated images using HuggingFace Mask2Former + OpenAI CLIP.

Drop-in replacement for evaluate_images.py that uses HuggingFace transformers
instead of mmdet. No mmdet/mmcv dependency required.

Usage:
    python evaluate_images_hf.py <imagedir> \
        --outfile results.jsonl \
        --detector facebook/mask2former-swin-large-coco-instance

    # Then summarize:
    python summary_scores.py results.jsonl

Dependencies:
    pip install transformers torch torchvision open_clip_torch clip-benchmark Pillow pandas
"""

import argparse
import json
import os
import re
import sys
import time

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GenEval images using HuggingFace Mask2Former + CLIP"
    )
    parser.add_argument("imagedir", type=str, help="Directory with generated images")
    parser.add_argument("--outfile", type=str, default="results.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--detector", type=str,
                        default="facebook/mask2former-swin-large-coco-instance",
                        help="HuggingFace model ID for instance segmentation")
    parser.add_argument("--clip-model", type=str, default="ViT-L-14",
                        help="OpenCLIP model architecture")
    parser.add_argument("--clip-pretrained", type=str, default="openai",
                        help="OpenCLIP pretrained weights name or local path")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--counting-threshold", type=float, default=0.9,
                        help="Stricter threshold for counting tasks")
    parser.add_argument("--max-objects", type=int, default=16,
                        help="Max detections per class")
    parser.add_argument("--nms-threshold", type=float, default=1.0,
                        help="NMS IoU threshold (1.0 = disabled)")
    parser.add_argument("--position-threshold", type=float, default=0.1,
                        help="Spatial tolerance for position checks")
    parser.add_argument("--bgcolor", type=str, default="#999",
                        help="Background color for masked crops")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    args = parser.parse_args()
    return args


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_detector(model_id, device):
    """Load HuggingFace Mask2Former for instance segmentation."""
    from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

    print(f"Loading detector: {model_id}", file=sys.stderr)
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    model = model.to(device).eval()
    print(f"Detector loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    return model, processor


def load_clip(arch, pretrained, device):
    """Load OpenCLIP model for color classification.

    Args:
        arch: Model architecture (e.g. "ViT-L-14")
        pretrained: Pretrained weights name ("openai") or local path to checkpoint
        device: torch device
    """
    print(f"Loading CLIP: {arch} (pretrained={pretrained})", file=sys.stderr)
    t0 = time.time()
    model, _, transform = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(arch)
    print(f"CLIP loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    return model, transform, tokenizer


def load_classnames():
    """Load COCO class names from object_names.txt."""
    names_file = os.path.join(os.path.dirname(__file__), "object_names.txt")
    with open(names_file) as f:
        return [line.strip() for line in f]


# ──────────────────────────────────────────────
# HuggingFace Mask2Former inference
# ──────────────────────────────────────────────

def run_detector(image: Image.Image, model, processor, device, classnames):
    """Run Mask2Former instance segmentation via HuggingFace.

    Returns:
        dict mapping classname -> list of (bbox_array_5, mask_or_None)
        where bbox_array_5 is [x1, y1, x2, y2, confidence]
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: get instance segmentation results
    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]]  # (height, width)
    )[0]

    # result has 'segments_info' (list of dicts) and 'segmentation' (H, W tensor)
    seg_map = result["segmentation"].cpu().numpy()  # (H, W) with instance IDs
    segments = result["segments_info"]  # list of {id, label_id, score}

    # Build label_id -> GenEval classname mapping.
    # HF Mask2Former uses PASCAL VOC names (e.g. "motorbike", "aeroplane", "sofa")
    # while GenEval uses COCO names (e.g. "motorcycle", "airplane", "couch").
    # Both have 80 classes in the same order, so we map by index.
    label2classname = {}
    for label_id in model.config.id2label:
        if label_id < len(classnames):
            label2classname[label_id] = classnames[label_id]

    detections = {}  # classname -> [(bbox_5, mask), ...]

    for seg in segments:
        label_id = seg["label_id"]
        score = seg["score"]
        instance_id = seg["id"]

        classname = label2classname.get(label_id, None)
        if classname is None:
            continue

        # Extract mask for this instance
        mask = (seg_map == instance_id).astype(np.uint8) * 255  # (H, W) uint8

        # Compute bbox from mask
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = np.array([x1, y1, x2, y2, score], dtype=np.float32)

        if classname not in detections:
            detections[classname] = []
        detections[classname].append((bbox, mask))

    # Sort each class by confidence (descending)
    for classname in detections:
        detections[classname].sort(key=lambda x: -x[0][4])

    return detections


# ──────────────────────────────────────────────
# Color classification (same as original)
# ──────────────────────────────────────────────

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform, bgcolor="#999"):
        self._image = image.convert("RGB")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            mask_pil = Image.fromarray(mask)
            image = Image.composite(self._image, self._blank, mask_pil)
        else:
            image = self._image
        image = image.crop(box[:4])
        return (self._transform(image), 0)


def color_classification(image, bboxes, classname, clip_model, clip_tokenizer, transform, bgcolor, device):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model, clip_tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            device
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes, transform, bgcolor),
        batch_size=16, num_workers=4
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, device)
        return [COLORS[index.item()] for index in pred.argmax(1)]


# ──────────────────────────────────────────────
# Evaluation logic (same as original)
# ──────────────────────────────────────────────

def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b, position_threshold):
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def evaluate(image, objects, metadata, clip_model, clip_tokenizer, clip_transform,
             bgcolor, position_threshold, device):
    correct = True
    reason = []
    matched_groups = []
    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                colors = color_classification(
                    image, found_objects, classname,
                    clip_model, clip_tokenizer, clip_transform, bgcolor, device
                )
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found "
                        f"{colors.count(req['color'])} {req['color']}; and "
                        + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if 'position' in req and matched:
                expected_rel, target_group = req['position']
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj, position_threshold)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found "
                                    + f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def evaluate_image(filepath, metadata, detector_model, detector_processor,
                   clip_model, clip_tokenizer, clip_transform, classnames, args, device):
    """Run detection + evaluation on a single image."""
    image = ImageOps.exif_transpose(Image.open(filepath)).convert("RGB")

    # Run HF Mask2Former
    raw_detections = run_detector(image, detector_model, detector_processor, device, classnames)

    # Apply thresholds and NMS (same logic as original)
    confidence_threshold = args.threshold if metadata['tag'] != "counting" else args.counting_threshold
    detected = {}

    for classname, det_list in raw_detections.items():
        # Filter by confidence
        filtered = [(bbox, mask) for bbox, mask in det_list if bbox[4] > confidence_threshold]
        # Limit
        filtered = filtered[:args.max_objects]
        # NMS
        kept = []
        remaining = list(filtered)
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            if args.nms_threshold < 1.0:
                remaining = [
                    det for det in remaining
                    if compute_iou(best[0], det[0]) < args.nms_threshold
                ]
        if kept:
            detected[classname] = kept

    is_correct, reason = evaluate(
        image, detected, metadata,
        clip_model, clip_tokenizer, clip_transform,
        args.bgcolor, args.position_threshold, device
    )
    return {
        'filename': filepath,
        'tag': metadata['tag'],
        'prompt': metadata['prompt'],
        'correct': is_correct,
        'reason': reason,
        'metadata': json.dumps(metadata),
        'details': json.dumps({
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        })
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=sys.stderr)

    # Load models
    detector_model, detector_processor = load_detector(args.detector, device)
    clip_model, clip_transform, clip_tokenizer = load_clip(args.clip_model, args.clip_pretrained, device)
    classnames = load_classnames()

    # Evaluate all images
    full_results = []
    folders = sorted([
        f for f in os.listdir(args.imagedir)
        if os.path.isdir(os.path.join(args.imagedir, f)) and f.isdigit()
    ])

    print(f"Found {len(folders)} prompt folders in {args.imagedir}", file=sys.stderr)

    for subfolder in folders:
        folderpath = os.path.join(args.imagedir, subfolder)
        metadata_path = os.path.join(folderpath, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            print(f"Skipping {subfolder}: no metadata.jsonl", file=sys.stderr)
            continue
        with open(metadata_path) as fp:
            metadata = json.load(fp)

        samples_dir = os.path.join(folderpath, "samples")
        if not os.path.isdir(samples_dir):
            continue

        for imagename in sorted(os.listdir(samples_dir)):
            imagepath = os.path.join(samples_dir, imagename)
            if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                continue
            result = evaluate_image(
                imagepath, metadata,
                detector_model, detector_processor,
                clip_model, clip_tokenizer, clip_transform,
                classnames, args, device
            )
            full_results.append(result)
            status = "✓" if result["correct"] else "✗"
            print(f"  {status} {subfolder}/{imagename} [{metadata['tag']}]", file=sys.stderr)

    # Save results
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(full_results).to_json(fp, orient="records", lines=True)

    # Quick summary
    df = pd.DataFrame(full_results)
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Total images: {len(df)}", file=sys.stderr)
    print(f"% correct: {df['correct'].mean():.2%}", file=sys.stderr)
    for tag, task_df in df.groupby('tag', sort=False):
        print(f"  {tag:<16} = {task_df['correct'].mean():.2%} "
              f"({task_df['correct'].sum()}/{len(task_df)})", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)


if __name__ == "__main__":
    main()
