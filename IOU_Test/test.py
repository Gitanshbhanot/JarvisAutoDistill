#!/usr/bin/env python3
"""
YOLO Evaluation Script for Standard Dataset Layout
==================================================
Layout expected for both GT and Predictions:
  <root>/
    images/
      *.jpg|*.png|...
    labels/
      *.txt  (YOLO format: class x_center y_center width height [confidence])

You provide two roots: --gt_root (human annotations) and --pred_root (LLM annotations).
The script:
  - Builds the union of image stems from <root>/images and <root>/labels
  - Reads labels if present, else treats as zero boxes
  - (NEW) Optional prediction class remapping via --pred_map (e.g. 0:2,1:2)
  - (NEW) Optional class filtering via --cls_filter (keep only certain ids)
  - Matches predictions to GT per class with an IoU threshold (or thresholds)
    * (NEW) Optional --class_agnostic_debug to ignore class during matching for diagnostics
  - Computes:
      * IoU per matched pair
      * TP/FP/FN, Precision/Recall/F1 per IoU threshold
      * Image-level coverage stats (images with preds, images missed, TP-image-rate)
      * Optional per-class metrics
      * Optional mAP@0.50 and COCO mAP@[.50:.95] (requires confidences for proper PR)
  - Writes summary.json plus CSVs (matches/errors/per-class/PR curves)

Examples:
  python yolo_eval.py \
    --gt_root /data/human_dataset \
    --pred_root /data/llm_dataset \
    --out_dir ./eval_out \
    --iou_thresholds 0.50 0.75 \
    --per_class \
    --map --map_coco \
    --names /data/classes.names \
    --pred_map 0:2 \
    --cls_filter 2
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

# ---------------------------- Data structures ----------------------------

@dataclass
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0
    line_num: int = -1

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.x - self.w / 2.0
        y1 = self.y - self.h / 2.0
        x2 = self.x + self.w / 2.0
        y2 = self.y + self.h / 2.0
        # clamp
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        return x1, y1, x2, y2

# ---------------------------- IO helpers ----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def list_image_stems(images_dir: str) -> Set[str]:
    if not images_dir or not os.path.isdir(images_dir):
        return set()
    return {stem(f) for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS}

def list_label_stems(labels_dir: str) -> Set[str]:
    if not labels_dir or not os.path.isdir(labels_dir):
        return set()
    return {stem(f) for f in os.listdir(labels_dir) if f.endswith(".txt")}

def parse_label_file(path: str, is_pred: bool) -> List[Box]:
    boxes: List[Box] = []
    if not path or not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if (is_pred and len(parts) >= 6) else 1.0
                boxes.append(Box(cls, x, y, w, h, conf, line_num=i))
            except Exception:
                # skip malformed line
                continue
    return boxes

def load_names(path: Optional[str]) -> Dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                out[idx] = name
    return out

# ---------------------------- Geometry + matching ----------------------------

def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def greedy_match(gts: List[Box], preds: List[Box], iou_thr: float, class_aware: bool = True
                 ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    g_xy = [g.to_xyxy() for g in gts]
    p_xy = [p.to_xyxy() for p in preds]
    pairs: List[Tuple[int, int, float]] = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if class_aware and g.cls != p.cls:
                continue
            val = iou_xyxy(g_xy[gi], p_xy[pi])
            if val >= iou_thr:
                pairs.append((gi, pi, val))
    # sort by IoU then prediction confidence
    pairs.sort(key=lambda t: (t[2], preds[t[1]].conf), reverse=True)

    matched_g, matched_p = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for gi, pi, val in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi); matched_p.add(pi)
        matches.append((gi, pi, val))

    un_g = [i for i in range(len(gts)) if i not in matched_g]
    un_p = [i for i in range(len(preds)) if i not in matched_p]
    return matches, un_g, un_p

# ---------------------------- Metrics ----------------------------

def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0

def compute_pr_ap(all_conf_and_tp: List[Tuple[float, int]], total_gt: int
                  ) -> Tuple[List[float], List[float], float]:
    # COCO 101-point interpolation
    if total_gt == 0:
        return [], [], 0.0
    arr = sorted(all_conf_and_tp, key=lambda t: t[0], reverse=True)
    tps = fps = 0
    precisions: List[float] = []
    recalls: List[float] = []
    for conf, is_tp in arr:
        if is_tp: tps += 1
        else:     fps += 1
        precisions.append(tps / (tps + fps))
        recalls.append(tps / total_gt)
    ap = 0.0
    for r in [i / 100.0 for i in range(101)]:
        p = 0.0
        for pr, rc in zip(precisions, recalls):
            if rc >= r and pr > p:
                p = pr
        ap += p
    ap /= 101.0
    return recalls, precisions, ap

# ---------------------------- Util ----------------------------

def _parse_map(s: Optional[str]) -> Dict[int, int]:
    """
    Parse mapping string like "0:2,1:2" -> {0:2, 1:2}
    """
    if not s:
        return {}
    out: Dict[int, int] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        a, b = pair.split(":")
        out[int(a)] = int(b)
    return out

# ---------------------------- Main ----------------------------

def main():
    import csv

    parser = argparse.ArgumentParser("Evaluate YOLO predictions (LLM) vs human GT with standard dataset layout.")
    parser.add_argument("--gt_root", required=True, help="GT dataset root containing images/ and labels/")
    parser.add_argument("--pred_root", required=True, help="Pred dataset root containing images/ and labels/")
    parser.add_argument("--out_dir", default="./eval_out", help="Output directory for reports")
    parser.add_argument("--names", default=None, help="Optional classes.names (one class per line)")
    parser.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.50],
                        help="IoU thresholds, e.g. 0.50 0.75")
    parser.add_argument("--per_class", action="store_true", help="Write per-class metrics")
    parser.add_argument("--map", action="store_true", help="Compute mAP@0.50")
    parser.add_argument("--map_coco", action="store_true", help="Compute mAP@[.50:.95]")
    # NEW:
    parser.add_argument("--cls_filter", type=int, nargs="+", default=None,
                        help="Keep only these class ids (e.g. --cls_filter 2 or --cls_filter 0 1)")
    parser.add_argument("--pred_map", type=str, default=None,
                        help="Remap prediction class ids, e.g. '0:2,1:2'")
    parser.add_argument("--gt_map", type=str, default=None,
                    help="Remap GT class ids, e.g. '0:2,1:2'")
    parser.add_argument("--class_agnostic_debug", action="store_true",
                        help="Ignore class during matching (debug/diagnostics).")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    class_names = load_names(args.names)
    pred_remap = _parse_map(args.pred_map)
    gt_remap = _parse_map(args.gt_map)
    keep_cls = set(args.cls_filter) if args.cls_filter is not None else None

    # Resolve dirs
    gt_images = os.path.join(args.gt_root, "images")
    gt_labels = os.path.join(args.gt_root, "labels")
    pr_images = os.path.join(args.pred_root, "images")
    pr_labels = os.path.join(args.pred_root, "labels")

    # Build union of stems (consider both images and labels in each root)
    stems = set()
    stems |= list_image_stems(gt_images)
    stems |= list_label_stems(gt_labels)
    stems |= list_image_stems(pr_images)
    stems |= list_label_stems(pr_labels)
    basenames = sorted(stems)

    # Load annotations per stem (absent file -> zero boxes)
    gt_by_img: Dict[str, List[Box]] = {}
    pred_by_img: Dict[str, List[Box]] = {}
    for s in basenames:
        gt_txt = os.path.join(gt_labels, f"{s}.txt")
        pr_txt = os.path.join(pr_labels, f"{s}.txt")
        gt_boxes = parse_label_file(gt_txt, is_pred=False)
        pred_boxes = parse_label_file(pr_txt, is_pred=True)

        # NEW: Apply GT class remapping first
        if gt_remap:
            for b in gt_boxes:
                if b.cls in gt_remap:
                    b.cls = gt_remap[b.cls]

        # Apply prediction class remapping
        if pred_remap:
            for b in pred_boxes:
                if b.cls in pred_remap:
                    b.cls = pred_remap[b.cls]

        # Apply optional class filter (after *both* remaps)
        if keep_cls is not None:
            gt_boxes = [b for b in gt_boxes if b.cls in keep_cls]
            pred_boxes = [b for b in pred_boxes if b.cls in keep_cls]

        gt_by_img[s] = gt_boxes
        pred_by_img[s] = pred_boxes


    # Image-level accounting (based on GT image presence)
    gt_image_stems = list_image_stems(gt_images)
    pred_image_stems = list_image_stems(pr_images)
    images_with_gt = len(gt_image_stems)
    images_with_pred = len(pred_image_stems & gt_image_stems)  # preds that correspond to a GT image stem
    images_with_zero_preds_but_gt = sum(
        1 for s in gt_image_stems if len(pred_by_img.get(s, [])) == 0
    )

    # Per-threshold accumulators
    iou_thresholds = sorted(set(args.iou_thresholds))
    per_thr_metrics = {thr: {"TP": 0, "FP": 0, "FN": 0, "matches": []} for thr in iou_thresholds}
    per_thr_class_metrics = {thr: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "IoUs": []})
                             for thr in iou_thresholds}
    per_thr_class_confTP = {thr: defaultdict(list) for thr in (iou_thresholds if (args.map or args.map_coco) else [])}
    gt_counts_per_class = Counter()
    images_with_any_tp = {thr: 0 for thr in iou_thresholds}

    # Count GT instances per class (after any filtering)
    for s in basenames:
        for g in gt_by_img[s]:
            gt_counts_per_class[g.cls] += 1

    # Start matching
    for s in basenames:
        gts = gt_by_img[s]
        preds = pred_by_img[s]

        # Per-image TP flag per thr
        img_tp_flag = {thr: False for thr in iou_thresholds}

        if not args.class_agnostic_debug:
            # class-aware matching per class
            g_by_cls = defaultdict(list)
            p_by_cls = defaultdict(list)
            for i, g in enumerate(gts): g_by_cls[g.cls].append((i, g))
            for j, p in enumerate(preds): p_by_cls[p.cls].append((j, p))

            for thr in iou_thresholds:
                for cls in set(list(g_by_cls.keys()) + list(p_by_cls.keys())):
                    g_idx, g_list = zip(*g_by_cls[cls]) if cls in g_by_cls else ([], [])
                    p_idx, p_list = zip(*p_by_cls[cls]) if cls in p_by_cls else ([], [])
                    g_list = list(g_list); p_list = list(p_list)

                    matches, un_g, un_p = greedy_match(g_list, p_list, thr, class_aware=True)

                    # TPs
                    for (lg, lp, iou_val) in matches:
                        gi = g_idx[lg] if g_idx else None
                        pi = p_idx[lp] if p_idx else None
                        if gi is None or pi is None:
                            continue
                        per_thr_metrics[thr]["TP"] += 1
                        per_thr_metrics[thr]["matches"].append({
                            "stem": s,
                            "gt_cls": gts[gi].cls,
                            "pred_cls": preds[pi].cls,
                            "pred_conf": preds[pi].conf,
                            "gt_line": gts[gi].line_num,
                            "pred_line": preds[pi].line_num,
                            "iou": iou_val
                        })
                        per_thr_class_metrics[thr][gts[gi].cls]["TP"] += 1
                        per_thr_class_metrics[thr][gts[gi].cls]["IoUs"].append(iou_val)
                        img_tp_flag[thr] = True
                        if thr in per_thr_class_confTP:
                            per_thr_class_confTP[thr][gts[gi].cls].append((preds[pi].conf, 1))

                    # FNs
                    for lg in un_g:
                        gi = g_idx[lg] if g_idx else None
                        if gi is None: continue
                        per_thr_metrics[thr]["FN"] += 1
                        per_thr_class_metrics[thr][gts[gi].cls]["FN"] += 1

                    # FPs
                    for lp in un_p:
                        pi = p_idx[lp] if p_idx else None
                        if pi is None: continue
                        per_thr_metrics[thr]["FP"] += 1
                        per_thr_class_metrics[thr][preds[pi].cls]["FP"] += 1
                        if thr in per_thr_class_confTP:
                            per_thr_class_confTP[thr][preds[pi].cls].append((preds[pi].conf, 0))
        else:
            # Class-agnostic matching (debug only)
            for thr in iou_thresholds:
                matches, un_g_idx, un_p_idx = greedy_match(gts, preds, thr, class_aware=False)

                # TPs
                for (gi, pi, iou_val) in matches:
                    per_thr_metrics[thr]["TP"] += 1
                    per_thr_metrics[thr]["matches"].append({
                        "stem": s,
                        "gt_cls": gts[gi].cls,
                        "pred_cls": preds[pi].cls,
                        "pred_conf": preds[pi].conf,
                        "gt_line": gts[gi].line_num,
                        "pred_line": preds[pi].line_num,
                        "iou": iou_val
                    })
                    per_thr_class_metrics[thr][gts[gi].cls]["TP"] += 1
                    per_thr_class_metrics[thr][gts[gi].cls]["IoUs"].append(iou_val)
                    img_tp_flag[thr] = True
                    if thr in per_thr_class_confTP:
                        per_thr_class_confTP[thr][gts[gi].cls].append((preds[pi].conf, 1))

                # FNs
                for gi in un_g_idx:
                    per_thr_metrics[thr]["FN"] += 1
                    per_thr_class_metrics[thr][gts[gi].cls]["FN"] += 1

                # FPs
                for pi in un_p_idx:
                    per_thr_metrics[thr]["FP"] += 1
                    per_thr_class_metrics[thr][preds[pi].cls]["FP"] += 1
                    if thr in per_thr_class_confTP:
                        per_thr_class_confTP[thr][preds[pi].cls].append((preds[pi].conf, 0))

        for thr in iou_thresholds:
            if img_tp_flag[thr] and (s in gt_image_stems):
                images_with_any_tp[thr] += 1

    # Build summaries
    summary = {"iou_thresholds": iou_thresholds, "overall": {}, "per_class": {}, "images": {}}

    # Image coverage stats (only considering stems that exist as GT images)
    detection_coverage = safe_div(images_with_pred, images_with_gt)
    for thr in iou_thresholds:
        tp_image_rate = safe_div(images_with_any_tp[thr], images_with_gt)
        summary["images"][thr] = {
            "images_with_gt": images_with_gt,
            "images_with_pred": images_with_pred,
            "images_with_zero_preds_but_gt": images_with_zero_preds_but_gt,
            "images_with_any_true_positive": images_with_any_tp[thr],
            "detection_coverage_rate": detection_coverage,
            "tp_image_rate": tp_image_rate
        }

    # Overall box-level metrics
    for thr in iou_thresholds:
        TP = per_thr_metrics[thr]["TP"]; FP = per_thr_metrics[thr]["FP"]; FN = per_thr_metrics[thr]["FN"]
        precision = safe_div(TP, TP + FP)
        recall    = safe_div(TP, TP + FN)
        f1        = safe_div(2 * precision * recall, precision + recall)
        avg_iou   = (sum(m["iou"] for m in per_thr_metrics[thr]["matches"]) / len(per_thr_metrics[thr]["matches"])
                     ) if per_thr_metrics[thr]["matches"] else 0.0
        summary["overall"][thr] = {
            "TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "f1": f1,
            "avg_iou_of_matches": avg_iou
        }

    # Per-class metrics
    if args.per_class:
        for thr in iou_thresholds:
            per_cls = {}
            for cls, stats in per_thr_class_metrics[thr].items():
                TP = stats["TP"]; FP = stats["FP"]; FN = stats["FN"]
                prec = safe_div(TP, TP + FP)
                rec  = safe_div(TP, TP + FN)
                f1   = safe_div(2 * prec * rec, prec + rec)
                avg_iou = sum(stats["IoUs"]) / len(stats["IoUs"]) if stats["IoUs"] else 0.0
                per_cls[cls] = {
                    "name": class_names.get(cls, str(cls)),
                    "TP": TP, "FP": FP, "FN": FN,
                    "precision": prec, "recall": rec, "f1": f1,
                    "avg_iou_of_matches": avg_iou,
                    "gt_count": gt_counts_per_class.get(cls, 0)
                }
            summary["per_class"][thr] = per_cls

    # mAP
    map_results = {}
    if (args.map or args.map_coco):
        map_thresholds = [0.50] if args.map and not args.map_coco else [round(0.5 + i*0.05, 2) for i in range(10)]
        # Ensure conf/TP lists exist for each threshold
        for thr in map_thresholds:
            if thr not in per_thr_class_confTP:
                per_thr_class_confTP[thr] = defaultdict(list)
                # Re-match at this thr to populate
                for s in basenames:
                    gts = gt_by_img[s]; preds = pred_by_img[s]
                    if args.class_agnostic_debug:
                        matches, un_g, un_p = greedy_match(gts, preds, thr, class_aware=False)
                        matched_p_locals = {m[1] for m in matches}
                        for gi, lp, _ in matches:
                            p = preds[lp]
                            per_thr_class_confTP[thr][gts[gi].cls].append((p.conf, 1))
                        for idx, p in enumerate(preds):
                            if idx not in matched_p_locals:
                                per_thr_class_confTP[thr][p.cls].append((p.conf, 0))
                    else:
                        g_by_cls = defaultdict(list); p_by_cls = defaultdict(list)
                        for i, g in enumerate(gts): g_by_cls[g.cls].append((i, g))
                        for j, p in enumerate(preds): p_by_cls[p.cls].append((j, p))
                        for cls in set(list(g_by_cls.keys()) + list(p_by_cls.keys())):
                            g_idx, g_list = zip(*g_by_cls[cls]) if cls in g_by_cls else ([], [])
                            p_idx, p_list = zip(*p_by_cls[cls]) if cls in p_by_cls else ([], [])
                            g_list = list(g_list); p_list = list(p_list)
                            matches, un_g, un_p = greedy_match(g_list, p_list, thr, class_aware=True)
                            matched_p_locals = {m[1] for m in matches}
                            for _, lp, _ in matches:
                                p = p_list[lp]
                                per_thr_class_confTP[thr][cls].append((p.conf, 1))
                            for idx, p in enumerate(p_list):
                                if idx not in matched_p_locals:
                                    per_thr_class_confTP[thr][cls].append((p.conf, 0))
        # Compute AP per class and mAP
        for thr in map_thresholds:
            ap_per_class = {}
            for cls in set(list(gt_counts_per_class.keys()) + list(per_thr_class_confTP[thr].keys())):
                total_gt = gt_counts_per_class.get(cls, 0)
                recalls, precisions, ap = compute_pr_ap(per_thr_class_confTP[thr].get(cls, []), total_gt)
                ap_per_class[cls] = {"name": class_names.get(cls, str(cls)), "AP": ap, "gt_count": total_gt}
                # Optionally dump PR curves
                pr_path = os.path.join(args.out_dir, f"pr_curves_{str(thr).replace('.','_')}.csv")
                if recalls and precisions:
                    with open(pr_path, "a", encoding="utf-8") as prf:
                        for rc, pr in zip(recalls, precisions):
                            prf.write(f"{cls},{rc:.6f},{pr:.6f}\n")
            valid = [c for c, n in gt_counts_per_class.items() if n > 0]
            mAP = sum(ap_per_class[c]["AP"] for c in valid) / len(valid) if valid else 0.0
            map_results[thr] = {"mAP": mAP, "per_class_AP": ap_per_class}
        summary["mAP"] = map_results

    # ----------- Write outputs -----------
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Matches / Errors / Per-class CSVs
    for thr in iou_thresholds:
        matches_path = os.path.join(args.out_dir, f"matches_{str(thr).replace('.','_')}.csv")
        with open(matches_path, "w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow(["stem", "gt_cls", "pred_cls", "pred_conf", "iou", "gt_line", "pred_line"])
            for m in per_thr_metrics[thr]["matches"]:
                writer.writerow([m["stem"], m["gt_cls"], m["pred_cls"],
                                 f"{m['pred_conf']:.6f}", f"{m['iou']:.6f}",
                                 m["gt_line"], m["pred_line"]])
        # Recompute unmatched for error listing
        errors_path = os.path.join(args.out_dir, f"errors_{str(thr).replace('.','_')}.csv")
        with open(errors_path, "w", newline="", encoding="utf-8") as ef:
            writer = csv.writer(ef)
            writer.writerow(["error_type", "stem", "cls", "conf_or_nan", "line_num_or_nan"])
            for s in basenames:
                gts = gt_by_img[s]; preds = pred_by_img[s]
                if not args.class_agnostic_debug:
                    g_by_cls = defaultdict(list); p_by_cls = defaultdict(list)
                    for i, g in enumerate(gts): g_by_cls[g.cls].append((i, g))
                    for j, p in enumerate(preds): p_by_cls[p.cls].append((j, p))
                    for cls in set(list(g_by_cls.keys()) + list(p_by_cls.keys())):
                        g_idx, g_list = zip(*g_by_cls[cls]) if cls in g_by_cls else ([], [])
                        p_idx, p_list = zip(*p_by_cls[cls]) if cls in p_by_cls else ([], [])
                        g_list = list(g_list); p_list = list(p_list)
                        matches, un_g, un_p = greedy_match(g_list, p_list, thr, class_aware=True)
                        for lg in un_g:
                            gi = g_idx[lg] if g_idx else None
                            if gi is None: continue
                            writer.writerow(["FN", s, gts[gi].cls, "nan", gts[gi].line_num])
                        for lp in un_p:
                            pi = p_idx[lp] if p_idx else None
                            if pi is None: continue
                            writer.writerow(["FP", s, preds[pi].cls, f"{preds[pi].conf:.6f}", preds[pi].line_num])
                else:
                    matches, un_g, un_p = greedy_match(gts, preds, thr, class_aware=False)
                    matched_g = {m[0] for m in matches}
                    matched_p = {m[1] for m in matches}
                    for gi in range(len(gts)):
                        if gi not in matched_g:
                            writer.writerow(["FN", s, gts[gi].cls, "nan", gts[gi].line_num])
                    for pi in range(len(preds)):
                        if pi not in matched_p:
                            writer.writerow(["FP", s, preds[pi].cls, f"{preds[pi].conf:.6f}", preds[pi].line_num])

    if args.per_class:
        for thr in iou_thresholds:
            path = os.path.join(args.out_dir, f"per_class_{str(thr).replace('.','_')}.csv")
            with open(path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(["cls", "name", "TP", "FP", "FN", "precision", "recall", "f1", "avg_iou", "gt_count"])
                for cls, stats in sorted(summary["per_class"][thr].items(), key=lambda kv: kv[0]):
                    writer.writerow([
                        cls, stats["name"], stats["TP"], stats["FP"], stats["FN"],
                        f"{stats['precision']:.6f}", f"{stats['recall']:.6f}", f"{stats['f1']:.6f}",
                        f"{stats['avg_iou_of_matches']:.6f}", stats["gt_count"]
                    ])

    # ----------- Console summary -----------
    print("==== YOLO Evaluation Summary ====")
    print(f"GT images: {images_with_gt} | Predicted images (that align to GT stems): {images_with_pred} | "
          f"GT images with zero predictions: {images_with_zero_preds_but_gt}")
    for thr in iou_thresholds:
        o = summary["overall"][thr]
        img = summary["images"][thr]
        print(f"IoU@{thr:.2f}: TP={o['TP']} FP={o['FP']} FN={o['FN']} | "
              f"P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f} | "
              f"AvgIoU={o['avg_iou_of_matches']:.4f}")
        print(f"  Image coverage: {img['detection_coverage_rate']:.4f} "
              f"(images_with_pred / images_with_gt) | "
              f"TP-image-rate@{thr:.2f}: {img['tp_image_rate']:.4f}")

    if "mAP" in summary:
        for thr, res in summary["mAP"].items():
            print(f"mAP@{thr:.2f} = {res['mAP']:.4f}")

    print(f"\nReports written to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()


# python3 test.py \
#   --gt_root ./test \
#   --pred_root ./arm \
#   --out_dir ./eval_out \
#   --iou_thresholds 0.50 0.75 \
#   --per_class \
#   --map \
#   --map_coco \
#   --names ./classes.names