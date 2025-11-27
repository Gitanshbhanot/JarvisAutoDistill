from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
from collections import defaultdict
from annotate.detectors.gemini import GeminiDetection

@dataclass
class TwoStageConfig:
    stage1_thresholds: Dict[str, float] = field(default_factory=lambda: {"Safety Helmet": 0.25, "High-Visibility Jacket": 0.20})
    stage2_thresholds: Dict[str, float] = field(default_factory=lambda: {"Safety Helmet": 0.50, "High-Visibility Jacket": 0.45})
    crop_margin: float = 0.15
    confirm_iou: float = 0.50
    wbf_iou: float = 0.60
    nms_iou: float = 0.55
    min_area_frac: float = 0.0015
    do_tta_flip: bool = False
    use_wbf: bool = True
    max_proposals_per_class: int = 300

class TwoStageRefiner:
    """
    Two-stage object detection refiner for enhancing detections from a base detector.
    """

    def run(
        self,
        detector: Any,  # GeminiDetection-compatible instance
        image: Image.Image,
        class_names: List[str],
        class_ids: List[int],
        cfg: TwoStageConfig,
    ) -> Dict[str, Any]:
        """
        Run the two-stage refinement pipeline.

        Args:
            detector: Base detector instance with .detector method.
            image: Input PIL Image.
            class_names: List of class names.
            class_ids: Corresponding class IDs.
            cfg: Configuration for the two-stage process.

        Returns:
            Dict with "abs": List[Dict[label, class_id, score, box:[ymin,xmin,ymax,xmax]] sorted by score desc (secondary -area, x, y),
                 "yolo": List[[class_id, xc, yc, w, h]] normalized to original image.
        """
        img_w, img_h = image.size
        name_to_id = dict(zip(class_names, class_ids))
        original_th = detector.confidence_threshold

        # Stage 1: Collect proposals per class with lower thresholds
        proposals: List[Dict[str, Any]] = []
        for i, name in enumerate(class_names):
            th = cfg.stage1_thresholds.get(name, 0.3)
            detector.confidence_threshold = th
            ok, yolo, labels_str = detector.detector(image, [name], [class_ids[i]])
            if not ok:
                continue
            labels = [l.strip() for l in labels_str.split(",") if l.strip()]
            if len(labels) != len(yolo):
                continue
            for j, yl in enumerate(yolo):
                class_id = int(yl[0])
                box_norm = yl[1:]
                box = self._yolo_to_abs(box_norm, img_w, img_h)
                label = labels[j]
                if label != name:
                    continue
                proposals.append({"label": label, "class_id": class_id, "score": 1.0, "box": box})

        # Cap proposals per class (sort by -area, x1, y1)
        per_class_proposals = defaultdict(list)
        for p in proposals:
            per_class_proposals[p["label"]].append(p)
        proposals = []
        for label, pl in per_class_proposals.items():
            pl.sort(key=lambda p: (-self._area(p["box"]), p["box"][1], p["box"][0]))
            proposals.extend(pl[:cfg.max_proposals_per_class])

        # Stage 2: Refine on crops
        refined: List[Dict[str, Any]] = []
        for prop in proposals:
            th = cfg.stage2_thresholds.get(prop["label"], 0.5)
            detector.confidence_threshold = th
            margin_box = self._expand_with_margin(prop["box"], cfg.crop_margin, img_w, img_h)
            crop_img = image.crop(margin_box)
            crop_w, crop_h = crop_img.size
            crop_refined = self._run_refine_on_crop(
                detector, crop_img, [prop["label"]], [prop["class_id"]], cfg, prop, margin_box, img_w, img_h
            )
            refined.extend(crop_refined)

        detector.confidence_threshold = original_th

        # Merge per class
        per_class_refined: Dict[str, List[Tuple[List[float], float]]] = defaultdict(list)
        for r in refined:
            per_class_refined[r["label"]].append((r["box"], r["score"]))

        merged: List[Dict[str, Any]] = []
        for label, dets in per_class_refined.items():
            if cfg.use_wbf:
                fused = self._merge_per_class_wbf(dets, cfg.wbf_iou)
            else:
                fused = self._merge_per_class_nms(dets, cfg.nms_iou)
            for box, score in fused:
                merged.append({"label": label, "class_id": name_to_id[label], "score": score, "box": box})

        # Post-filters: min area
        min_area = cfg.min_area_frac * img_w * img_h
        merged = [m for m in merged if self._area(m["box"]) >= min_area]

        # Sort by -score (primary), -area (secondary), x1, y1
        merged.sort(key=lambda m: (-m["score"], -self._area(m["box"]), m["box"][1], m["box"][0]))

        # Convert to YOLO
        yolo_out = [self._abs_to_yolo(m["box"], img_w, img_h, m["class_id"]) for m in merged]

        return {"abs": merged, "yolo": yolo_out}

    def _run_refine_on_crop(
        self,
        detector: Any,
        crop_img: Image.Image,
        crop_class_names: List[str],
        crop_class_ids: List[int],
        cfg: TwoStageConfig,
        prop: Dict[str, Any],
        margin_box: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
    ) -> List[Dict[str, Any]]:
        """
        Run refinement on a crop, with optional TTA.

        Args:
            detector: Base detector instance.
            crop_img: Cropped PIL Image.
            crop_class_names: Class names for this crop (single).
            crop_class_ids: Corresponding class IDs.
            cfg: Configuration.
            prop: Original proposal dict.
            margin_box: Crop box (y1, x1, y2, x2).
            img_w: Original image width.
            img_h: Original image height.

        Returns:
            List of refined detections that confirm against the proposal.
        """
        crop_w, crop_h = crop_img.size
        crop_refined = []

        # Run on original crop
        ok, yolo, labels_str = detector.detector(crop_img, crop_class_names, crop_class_ids)
        if ok:
            labels = [l.strip() for l in labels_str.split(",") if l.strip()]
            if len(labels) == len(yolo):
                for j, yl in enumerate(yolo):
                    class_id = int(yl[0])
                    box_crop = self._yolo_to_abs(yl[1:], crop_w, crop_h)
                    label = labels[j]
                    global_box = [
                        box_crop[0] + margin_box[0],
                        box_crop[1] + margin_box[1],
                        box_crop[2] + margin_box[0],
                        box_crop[3] + margin_box[1],
                    ]
                    global_box = self._clip(global_box, img_w, img_h)
                    if self._area(global_box) > 0:
                        crop_refined.append({"label": label, "class_id": class_id, "score": 1.0, "box": global_box})

        # Optional TTA flip
        if cfg.do_tta_flip:
            flipped_crop = crop_img.transpose(Image.FLIP_LEFT_RIGHT)
            ok, yolo, labels_str = detector.detector(flipped_crop, crop_class_names, crop_class_ids)
            if ok:
                labels = [l.strip() for l in labels_str.split(",") if l.strip()]
                if len(labels) == len(yolo):
                    for j, yl in enumerate(yolo):
                        class_id = int(yl[0])
                        box_crop = self._yolo_to_abs(yl[1:], crop_w, crop_h)
                        flipped_box = self._flip_horiz_boxes([box_crop], crop_w)[0]
                        global_box = [
                            flipped_box[0] + margin_box[0],
                            flipped_box[1] + margin_box[1],
                            flipped_box[2] + margin_box[0],
                            flipped_box[3] + margin_box[1],
                        ]
                        global_box = self._clip(global_box, img_w, img_h)
                        if self._area(global_box) > 0:
                            crop_refined.append({"label": label, "class_id": class_id, "score": 1.0, "box": global_box})

        # Confirm against proposal
        confirmed = []
        for r in crop_refined:
            if r["label"] == prop["label"] and self._iou(r["box"], prop["box"]) >= cfg.confirm_iou:
                confirmed.append(r)
        return confirmed

    def _expand_with_margin(self, box: List[float], margin: float, W: int, H: int) -> Tuple[int, int, int, int]:
        """
        Expand box with margin, clamp to image boundaries.

        Args:
            box: [ymin, xmin, ymax, xmax]
            margin: Relative margin to add.
            W: Image width.
            H: Image height.

        Returns:
            (ymin, xmin, ymax, xmax) as ints, clamped.
        """
        y1, x1, y2, x2 = box
        h = y2 - y1
        w = x2 - x1
        exp_h = h * margin
        exp_w = w * margin
        new_y1 = max(0, y1 - exp_h)
        new_x1 = max(0, x1 - exp_w)
        new_y2 = min(H, y2 + exp_h)
        new_x2 = min(W, x2 + exp_w)
        return int(new_y1), int(new_x1), int(new_y2), int(new_x2)

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        """
        Compute IoU between two boxes [ymin, xmin, ymax, xmax].

        Args:
            boxA: First box.
            boxB: Second box.

        Returns:
            IoU value.
        """
        ay1, ax1, ay2, ax2 = boxA
        by1, bx1, by2, bx2 = boxB
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        areaA = (ax2 - ax1) * (ay2 - ay1)
        areaB = (bx2 - bx1) * (by2 - by1)
        return inter / (areaA + areaB - inter)

    def _yolo_to_abs(self, box: List[float], w: int, h: int) -> List[float]:
        """
        Convert YOLO [xc, yc, ww, hh] normalized to absolute [ymin, xmin, ymax, xmax].

        Args:
            box: YOLO box [xc, yc, ww, hh].
            w: Width.
            h: Height.

        Returns:
            Absolute box [ymin, xmin, ymax, xmax].
        """
        xc, yc, ww, hh = box
        x1 = (xc - ww / 2) * w
        y1 = (yc - hh / 2) * h
        x2 = (xc + ww / 2) * w
        y2 = (yc + hh / 2) * h
        return [y1, x1, y2, x2]

    def _abs_to_yolo(self, abs_box: List[float], w: int, h: int, class_id: int) -> List[float]:
        """
        Convert absolute [ymin, xmin, ymax, xmax] to YOLO [class_id, xc, yc, ww, hh] normalized.

        Args:
            abs_box: Absolute box [ymin, xmin, ymax, xmax].
            w: Width.
            h: Height.
            class_id: Class ID.

        Returns:
            YOLO box [class_id, xc, yc, ww, hh].
        """
        y1, x1, y2, x2 = abs_box
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        ww = (x2 - x1) / w
        hh = (y2 - y1) / h
        return [class_id, xc, yc, ww, hh]

    def _clip(self, box: List[float], w: float, h: float) -> List[float]:
        """
        Clip box [ymin, xmin, ymax, xmax] to [0, h] x [0, w].

        Args:
            box: Box to clip.
            w: Width.
            h: Height.

        Returns:
            Clipped box.
        """
        y1, x1, y2, x2 = box
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        return [y1, x1, y2, x2]

    def _area(self, box: List[float]) -> float:
        """
        Compute area of box [ymin, xmin, ymax, xmax].

        Args:
            box: Box.

        Returns:
            Area.
        """
        y1, x1, y2, x2 = box
        return max(0, (x2 - x1) * (y2 - y1))

    def _flip_horiz_boxes(self, boxes: List[List[float]], img_w: int) -> List[List[float]]:
        """
        Flip boxes horizontally for TTA mapping back.

        Args:
            boxes: List of boxes [ymin, xmin, ymax, xmax].
            img_w: Width to flip relative to.

        Returns:
            Flipped boxes.
        """
        flipped = []
        for y1, x1, y2, x2 in boxes:
            flipped.append([y1, img_w - x2, y2, img_w - x1])
        return flipped

    def _merge_per_class_wbf(self, dets: List[Tuple[List[float], float]], iou_thresh: float) -> List[Tuple[List[float], float]]:
        """
        Weighted Box Fusion for single class detections.
        
        Args:
            dets: List of (box, score) tuples where box is [ymin, xmin, ymax, xmax]
            iou_thresh: IoU threshold for fusion
            
        Returns:
            List of fused (box, score) tuples
        """
        if not dets:
            return []
            
        # Simple WBF implementation - group overlapping boxes and average them weighted by score
        fused = []
        used = [False] * len(dets)
        
        for i, (box_i, score_i) in enumerate(dets):
            if used[i]:
                continue
                
            # Find all boxes that overlap with box_i
            group_boxes = [box_i]
            group_scores = [score_i]
            used[i] = True
            
            for j, (box_j, score_j) in enumerate(dets):
                if used[j] or i == j:
                    continue
                if self._iou(box_i, box_j) >= iou_thresh:
                    group_boxes.append(box_j)
                    group_scores.append(score_j)
                    used[j] = True
            
            # Weighted average of boxes
            total_score = sum(group_scores)
            if total_score > 0:
                avg_box = [0, 0, 0, 0]
                for k, (box, score) in enumerate(zip(group_boxes, group_scores)):
                    weight = score / total_score
                    for dim in range(4):
                        avg_box[dim] += box[dim] * weight
                
                # Average score
                avg_score = total_score / len(group_scores)
                fused.append((avg_box, avg_score))
        
        return fused

    def _merge_per_class_nms(self, dets: List[Tuple[List[float], float]], iou_thresh: float) -> List[Tuple[List[float], float]]:
        """
        Non-Maximum Suppression for single class detections.
        
        Args:
            dets: List of (box, score) tuples where box is [ymin, xmin, ymax, xmax]  
            iou_thresh: IoU threshold for suppression
            
        Returns:
            List of kept (box, score) tuples
        """
        if not dets:
            return []
            
        # Sort by score descending
        sorted_dets = sorted(dets, key=lambda x: x[1], reverse=True)
        
        keep = []
        for i, (box_i, score_i) in enumerate(sorted_dets):
            should_keep = True
            for kept_box, _ in keep:
                if self._iou(box_i, kept_box) >= iou_thresh:
                    should_keep = False
                    break
            if should_keep:
                keep.append((box_i, score_i))
                
        return keep

class TwoStageDetector:
    """
    Detector that uses TwoStageRefiner with an underlying Gemini detector.
    """
    def __init__(
        self,
        model_name: str,
        object_to_detect: str,
        problem_statement: str = "",
        sample_images: List[Image.Image] = None,
        confidence_threshold: float = 0.8,
        image_size: int = 1024,
        upscale_image: bool = False
    ):
        """
        Initialize Two-Stage detector with a Gemini detector and default configuration.

        Args:
            model_name: Gemini model name (e.g., "gemini-2.5-flash-lite").
            object_to_detect: Primary object to detect.
            problem_statement: Optional problem context.
            sample_images: List of sample PIL Images for context.
            confidence_threshold: Confidence threshold for detections.
            image_size: Target image size for resizing.
            upscale_image: Whether to upscale the image.
        """
        self.refiner = TwoStageRefiner()
        self.config = TwoStageConfig()  # Use default values
        self.gemini_detector = GeminiDetection(
            model_name=model_name,
            object_to_detect=object_to_detect,
            problem_statement=problem_statement,
            sample_images=sample_images or [],
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            upscale_image=upscale_image
        )
        self.model_name = f"twostage:{model_name}"  # For logging consistency

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        """
        Run two-stage refinement detection using TwoStageRefiner and GeminiDetection.

        Args:
            image: Input PIL Image.
            class_names: List of class names.
            class_ids: Corresponding class IDs.

        Returns:
            Tuple[bool, List[List[float]], str]:
                - bool: True if detections found, False otherwise.
                - List[[class_id, xc, yc, w, h]]: YOLO format detections normalized to original image.
                - str: Comma-separated labels corresponding to detections.
        """
        try:
            result = self.refiner.run(self.gemini_detector, image, class_names, class_ids, self.config)
            yolo_annotations = result['yolo']
            labels = [det['label'] for det in result['abs']]
            return bool(yolo_annotations), yolo_annotations, ", ".join(labels)
        except Exception as e:
            print(f"❌ Two-stage refiner error: {e}")
            return False, [], f"Two-stage error: {e}"

if __name__ == "__main__":
    # Minimal unit tests
    class MockGeminiDetector:
        def __init__(self):
            self.confidence_threshold = 0.8

        def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
            w, h = image.size
            yolo = []
            labels = []
            for i, name in enumerate(class_names):
                if self.confidence_threshold > 0.6:  # Simulate filtering
                    continue
                yolo.append([class_ids[i], 0.5, 0.5, 0.3, 0.3])
                labels.append(name)
                yolo.append([class_ids[i], 0.6, 0.6, 0.4, 0.4])
                labels.append(name)
            return True, yolo, ", ".join(labels)

    # Test 1: Basic two-stage refinement
    image = Image.new("RGB", (1920, 1080))
    detector = TwoStageDetector(
        model_name="mock",
        object_to_detect="Safety Helmet",
        confidence_threshold=0.5
    )
    detector.gemini_detector = MockGeminiDetector()  # Override for testing
    ok, yolo, labels = detector.detector(image, ["Safety Helmet", "High-Visibility Jacket"], [0, 1])
    print(f"Test 1 - Success: {ok}, Detections: {len(yolo)}, Labels: {labels}")

    # Test 2: With TTA
    detector.config.do_tta_flip = True
    ok, yolo, labels = detector.detector(image, ["Safety Helmet", "High-Visibility Jacket"], [0, 1])
    print(f"Test 2 (TTA) - Success: {ok}, Detections: {len(yolo)}, Labels: {labels}")

    # Test 3: NMS fallback
    detector.config.use_wbf = False
    ok, yolo, labels = detector.detector(image, ["Safety Helmet", "High-Visibility Jacket"], [0, 1])
    print(f"Test 3 (NMS) - Success: {ok}, Detections: {len(yolo)}, Labels: {labels}")