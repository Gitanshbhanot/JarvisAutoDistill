from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
from annotate.detectors.gemini import GeminiDetection

@dataclass
class TilingConfig:
    tile_size: int = 1024
    overlap: float = 0.15  # 0–0.5
    wbf_iou: float = 0.60
    nms_iou: float = 0.55
    min_area_frac: float = 0.0015  # relative to full image area
    per_class_thresholds: Dict[str, float] = field(default_factory=lambda: {"Safety Helmet": 0.45, "High-Visibility Jacket": 0.35})
    use_wbf: bool = True
    do_tta_flip: bool = False  # horizontal flip TTA

class SAHITiler:
    """
    SAHI-style tiling and merging runner for object detection.
    """
    def make_tiles(self, image: Image.Image, cfg: TilingConfig) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Split the input image into overlapping tiles.

        Args:
            image: Input PIL Image.
            cfg: Tiling configuration.

        Returns:
            List of (tile_image, (x0, y0, x1, y1)) where coordinates are absolute in the original image.
        """
        w, h = image.size
        tile_size = cfg.tile_size
        overlap_px = int(tile_size * cfg.overlap)
        stride = tile_size - overlap_px

        tiles = []
        for y0 in range(0, h, stride):
            y1 = min(y0 + tile_size, h)
            if y1 - y0 < overlap_px and y0 > 0:
                y0 = max(0, h - tile_size)
                y1 = h
            for x0 in range(0, w, stride):
                x1 = min(x0 + tile_size, w)
                if x1 - x0 < overlap_px and x0 > 0:
                    x0 = max(0, w - tile_size)
                    x1 = w
                tile_img = image.crop((x0, y0, x1, y1))
                tiles.append((tile_img, (x0, y0, x1, y1)))
        return tiles

    def run_on_tiles(
        self,
        detector: Any,
        image: Image.Image,
        class_names: List[str],
        class_ids: List[int],
        cfg: TilingConfig,
    ) -> Dict[str, Any]:
        """
        Run detection on tiles, merge results per class, apply filters.

        Args:
            detector: Instance with .detector(image, class_names, class_ids) method.
            image: Input PIL Image.
            class_names: List of class names.
            class_ids: Corresponding class IDs.
            cfg: Tiling configuration.

        Returns:
            Dict with "abs": List[Dict[label, class_id, score, box:[ymin,xmin,ymax,xmax]] sorted by score desc,
                 "yolo": List[List[float]],    # [[class_id, xc, yc, w, h] normalized to original image]
        """
        from collections import defaultdict

        img_w, img_h = image.size
        full_area = img_w * img_h
        min_area = cfg.min_area_frac * full_area

        # Process original tiles
        detections = self._get_detections_from_tiles(detector, image, class_names, class_ids, cfg)

        # Optional TTA: horizontal flip
        if cfg.do_tta_flip:
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_dets = self._get_detections_from_tiles(detector, flipped_image, class_names, class_ids, cfg)
            back_flipped = []
            for det in flipped_dets:
                flipped_box = self._flip_horiz_boxes([det["box"]], img_w)[0]
                back_flipped.append(
                    {"label": det["label"], "class_id": det["class_id"], "score": det["score"], "box": flipped_box}
                )
            detections.extend(back_flipped)

        # Group by class/label
        per_class_dets: Dict[str, List[Tuple[List[float], float]]] = defaultdict(list)
        for det in detections:
            per_class_dets[det["label"]].append((det["box"], det["score"]))

        # Merge per class
        merged_per_class = {}
        for label, dets_list in per_class_dets.items():
            if cfg.use_wbf:
                merged = self._merge_per_class_wbf(dets_list, cfg.wbf_iou)
            else:
                merged = self._merge_per_class_nms(dets_list, cfg.nms_iou)
            merged_per_class[label] = merged

        # Collect final, apply filters
        final_abs: List[Dict[str, Any]] = []
        name_to_id = {name: cid for name, cid in zip(class_names, class_ids)}
        for label, merged in merged_per_class.items():
            thresh = cfg.per_class_thresholds.get(label, 0.5)
            for box, score in merged:
                if score < thresh:
                    continue
                area = self._area(box)
                if area < min_area:
                    continue
                class_id = name_to_id.get(label, 0)
                final_abs.append({"label": label, "class_id": class_id, "score": score, "box": box})

        # Sort by score descending
        final_abs.sort(key=lambda d: d["score"], reverse=True)

        # Convert to YOLO format
        yolo_out = []
        for fa in final_abs:
            yolo_box = self._abs_to_yolo(fa["box"], img_w, img_h)
            yolo_out.append([fa["class_id"]] + yolo_box)

        return {"abs": final_abs, "yolo": yolo_out}

    def _get_detections_from_tiles(
        self,
        detector: Any,
        image: Image.Image,
        class_names: List[str],
        class_ids: List[int],
        cfg: TilingConfig,
    ) -> List[Dict[str, Any]]:
        """
        Get absolute detections from tiles on the given image.
        """
        tiles = self.make_tiles(image, cfg)
        detections = []
        img_w, img_h = image.size
        for tile_img, (x0, y0, x1, y1) in tiles:
            tile_w, tile_h = x1 - x0, y1 - y0
            ok, yolo, labels_str = detector.detector(tile_img, class_names, class_ids)
            if not ok:
                continue
            labels = [l.strip() for l in labels_str.split(",")] if labels_str else []
            if len(labels) != len(yolo):
                continue
            for i, yl in enumerate(yolo):
                class_id = int(yl[0])
                box_norm = yl[1:]
                abs_tile = self._yolo_to_abs(box_norm, tile_w, tile_h)
                abs_tile = self._clip(abs_tile, tile_w, tile_h)
                if self._area(abs_tile) == 0:
                    continue
                global_box = [abs_tile[0] + y0, abs_tile[1] + x0, abs_tile[2] + y0, abs_tile[3] + x0]
                global_box = self._clip(global_box, img_w, img_h)
                if self._area(global_box) == 0:
                    continue
                label = labels[i]
                detections.append({"label": label, "class_id": class_id, "score": 1.0, "box": global_box})
        return detections

    def _yolo_to_abs(self, box: List[float], w: int, h: int) -> List[float]:
        """
        Convert YOLO [xc, yc, ww, hh] normalized to absolute [ymin, xmin, ymax, xmax].
        """
        xc, yc, ww, hh = box
        x1 = (xc - ww / 2) * w
        y1 = (yc - hh / 2) * h
        x2 = (xc + ww / 2) * w
        y2 = (yc + hh / 2) * h
        return [y1, x1, y2, x2]

    def _abs_to_yolo(self, abs_box: List[float], w: int, h: int) -> List[float]:
        """
        Convert absolute [ymin, xmin, ymax, xmax] to YOLO [xc, yc, ww, hh] normalized.
        """
        y1, x1, y2, x2 = abs_box
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        ww = (x2 - x1) / w
        hh = (y2 - y1) / h
        return [xc, yc, ww, hh]

    def _merge_per_class_wbf(
        self, dets: List[Tuple[List[float], float]], iou_thr: float
    ) -> List[Tuple[List[float], float]]:
        """
        Weighted Box Fusion (WBF) for a single class.
        """
        if not dets:
            return []
        dets = sorted(dets, key=lambda bs: (-bs[1], bs[0][1], bs[0][0]))
        fused = []
        while dets:
            best_box, best_score = dets.pop(0)
            cluster = [(best_box, best_score)]
            i = 0
            while i < len(dets):
                box, score = dets[i]
                if self._iou(best_box, box) >= iou_thr:
                    cluster.append((box, score))
                    del dets[i]
                else:
                    i += 1
            if len(cluster) > 1:
                weights = np.array([s for _, s in cluster])
                coords = np.array([b for b, _ in cluster]).T
                fused_coords = np.dot(weights, coords.T) / weights.sum()
                fused_box = fused_coords.tolist()
                fused_score = max(s for _, s in cluster)
            else:
                fused_box = best_box
                fused_score = best_score
            fused.append((fused_box, fused_score))
        return fused

    def _merge_per_class_nms(
        self, dets: List[Tuple[List[float], float]], iou_thr: float
    ) -> List[Tuple[List[float], float]]:
        """
        Non-Max Suppression (NMS) for a single class.
        """
        if not dets:
            return []
        dets = sorted(dets, key=lambda bs: (-bs[1], bs[0][1], bs[0][0]))
        kept = []
        while dets:
            best_box, best_score = dets.pop(0)
            kept.append((best_box, best_score))
            dets = [(box, score) for box, score in dets if self._iou(best_box, box) < iou_thr]
        return kept

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        """
        Compute IoU between two boxes [ymin, xmin, ymax, xmax].
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

    def _clip(self, box: List[float], w: float, h: float) -> List[float]:
        """
        Clip box [ymin, xmin, ymax, xmax] to [0, h] x [0, w].
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
        """
        y1, x1, y2, x2 = box
        return max(0, (x2 - x1) * (y2 - y1))

    def _flip_horiz_boxes(self, boxes: List[List[float]], img_w: int) -> List[List[float]]:
        """
        Flip boxes horizontally for TTA mapping back.
        """
        flipped = []
        for y1, x1, y2, x2 in boxes:
            flipped.append([y1, img_w - x2, y2, img_w - x1])
        return flipped

class SAHIDetector:
    """
    Detector that uses SAHI tiling with an underlying Gemini detector.
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
        Initialize SAHI detector with a Gemini detector and default tiling configuration.

        Args:
            model_name: Gemini model name (e.g., "gemini-2.5-flash-lite").
            object_to_detect: Primary object to detect.
            problem_statement: Optional problem context.
            sample_images: List of sample PIL Images for context.
            confidence_threshold: Confidence threshold for detections.
            image_size: Target image size for resizing.
            upscale_image: Whether to upscale the image.
        """
        self.tiler = SAHITiler()
        self.tiling_config = TilingConfig()  # Use default values
        self.gemini_detector = GeminiDetection(
            model_name=model_name,
            object_to_detect=object_to_detect,
            problem_statement=problem_statement,
            sample_images=sample_images or [],
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            upscale_image=upscale_image
        )

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        """
        Run tiled detection using SAHITiler and GeminiDetection.

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
            result = self.tiler.run_on_tiles(
                self.gemini_detector, image, class_names, class_ids, self.tiling_config
            )
            yolo_annotations = result['yolo']
            labels = [det['label'] for det in result['abs']]
            return bool(yolo_annotations), yolo_annotations, ", ".join(labels)
        except Exception as e:
            print(f"❌ SAHI tiling error: {e}")
            return False, [], f"SAHI error: {e}"