# gemini_detector_refined.py
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import os
import json
import time
import logging
import re
import ast
import requests
import math
import io

import numpy as np

# Optional deps (guarded)
try:
    import torch  # for EfficientSAM (optional)
except Exception:
    torch = None

try:
    import cv2  # for edge tighten & mask ops (optional but recommended)
except Exception:
    cv2 = None

# ---- Your existing Google Gemini client imports ----
from google import genai
from google.genai import types

# Optional: your enhancer (kept as-is)
try:
    from ..enhance import edsr_enhance
except Exception:
    edsr_enhance = None

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ===============================
# Optional SAM-lite refiner
# ===============================
class SAMRefiner:
    """
    Thin wrapper around EfficientSAM.
    Uses box prompts (x1,y1,x2,y2) to generate refined masks.
    """

    def __init__(self, device: Optional[str] = None, variant: str = "vits"):
        if torch is None:
            raise RuntimeError("PyTorch not available for SAMRefiner.")
        try:
            from efficient_sam.build_efficient_sam import (
                build_efficient_sam_vitt,
                build_efficient_sam_vits,
            )
        except Exception as e:
            raise RuntimeError(
                "EfficientSAM not installed. "
                "pip install git+https://github.com/yformer/EfficientSAM.git"
            ) from e

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if variant == "vitt":
            self.model = build_efficient_sam_vitt()
        else:
            self.model = build_efficient_sam_vits()
        self.model.to(self.device).eval()

    @staticmethod
    def _pil_to_tensor(img: Image.Image):
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        return t

    @torch.no_grad()
    def refine_with_boxes(
        self, image_pil: Image.Image, boxes_xyxy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            image_pil: PIL RGB image
            boxes_xyxy: (N,4) float32 absolute pixel coords [x1,y1,x2,y2]
        Returns:
            masks: (N,H,W) bool
            refined_boxes: (N,4) float32 xyxy
            scores: (N,) float32 (SAM IoU-like)
        """
        if boxes_xyxy.size == 0:
            H, W = image_pil.height, image_pil.width
            return (
                np.zeros((0, H, W), dtype=bool),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        img_t = self._pil_to_tensor(image_pil).to(self.device)

        all_masks, all_boxes, all_scores = [], [], []

        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(float, box)
            # Build box prompt
            pts = [[x1, y1], [x2, y2]]
            lbs = [2, 3]
            pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device).view(1, 1, -1, 2)
            lbs_t = torch.tensor(lbs, dtype=torch.int64, device=self.device).view(1, 1, -1)

            pred_logits, pred_iou = self.model(img_t, pts_t, lbs_t)

            # Pick best mask by IoU
            idx = torch.argmax(pred_iou.flatten())
            mask_logits = pred_logits[0, 0, idx]
            mask_prob = torch.sigmoid(mask_logits)
            mask = (mask_prob >= 0.5).cpu().numpy().astype(bool)

            all_masks.append(mask)
            all_scores.append(float(pred_iou.view(-1)[idx].item()))

            ys, xs = np.where(mask)
            if len(xs) == 0:
                all_boxes.append([0, 0, 0, 0])
            else:
                nx1, nx2, ny1, ny2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                all_boxes.append([nx1, ny1, nx2, ny2])

        return (
            np.stack(all_masks, axis=0),
            np.array(all_boxes, dtype=np.float32),
            np.array(all_scores, dtype=np.float32),
        )


# ===============================
# Utility helpers
# ===============================
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Box IoU for [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua + ub - inter + 1e-6
    return inter / union


def mask_iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boolean masks, handling different shapes by resizing to common size."""
    if a.shape == b.shape:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum() + 1e-6
        return float(inter / union)
    
    # If shapes don't match, resize both to a common size for comparison
    if cv2 is None:
        # Fallback: if opencv not available, return 0.0 for different shapes
        return 0.0
    
    # Use the smaller dimension to avoid upscaling
    target_h = min(a.shape[0], b.shape[0])
    target_w = min(a.shape[1], b.shape[1])
    target_size = (target_w, target_h)
    
    # Resize masks to common size
    a_resized = cv2.resize(a.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST) > 0
    b_resized = cv2.resize(b.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST) > 0
    
    inter = np.logical_and(a_resized, b_resized).sum()
    union = np.logical_or(a_resized, b_resized).sum() + 1e-6
    return float(inter / union)


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def tighten_by_edges(img: Image.Image, xyxy: List[float]) -> List[float]:
    if cv2 is None:
        return xyxy
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    crop = np.array(img.convert("RGB"))[y1:y2, x1:x2, :]
    if crop.size == 0:
        return xyxy
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return xyxy
    nx1, nx2, ny1, ny2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    new_box = [x1 + nx1, y1 + ny1, x1 + nx2, y1 + ny2]
    old_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    new_area = max(1.0, float((new_box[2] - new_box[0]) * (new_box[3] - new_box[1])))
    # Avoid over-shrinking (keep if at least 30% of old area)
    return new_box if new_area / old_area >= 0.3 else xyxy


def crop_to_pil(img: Image.Image, box: List[float]) -> Image.Image:
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    return img.crop((x1, y1, x2, y2))


def mask_from_box(box: List[float], H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    m = np.zeros((H, W), dtype=bool)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def normalize_mask_to_original(mask: np.ndarray, variant_dims: Tuple[int, int], original_dims: Tuple[int, int], scale: float) -> np.ndarray:
    """
    Normalize a mask from variant coordinate space to original image coordinate space.
    
    Args:
        mask: Boolean mask in variant coordinate space
        variant_dims: (H, W) of the variant image
        original_dims: (H, W) of the original image  
        scale: Scale factor used to create the variant
        
    Returns:
        Boolean mask normalized to original image dimensions
    """
    if cv2 is None:
        # Fallback: create a box-based mask in original space
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.zeros(original_dims, dtype=bool)
        # Scale coordinates back to original space
        x1, x2 = int(xs.min() / scale), int(xs.max() / scale)
        y1, y2 = int(ys.min() / scale), int(ys.max() / scale)
        return mask_from_box([x1, y1, x2+1, y2+1], original_dims[0], original_dims[1])
    
    # Resize mask back to original dimensions
    original_h, original_w = original_dims
    mask_resized = cv2.resize(
        mask.astype(np.uint8), 
        (original_w, original_h), 
        interpolation=cv2.INTER_NEAREST
    ) > 0
    return mask_resized


def generate_tiles(W: int, H: int, tile: int = 1024, overlap: float = 0.2) -> List[List[int]]:
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    # Ensure coverage of right/bottom edges
    if len(xs) == 0 or xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if len(ys) == 0 or ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append([x, y, min(x + tile, W), min(y + tile, H)])
    return tiles


def flip_boxes_horiz(boxes: List[List[float]], W: int) -> List[List[float]]:
    out = []
    for x1, y1, x2, y2 in boxes:
        nx1 = W - x2
        nx2 = W - x1
        out.append([nx1, y1, nx2, y2])
    return out


# ===============================
# Main detector (drop-in, API compatible)
# ===============================
class SAMGeminiDetection:
    """
    Enhanced Gemini-based detector with:
    - tiling + multi-scale proposals
    - SAM-lite refinement / fallback edge tightening
    - mask quality filters, mask-aware NMS
    - optional semantic verification on masked crops
    - score fusion

    Public API unchanged:
      detector(image, class_names, class_ids) -> (bool, yolo_list, label_summary)
    """
    class Cell:
        def __init__(self, id: int, left: int, top: int, right: int, bottom: int):
            self.id = id
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom

    def __init__(
        self,
        model_name: str,
        object_to_detect: str,
        problem_statement: str = "",
        sample_images: Optional[List[Image.Image]] = None,
        confidence_threshold: float = 0.8,
        multiple_predictions: bool = True,
        image_size: int = 1024,
        upscale_image: bool = False,
        golden_examples: Optional[List[Tuple[Image.Image, List[List[float]]]]] = None,
        # NEW knobs (all optional)
        enable_tiling: bool = True,
        tile_size: int = 1024,
        tile_overlap: float = 0.2,
        scales: Optional[List[float]] = None,       # e.g. [1.0, 1.5]
        enable_tta_flip: bool = True,
        use_sam: bool = True,
        use_edge_tighten_fallback: bool = True,
        min_mask_area_ratio: float = 0.001,         # relative to image area
        min_box_mask_iou: float = 0.30,
        mask_nms_iou: float = 0.50,
        enable_verification: bool = True,           # tiny recheck on masked crops
        verify_threshold: float = 0.55,
        score_fusion_weights: Dict[str, float] = None,  # g_conf, s_conf, v_conf, consistency
    ):
        self.model_name = model_name
        self.object_to_detect = object_to_detect
        self.problem_statement = problem_statement
        self.sample_images = sample_images or []
        self.confidence_threshold = confidence_threshold
        self.multiple_predictions = multiple_predictions
        self.image_size = image_size
        self.upscale_image = upscale_image
        self.golden_examples = golden_examples or []
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # pricing (kept shorter but structure preserved)
        self.pricing = {
            'gemini-2.5-pro': {
                'input': {'<=200k': 1.25, '>200k': 2.50},
                'output': {'<=200k': 10.00, '>200k': 15.00}
            },
            'gemini-2.5-flash': {'input': 0.30, 'output': 2.50},
            'gemini-2.5-flash-lite': {'input': 0.10, 'output': 0.40}
        }

        # new pipeline knobs
        self.enable_tiling = enable_tiling
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.scales = scales if scales is not None else [1.0, 1.5]
        self.enable_tta_flip = enable_tta_flip

        self.use_sam = use_sam
        self._sam_refiner = None  # lazy
        self.use_edge_tighten_fallback = use_edge_tighten_fallback

        self.min_mask_area_ratio = min_mask_area_ratio
        self.min_box_mask_iou = min_box_mask_iou
        self.mask_nms_iou = mask_nms_iou

        self.enable_verification = enable_verification
        self.verify_threshold = verify_threshold

        # fusion weights
        self.score_w = score_fusion_weights or {
            "g_conf": 0.45, "s_conf": 0.30, "v_conf": 0.15, "consistency": 0.10
        }

    # ---------- Prompts & parsing (mostly your original code, lightly adapted) ----------
    @staticmethod
    def get_system_prompt(normalization_factor: int) -> str:
        return ""

    @staticmethod
    def get_user_prompt(object_of_interest: str, normalization_factor: int) -> str:
        return (
            f"Detect all of the prominent items in the image that corresponds to "
            f"{object_of_interest}. The box_2d should be [ymin, xmin, ymax, xmax] "
            f"normalized to 0-{normalization_factor}."
        )

    @staticmethod
    def normalize_label_for_matching(label):
        if not label:
            return ""
        normalized = label.lower().strip()
        normalized = re.sub(r'[_\-\.\,\;\:\!\?\(\)\[\]{}]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        stop_words = ['the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with']
        words = normalized.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words).strip()

    @staticmethod
    def _fix_duplicate_fields(text):
        pattern = r'("box_2d":\s*\[[^\]]+\]),([^}]*),\s*("box_2d":\s*\[[^\]]+\])'
        def replace_duplicate(match):
            first_box = match.group(1)
            middle_content = match.group(2)
            return f'{first_box},{middle_content}'
        return re.sub(pattern, replace_duplicate, text)

    @staticmethod
    def _convert_python_to_json(text):
        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)
        text = re.sub(r"\bNone\b", "null", text)
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        return text

    @staticmethod
    def _try_parse_json_or_python(text):
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            obj = ast.literal_eval(text)
            return json.loads(json.dumps(obj))
        except (ValueError, SyntaxError):
            pass
        converted = SAMGeminiDetection._convert_python_to_json(text)
        if converted != text:
            try:
                return json.loads(converted)
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    @staticmethod
    def _find_matching_bracket(text, start_idx):
        if start_idx >= len(text):
            return None
        open_ch = text[start_idx]
        close_ch = "}" if open_ch == "{" else ("]" if open_ch == "[" else None)
        if close_ch is None:
            return None
        depth = 0
        in_string = False
        string_char = None
        escape_next = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch in "\"'":
                if not in_string:
                    in_string = True
                    string_char = ch
                elif ch == string_char:
                    in_string = False
                    string_char = None
            if not in_string:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return i
        return None

    @staticmethod
    def _extract_all_structures(text):
        candidates = []
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            if text[i] in "{[":
                start_pos = i
                end_pos = SAMGeminiDetection._find_matching_bracket(text, i)
                if end_pos is not None:
                    candidates.append({"text": text[start_pos:end_pos + 1], "start": start_pos, "end": end_pos})
                    i = end_pos + 1
                else:
                    i += 1
            else:
                i += 1
        return candidates

    @staticmethod
    def _validate_and_fix_detections(detections):
        if not isinstance(detections, list):
            return detections
        fixed_detections = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            if "box_2d" not in detection:
                continue
            fixed_detection = {
                "box_2d": detection["box_2d"],
                "label": detection.get("label", "object"),
                "confidence": detection.get("confidence", 0.5)
            }
            box_2d = fixed_detection["box_2d"]
            if isinstance(box_2d, list) and len(box_2d) == 4:
                try:
                    fixed_detection["box_2d"] = [float(coord) for coord in box_2d]
                    fixed_detections.append(fixed_detection)
                except (ValueError, TypeError):
                    continue
        return fixed_detections

    @staticmethod
    def parse_detection_output(output_text):
        if not output_text:
            return None
        cleaned = output_text.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1]
        cleaned = SAMGeminiDetection._fix_duplicate_fields(cleaned)
        code_fence_match = re.search(r"```(?:json|python)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
        if code_fence_match:
            fence_content = code_fence_match.group(1).strip()
            result = SAMGeminiDetection._try_parse_json_or_python(fence_content)
            if result is not None:
                return SAMGeminiDetection._validate_and_fix_detections(result)
        cleaned = re.sub(r"^```.*?\n|```$", "", cleaned, flags=re.S)
        cleaned = re.sub(r"//.*", "", cleaned)
        cleaned = re.sub(r"\(\s*(\d+)\s*\)", r"[\1]", cleaned)
        candidates = SAMGeminiDetection._extract_all_structures(cleaned)
        dicts = [c for c in candidates if c["text"].strip().startswith("{")]
        lists = [c for c in candidates if c["text"].strip().startswith("[")]
        for candidate in dicts:
            result = SAMGeminiDetection._try_parse_json_or_python(candidate["text"])
            if result is not None:
                return SAMGeminiDetection._validate_and_fix_detections(result)
        for candidate in lists:
            result = SAMGeminiDetection._try_parse_json_or_python(candidate["text"])
            if result is not None:
                return SAMGeminiDetection._validate_and_fix_detections(result)
        return None

    @staticmethod
    def _bbox_to_yolo(x0, y0, x1, y1, W, H, class_id, conf=1.0):
        xc = (x0 + x1) / 2 / W
        yc = (y0 + y1) / 2 / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H
        return [class_id, xc, yc, w, h, conf]

    # ---------- Cost tracking (kept compatible) ----------
    def get_usd_to_inr_exchange_rate(self):
        try:
            response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=10)
            data = response.json()
            return data['rates']['INR']
        except Exception as e:
            logger.warning(f"Failed to fetch exchange rate: {e}. Using fallback rate of 83.0")
            return 83.0

    def calculate_gemini_cost(self, input_tokens, output_tokens):
        try:
            model_key = None
            for key in self.pricing.keys():
                if key in self.model_name.lower():
                    model_key = key
                    break
            if not model_key:
                model_key = 'gemini-2.5-flash'
                logger.warning(f"Unknown model {self.model_name}, using default flash pricing")

            pricing_info = self.pricing[model_key]
            if isinstance(pricing_info['input'], dict):
                input_tier = '<=200k' if input_tokens <= 200000 else '>200k'
                output_tier = '<=200k' if output_tokens <= 200000 else '>200k'
                input_cost_usd = (input_tokens / 1_000_000) * pricing_info['input'][input_tier]
                output_cost_usd = (output_tokens / 1_000_000) * pricing_info['output'][output_tier]
            else:
                input_cost_usd = (input_tokens / 1_000_000) * pricing_info['input']
                output_cost_usd = (output_tokens / 1_000_000) * pricing_info['output']
            total_cost_usd = input_cost_usd + output_cost_usd
            exchange_rate = self.get_usd_to_inr_exchange_rate()
            total_cost_inr = total_cost_usd * exchange_rate
            logger.info(f"Gemini API cost: {total_cost_usd:.6f} USD = {total_cost_inr:.4f} INR (tokens: {input_tokens}+{output_tokens})")
            return total_cost_inr
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0

    def log_cost_to_file(self, cost_inr, dataset_name="unknown"):
        try:
            raw_data_path = "./data/raw_data"
            os.makedirs(raw_data_path, exist_ok=True)
            cost_file_path = os.path.join(raw_data_path, "cost.txt")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(cost_file_path, 'a', encoding='utf-8') as file:
                file.write(f"{timestamp} | {dataset_name} | Gemini API ({self.model_name}) | ₹{cost_inr:.4f}\n")
            logger.info(f"Cost logged to {cost_file_path}: ₹{cost_inr:.4f}")
        except Exception as e:
            logger.error(f"Error logging cost to file: {e}")

    # ---------- Internal: one Gemini pass on a PIL image -> list of raw detections ----------
    def _gemini_pass(self, im: Image.Image, queries: List[str], normalization_factor: int = 1000) -> List[Dict[str, Any]]:
        # problem statement, samples, golden examples
        context = []
        if self.problem_statement:
            context.append(f"Context: {self.problem_statement}")
        context.extend(self.sample_images)
        for golden_img, golden_anns in self.golden_examples:
            context.append(golden_img)
            class_names = queries
            golden_str = ", ".join([
                f"{class_names[int(ann[0])]} at [{ann[1]:.2f},{ann[2]:.2f},{ann[3]:.2f},{ann[4]:.2f}] (conf {ann[5]:.2f})"
                for ann in golden_anns
            ])
            context.append(f"Example: {golden_str}")

        base_prompt = self.get_user_prompt(", ".join(queries), normalization_factor)
        prompt = f"""
{base_prompt}

Think step by step: 
1. Analyze the overall scene in the image.
2. Identify and locate all instances of the specified objects.
3. For each detection, provide a confidence score and normalized bounding box.
4. Output ONLY a JSON array of detections, each with:
   {{
     "label": "<one of: {', '.join(queries)}>",
     "confidence": <float 0..1>,
     "box_2d": [y_min, x_min, y_max, x_max]  # normalized 0..{normalization_factor}
   }}
Rules:
- Ensure y_min < y_max and x_min < x_max.
- Include detections with confidence >= {self.confidence_threshold}.
- Order by confidence descending.
- Return [] if no detections meet criteria.
"""
        contents = [im, *context, prompt]
        thinking_budget = -1 if "pro" in self.model_name.lower() else 0

        # call with retries
        resp = None
        for attempt in range(3):
            try:
                start = time.time()
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        top_p=0.05,
                        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                        response_mime_type="text/plain"
                    ),
                )
                logger.info("Gemini call %.2fs (attempt %d)", time.time() - start, attempt + 1)

                # cost tracking
                try:
                    if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
                        input_tokens = getattr(resp.usage_metadata, 'prompt_token_count', 0)
                        output_tokens = getattr(resp.usage_metadata, 'candidates_token_count', 0)
                        if input_tokens > 0 or output_tokens > 0:
                            cost_inr = self.calculate_gemini_cost(input_tokens, output_tokens)
                            dataset_name = "annotation_job"
                            try:
                                cwd = os.getcwd()
                                if "raw_data" in cwd:
                                    dataset_name = os.path.basename(cwd)
                                elif self.object_to_detect:
                                    dataset_name = f"{self.object_to_detect}_detection"
                            except Exception:
                                pass
                            self.log_cost_to_file(cost_inr, dataset_name)
                except Exception as cost_error:
                    logger.error(f"Error in cost tracking: {cost_error}")

                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(0.5)

        text = getattr(resp, "text", "") if resp else ""
        raw = self.parse_detection_output(text)
        if isinstance(raw, dict) and "box_2d" in raw:
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return raw

    # ---------- Optional tiny verification using Gemini on masked crop ----------
    def _verify_crop_gemini(self, crop: Image.Image, label: str, queries: List[str]) -> float:
        """
        Returns v_conf in [0,1] = confidence it's the target label.
        This is intentionally tiny: single image + one-line instruction, temperature 0.0.
        """
        try:
            prompt = (
                f"Answer strictly in JSON: {{\"is_target\": <bool>, \"confidence\": <0..1>}}.\n"
                f"Question: Is the main object in this image an instance of '{label}' "
                f"(among {queries})?"
            )
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[crop, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0, top_p=0.01, response_mime_type="text/plain"
                ),
            )
            txt = getattr(resp, "text", "") or ""
            parsed = self._try_parse_json_or_python(txt)
            if isinstance(parsed, dict):
                conf = float(parsed.get("confidence", 0.0))
                if parsed.get("is_target", False):
                    return max(0.0, min(1.0, conf))
                else:
                    return 1.0 - max(0.0, min(1.0, conf))
        except Exception:
            pass
        return 0.5  # neutral fallback

    # ---------- Public API (unchanged signature) ----------
    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        try:
            queries = class_names if class_names else [self.object_to_detect]
            name_to_id = {n: cid for n, cid in zip(class_names, class_ids)}
            original = image.copy()
            if self.upscale_image and edsr_enhance is not None:
                try:
                    original = edsr_enhance(original)
                except Exception as e:
                    logger.warning(f"Upscale failed, using original: {e}")

            W0, H0 = original.width, original.height

            # Prepare SAM if enabled
            if self.use_sam and self._sam_refiner is None:
                try:
                    self._sam_refiner = SAMRefiner()
                except Exception as e:
                    logger.warning(f"SAM unavailable, fallback to edge tighten: {e}")
                    self._sam_refiner = None

            # Build inference variants (scales x flips)
            variants = []
            for s in (self.scales or [1.0]):
                scaled_w, scaled_h = int(W0 * s), int(H0 * s)
                img_s = original.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS) if s != 1.0 else original
                variants.append({"img": img_s, "scale": s, "flip": False})
                if self.enable_tta_flip:
                    variants.append({"img": img_s.transpose(Image.Transpose.FLIP_LEFT_RIGHT), "scale": s, "flip": True})

            all_candidate_objs = []  # gather detections across variants for consensus

            # ---- Run variants ----
            for var_idx, var in enumerate(variants):
                img_v: Image.Image = var["img"]
                s = var["scale"]
                flipped = var["flip"]
                W, H = img_v.width, img_v.height

                tiles = [[0, 0, W, H]]
                if self.enable_tiling:
                    tiles = generate_tiles(W, H, tile=self.tile_size, overlap=self.tile_overlap)

                variant_objs = []

                for t_idx, (x1, y1, x2, y2) in enumerate(tiles):
                    tile = img_v.crop((x1, y1, x2, y2))
                    # Downscale tile for token cost (preserve aspect)
                    tile_small = tile.copy()
                    tile_small.thumbnail([self.image_size, self.image_size], Image.Resampling.LANCZOS)

                    raw = self._gemini_pass(tile_small, queries, normalization_factor=1000)
                    if not raw:
                        continue

                    # Map normalized 0..1000 -> tile abs coords in variant space
                    det_boxes = []
                    det_labels = []
                    det_gconfs = []
                    for it in raw:
                        label = it.get("label", "")
                        conf = float(it.get("confidence", 0.0) or 0.0)
                        if conf < self.confidence_threshold:
                            continue

                        # label match (exact/normalized/partial)
                        matched_label = None
                        if label in name_to_id:
                            matched_label = label
                        else:
                            label_norm = label.lower().strip()
                            for cname in name_to_id.keys():
                                if label_norm == cname.lower().strip():
                                    matched_label = cname
                                    break
                            if not matched_label:
                                ldeep = self.normalize_label_for_matching(label)
                                for cname in name_to_id.keys():
                                    if ldeep == self.normalize_label_for_matching(cname):
                                        matched_label = cname
                                        break
                            if not matched_label:
                                for cname in name_to_id.keys():
                                    if (label_norm in cname.lower().strip()) or (cname.lower().strip() in label_norm):
                                        matched_label = cname
                                        break
                        if not matched_label:
                            continue

                        box = it.get("box_2d", None)
                        if not (isinstance(box, list) and len(box) == 4):
                            continue
                        # [ymin,xmin,ymax,xmax] normalized to 0..1000 in TILE space
                        ny1, nx1, ny2, nx2 = [float(v) for v in box]
                        # tile abs
                        ax1 = x1 + (nx1 / 1000.0) * (x2 - x1)
                        ax2 = x1 + (nx2 / 1000.0) * (x2 - x1)
                        ay1 = y1 + (ny1 / 1000.0) * (y2 - y1)
                        ay2 = y1 + (ny2 / 1000.0) * (y2 - y1)

                        # flip back if needed
                        if flipped:
                            fx1 = W - ax2
                            fx2 = W - ax1
                            ax1, ax2 = fx1, fx2

                        det_boxes.append([ax1, ay1, ax2, ay2])
                        det_labels.append(matched_label)
                        det_gconfs.append(conf)

                    if not det_boxes:
                        continue

                    det_boxes_np = np.array(det_boxes, dtype=np.float32)

                    # --- Refinement: SAM or edge tighten ---
                    masks = []
                    s_scores = []
                    refined_boxes = []
                    if self._sam_refiner is not None:
                        try:
                            mks, rboxes, ssc = self._sam_refiner.refine_with_boxes(img_v if not flipped else img_v.transpose(Image.Transpose.FLIP_LEFT_RIGHT), det_boxes_np if not flipped else np.array(flip_boxes_horiz(det_boxes, W), dtype=np.float32))
                            # flip masks/boxes back if variant was flipped
                            if flipped:
                                # flip masks back horizontally
                                mks = np.flip(mks, axis=2)  # (N,H,W)
                                # boxes already flipped above when sending; no need to flip rboxes again (we passed flipped boxes)
                            masks = [m for m in mks]
                            refined_boxes = rboxes.tolist()
                            s_scores = ssc.tolist()
                        except Exception as e:
                            logger.warning(f"SAM refine failed, falling back: {e}")
                            masks = []
                            s_scores = []
                            refined_boxes = []

                    if (self._sam_refiner is None) and self.use_edge_tighten_fallback:
                        for b in det_boxes:
                            tb = tighten_by_edges(img_v, b)
                            refined_boxes.append(tb)
                            # synthetic mask as box (very weak but better than nothing)
                            masks.append(mask_from_box(tb, H, W))
                            s_scores.append(0.5)

                    if not refined_boxes:
                        # No refinement at all — keep originals
                        refined_boxes = det_boxes
                        masks = [mask_from_box(b, H, W) for b in det_boxes]
                        s_scores = [0.5] * len(det_boxes)

                    # --- Sanity filters + (optional) expansion check for partials ---
                    area_min = self.min_mask_area_ratio * (W * H)
                    filtered = []
                    for i, (rb, m, gconf, lab, sconf) in enumerate(zip(refined_boxes, masks, det_gconfs, det_labels, s_scores)):
                        m_area = float(m.sum())
                        if m_area < area_min:
                            continue
                        # box-mask IoU
                        bmi = mask_iou_bool(mask_from_box(rb, H, W), m)
                        if bmi < self.min_box_mask_iou:
                            continue
                        # (Optional) part->whole heuristic: expand 12% and re-tighten via edges (cheap)
                        exp = [
                            max(0.0, rb[0] - 0.12 * (rb[2]-rb[0])),
                            max(0.0, rb[1] - 0.12 * (rb[3]-rb[1])),
                            min(W * 1.0, rb[2] + 0.12 * (rb[2]-rb[0])),
                            min(H * 1.0, rb[3] + 0.12 * (rb[3]-rb[1])),
                        ]
                        if self.use_edge_tighten_fallback:
                            exp_tight = tighten_by_edges(img_v, exp)
                            m_exp = mask_from_box(exp_tight, H, W)
                            if m_exp.sum() > m_area * 1.25:  # object likely larger
                                rb = exp_tight
                                m = m_exp
                                sconf = max(sconf, 0.6)

                        # Optional semantic verification (masked crop only)
                        vconf = 0.5
                        if self.enable_verification and self._crop_safe(m):
                            crop = self._masked_crop(img_v, m)
                            vconf = self._verify_crop_gemini(crop, lab, queries)

                        # Normalize mask to original image dimensions for consistent comparison
                        m_normalized = normalize_mask_to_original(m, (H, W), (H0, W0), s)
                        
                        # Also normalize box coordinates to original image space
                        rb_normalized = [
                            rb[0] / s,  # x1
                            rb[1] / s,  # y1  
                            rb[2] / s,  # x2
                            rb[3] / s   # y2
                        ]
                        
                        filtered.append({
                            "label": lab,
                            "box": rb_normalized,
                            "mask": m_normalized,
                            "g_conf": float(gconf),
                            "s_conf": float(sconf),
                            "v_conf": float(vconf),
                            "origin": {"var": var_idx, "tile": t_idx, "scale": s, "flip": flipped}
                        })

                    variant_objs.extend(filtered)

                # consensus later; store per variant
                all_candidate_objs.extend(variant_objs)

            if not all_candidate_objs:
                return False, [], ", ".join(queries)

            # ---- Self-consistency (consensus over variants) ----
            # Group by mask/box IoU>=0.5 across variants and count frequency
            groups = []
            used = [False] * len(all_candidate_objs)
            for i, a in enumerate(all_candidate_objs):
                if used[i]:
                    continue
                group = [i]
                used[i] = True
                for j in range(i + 1, len(all_candidate_objs)):
                    if used[j]:
                        continue
                    b = all_candidate_objs[j]
                    if a["label"] != b["label"]:
                        continue
                    # prefer mask IoU; fallback to box IoU
                    miou = mask_iou_bool(a["mask"], b["mask"])
                    if miou >= 0.5 or iou_xyxy(a["box"], b["box"]) >= 0.5:
                        used[j] = True
                        group.append(j)
                groups.append(group)

            survivors = []
            for g in groups:
                objs = [all_candidate_objs[k] for k in g]
                # pick the best by fused score; also compute consistency
                consistency = len(objs) / max(1.0, float(len(variants)))
                best = None
                best_score = -1.0
                for o in objs:
                    score = (
                        self.score_w["g_conf"] * o["g_conf"] +
                        self.score_w["s_conf"] * o["s_conf"] +
                        self.score_w["v_conf"] * o["v_conf"] +
                        self.score_w["consistency"] * consistency
                    )
                    if score > best_score:
                        best_score = score
                        best = dict(o)
                        best["score"] = float(score)
                        best["consistency"] = float(consistency)
                if best is not None:
                    survivors.append(best)

            # ---- Mask-aware NMS on survivors ----
            survivors = sorted(survivors, key=lambda d: d["score"], reverse=True)
            kept = []
            suppressed = [False] * len(survivors)
            for i, a in enumerate(survivors):
                if suppressed[i]:
                    continue
                kept.append(a)
                for j in range(i + 1, len(survivors)):
                    if suppressed[j]:
                        continue
                    b = survivors[j]
                    if a["label"] != b["label"]:
                        continue
                    if mask_iou_bool(a["mask"], b["mask"]) >= self.mask_nms_iou:
                        suppressed[j] = True

            # final thresholding by fused score
            final_objs = [k for k in kept if k["score"] >= self.verify_threshold]

            # reduce to single best if multiple_predictions is False
            if not self.multiple_predictions and final_objs:
                final_objs = [final_objs[0]]

            # Convert to YOLO (boxes are already in original image coordinates)
            yolo = []
            labels_out = []
            for idx, det in enumerate(final_objs):
                lab = det["label"]
                cid = name_to_id.get(lab, 0)
                x0, y0, x1, y1 = det["box"]
                conf = float(det.get("g_conf", 0.5))
                # blend SAM + verification into conf to report (still keep your original meaning)
                conf = 0.7 * conf + 0.3 * float(det.get("s_conf", 0.5))
                yolo.append(self._bbox_to_yolo(x0, y0, x1, y1, W0, H0, cid, conf))
                labels_out.append(lab)

            return (len(yolo) > 0), yolo, (", ".join(labels_out) if labels_out else ", ".join(queries))

        except Exception as e:
            logger.exception("Gemini detector (enhanced) error")
            return False, [], f"Error processing detection: {e}"

    # ---------- small helpers ----------
    @staticmethod
    def _crop_safe(mask: np.ndarray) -> bool:
        return mask is not None and isinstance(mask, np.ndarray) and mask.ndim == 2 and mask.sum() > 0

    @staticmethod
    def _masked_crop(img: Image.Image, mask: np.ndarray) -> Image.Image:
        """Return a tight masked crop pasted on white background to avoid background leakage."""
        box = bbox_from_mask(mask)
        if box is None:
            return img.copy()
        crop = crop_to_pil(img, box)
        # apply mask to crop (if cv2 available)
        if cv2 is None:
            return crop
        submask = mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        arr = np.array(crop.convert("RGB"))
        bg = np.ones_like(arr) * 255
        submask_u8 = (submask.astype(np.uint8) * 255)[:, :, None]
        out = np.where(submask_u8 == 255, arr, bg)
        return Image.fromarray(out.astype(np.uint8))
