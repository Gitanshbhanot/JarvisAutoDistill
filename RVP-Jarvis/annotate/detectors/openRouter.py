from typing import List, Tuple
from PIL import Image
import os
import json
import base64
import requests
from dotenv import load_dotenv
import logging
from io import BytesIO
import time
import re
import ast
from ..detector import BaseDetector  # Assuming this is defined elsewhere
from ..enhance import edsr_enhance

load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterDetection(BaseDetector):
    class Cell:
        def __init__(self, id: int, left: int, top: int, right: int, bottom: int):
            self.id = id
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom

    def __init__(self, model_name: str, object_to_detect: str,
                 problem_statement: str = "", sample_images: List[Image.Image] = None,
                 confidence_threshold: float = 0.8, multiple_predictions: bool = True, image_size=1024, upscale_image=False):
        self.model_name = model_name
        self.object_to_detect = object_to_detect
        self.problem_statement = problem_statement
        self.sample_images = sample_images or []
        self.confidence_threshold = confidence_threshold
        self.multiple_predictions = multiple_predictions
        self.image_size = image_size
        self.upscale_image = upscale_image
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Validate API key
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please add it to your .env file.")
        
        print(f"🤖 Initialized OpenRouter detector with model: {self.model_name}")
        print(f"🔑 API key found: {'Yes' if self.api_key else 'No'}")

    @staticmethod
    def get_system_prompt(normalization_factor: int) -> str:
        """Get the system prompt for OpenRouter object detection."""
        return "You are a vision model that returns STRICT JSON only. Do not include any text outside JSON."

    @staticmethod
    def get_user_prompt(object_of_interest: str, normalization_factor: int) -> str:
        """Get the user prompt for OpenRouter object detection."""
        return f"""Detect all of the prominent items in the image that corresponds to {object_of_interest}. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-{normalization_factor}."""

    @staticmethod
    def parse_detection_output(output_text):
        """Forgiving parse of dicts/lists, handles outer quotes, comments & code-fences."""
        if not output_text:
            return None
        cleaned = output_text.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1]
        code_fence_match = re.search(
            r"```(?:json|python)?\s*\n(.*?)\n```", cleaned, re.DOTALL
        )
        if code_fence_match:
            fence_content = code_fence_match.group(1).strip()
            result = OpenRouterDetection._try_parse_json_or_python(fence_content)
            if result is not None:
                return result
        cleaned = re.sub(r"^```.*?\n|```$", "", cleaned, flags=re.S)
        cleaned = re.sub(r"//.*", "", cleaned)
        cleaned = re.sub(r"\(\s*(\d+)\s*\)", r"[\1]", cleaned)
        candidates = OpenRouterDetection._extract_all_structures(cleaned)
        dicts = [c for c in candidates if c["text"].strip().startswith("{")]
        lists = [c for c in candidates if c["text"].strip().startswith("[")]
        for candidate in dicts:
            result = OpenRouterDetection._try_parse_json_or_python(candidate["text"])
            if result is not None:
                return result
        for candidate in lists:
            result = OpenRouterDetection._try_parse_json_or_python(candidate["text"])
            if result is not None:
                return result
        return None

    @staticmethod
    def _try_parse_json_or_python(text):
        """Try to parse text as JSON or Python literal."""
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
        converted = OpenRouterDetection._convert_python_to_json(text)
        if converted != text:
            try:
                return json.loads(converted)
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    @staticmethod
    def _convert_python_to_json(text):
        """Convert common Python syntax to JSON syntax."""
        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)
        text = re.sub(r"\bNone\b", "null", text)
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        return text

    @staticmethod
    def _extract_all_structures(text):
        """Extract all dict/list structures with proper string and comment handling."""
        candidates = []
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            if text[i] in "{[":
                start_pos = i
                end_pos = OpenRouterDetection._find_matching_bracket(text, i)
                if end_pos is not None:
                    candidates.append(
                        {
                            "text": text[start_pos:end_pos + 1],
                            "start": start_pos,
                            "end": end_pos,
                        }
                    )
                    i = end_pos + 1
                else:
                    i += 1
            else:
                i += 1
        return candidates

    @staticmethod
    def _find_matching_bracket(text, start_idx):
        """Find matching bracket with proper string handling."""
        if start_idx >= len(text):
            return None
        open_ch = text[start_idx]
        if open_ch == "{":
            close_ch = "}"
        elif open_ch == "[":
            close_ch = "]"
        else:
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
    def _img_b64_resized(img: Image.Image, max_side=1024, fmt="JPEG") -> str:
        im = img.copy()
        im.thumbnail([max_side, max_side], Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format=fmt, quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), im.size

    @staticmethod
    def _norm_label(s: str) -> str:
        return (s or "").strip().lower()

    @staticmethod
    def _bbox_to_yolo(x0, y0, x1, y1, W, H, class_id):
        xc = (x0 + x1) / 2 / W
        yc = (y0 + y1) / 2 / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H
        return [class_id, xc, yc, w, h]

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        try:
            queries = class_names if class_names else [self.object_to_detect]
            name_to_id = {self._norm_label(n): cid for n, cid in zip(class_names, class_ids)}

            original_size = image.size
            if self.upscale_image:
                try:
                    image = edsr_enhance(image)
                except Exception as e:
                    print(f"Warning: Image enhancement failed, using original image: {e}")
                    print("Using original image...")
                    # Continue with the original image
            b64, (scaled_w, scaled_h) = self._img_b64_resized(image, max_side=self.image_size, fmt="JPEG")
            sample_b64s = []
            for si in self.sample_images[:3]:
                sb64, _ = self._img_b64_resized(si, max_side=self.image_size, fmt="JPEG")
                sample_b64s.append(sb64)

            # Prepare context parts
            context_parts = []
            if self.problem_statement:
                context_parts.append({"type": "text", "text": f"Context: {self.problem_statement}"})
            for sb64 in sample_b64s:
                context_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{sb64}"}})

            # Generate prompt with reasoning
            normalization_factor = 1000
            base_prompt = self.get_user_prompt(
                object_of_interest=", ".join(queries),
                normalization_factor=normalization_factor
            )
            prompt_text = f"""
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

            user_parts = [
                *context_parts,
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]

            system_msg = {
                "role": "system",
                "content": self.get_system_prompt(normalization_factor)
            }

            # Schema remains for enforcement
            schema = {
                "name": "detections_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "detections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "box_2d": {
                                        "anyOf": [
                                            {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "x_min":{"type":"number"}, "y_min":{"type":"number"},
                                                    "x_max":{"type":"number"}, "y_max":{"type":"number"}
                                                },
                                                "required": ["x_min","y_min","x_max","y_max"],
                                                "additionalProperties": False
                                            }
                                        ]
                                    }
                                },
                                "required": ["label","confidence","box_2d"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["detections"]
                }
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    system_msg,
                    {"role": "user", "content": user_parts}
                ],
                "response_format": {"type": "json_schema", "json_schema": schema},
                "temperature": 0.0,
                "top_p": 0.05,
                "max_tokens": 8192,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "RVP-Jarvis Annotator"),
            }

            print(f"🌐 Making OpenRouter API call to model: {self.model_name}")

            # Retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    start = time.time()
                    r = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
                    logger.info("OpenRouter call %.2fs (attempt %d)", time.time() - start, attempt + 1)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error("Failed after %d retries: %s", max_retries, e)
                        return False, [], f"API error after retries: {e}"
                    time.sleep(1)

            if r.status_code != 200:
                error_msg = f"OpenRouter API error {r.status_code}: {r.text}"
                print(f"❌ {error_msg}")
                logger.error(error_msg)
                return False, [], error_msg
                
            data = r.json()
            print(f"✅ OpenRouter API call successful", data)

            # Debug raw response
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print("----------------OpenRouter Detection LOGGING REASONING ----------------")
            print(content)
            print("----------------OpenRouter Detection LOGGING REASONING ----------------")

            # Parse response
            try:
                raw = self.parse_detection_output(content)
                if not isinstance(raw, list):
                    raw = raw.get("detections", []) if isinstance(raw, dict) else []
            except Exception as e:
                logger.error("Parse error: %s; text=%s", e, content)
                return False, [], ""

            # Process detections into Cell objects
            converted_bounding_boxes = []
            labels = []

            for i, it in enumerate(raw):
                if not isinstance(it, dict):
                    continue
                label_norm = self._norm_label(it.get("label", ""))
                if label_norm not in name_to_id:
                    # Try partial matching
                    matched = False
                    for k in name_to_id:
                        if label_norm in k or k in label_norm:
                            matched = True
                            break
                    if not matched:
                        continue
                conf = it.get("confidence", 1.0)
                try:
                    conf = float(conf)
                except:
                    conf = 1.0
                if conf < self.confidence_threshold:
                    continue
                box = it.get("box_2d") or it.get("box")
                if not (isinstance(box, list) and len(box) == 4):
                    if isinstance(box, dict):
                        box = [box.get("y_min", 0), box.get("x_min", 0), box.get("y_max", 0), box.get("x_max", 0)]
                    else:
                        continue

                # Convert normalized (0..1000) to absolute coordinates (using original_size)
                abs_y1 = int(box[0] / 1000 * original_size[1])
                abs_x1 = int(box[1] / 1000 * original_size[0])
                abs_y2 = int(box[2] / 1000 * original_size[1])
                abs_x2 = int(box[3] / 1000 * original_size[0])
                if abs_y1 >= abs_y2 or abs_x1 >= abs_x2:
                    continue

                cell = self.Cell(id=i, left=abs_x1, top=abs_y1, right=abs_x2, bottom=abs_y2)
                converted_bounding_boxes.append(cell)
                labels.append(it.get("label", ""))

            # Sort by confidence descending if not already (assuming prompt followed, but enforce)
            if converted_bounding_boxes:
                # Need conf per item; since raw has conf, sort raw first
                raw_sorted = sorted(raw, key=lambda x: x.get("confidence", 0), reverse=True)
                # Re-process sorted
                converted_bounding_boxes = []
                labels = []
                for i, it in enumerate(raw_sorted):
                    label = it.get("label", "")
                    box = it.get("box_2d", [])
                    abs_y1 = int(box[0] / 1000 * original_size[1])
                    abs_x1 = int(box[1] / 1000 * original_size[0])
                    abs_y2 = int(box[2] / 1000 * original_size[1])
                    abs_x2 = int(box[3] / 1000 * original_size[0])
                    if abs_y1 >= abs_y2 or abs_x1 >= abs_x2:
                        continue
                    cell = self.Cell(id=i, left=abs_x1, top=abs_y1, right=abs_x2, bottom=abs_y2)
                    converted_bounding_boxes.append(cell)
                    labels.append(label)

            # Apply multiple_predictions logic
            if not self.multiple_predictions and converted_bounding_boxes:
                converted_bounding_boxes = [converted_bounding_boxes[0]]
                labels = [labels[0]]

            # Convert Cell objects to YOLO format
            yolo = []
            for cell in converted_bounding_boxes:
                label = labels[cell.id]
                cid = name_to_id.get(self._norm_label(label))
                if cid is None:
                    continue
                x0, y0, x1, y1 = cell.left, cell.top, cell.right, cell.bottom
                yolo.append(self._bbox_to_yolo(x0, y0, x1, y1, original_size[0], original_size[1], cid))

            return (len(yolo) > 0), yolo, (", ".join(labels) if labels else ", ".join(queries))

        except Exception as e:
            logger.exception("OpenRouter detector error")
            return False, [], f"Error processing detection: {e}"