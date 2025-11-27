from typing import List, Tuple, Dict, Any, Set, Optional
from PIL import Image
import os
import json
import base64
import requests
import logging
import math
import time
import re
import ast
import random
from io import BytesIO
from ..enhance import edsr_enhance

logger = logging.getLogger(__name__)

class AdvancedReasoningDetection:
    """
    Advanced spatial reasoning detection pipeline inspired by spatial-reasoning repository.
    Uses coarse detection, grid occupancy, tile detection, refinement, and NMS.
    Updated to use tool-calling for dynamic reasoning, allowing the LLM to decide on stages recursively for better accuracy.
    """

    class Cell:
        def __init__(self, id: int, left: int, top: int, right: int, bottom: int):
            self.id = id
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom

    class BaseAgent:
        def __init__(self, model: str, api_key: Optional[str] = None):
            self.model = model
            self.api_key = api_key
            self._client = None

        @property
        def client(self):
            if self._client is None:
                self._client = requests.Session()
                self._client.headers.update({
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    # OpenRouter attribution headers
                    "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
                    "X-Title": os.getenv("APP_NAME", "Spatial Reasoning Detector"),
                })
            return self._client

        def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            IMPORTANT: Preserve tool calling fields: 'tool_calls', 'tool_call_id', and 'name'.
            Convert text/list content into the OpenAI-compatible 'content' format.
            """
            formatted = []
            for msg in messages:
                role = msg.get("role")
                obj: Dict[str, Any] = {"role": role}

                # Pass through assistant tool calls if present
                if role == "assistant" and "tool_calls" in msg:
                    obj["tool_calls"] = msg["tool_calls"]
                    # content may be empty or a string
                    obj["content"] = msg.get("content", "")

                # Tool role: must include tool_call_id (and name helps)
                elif role == "tool":
                    if "tool_call_id" in msg:
                        obj["tool_call_id"] = msg["tool_call_id"]
                    if "name" in msg:
                        obj["name"] = msg["name"]
                    content = msg.get("content", "")
                    # Tool content must be a string
                    if isinstance(content, (dict, list)):
                        content = json.dumps(content)
                    obj["content"] = content

                else:
                    # Normal user/system/assistant content handling (including multimodal)
                    msg_content = msg.get("content", [])
                    if isinstance(msg_content, str):
                        obj["content"] = msg_content
                    elif isinstance(msg_content, list):
                        content_items = []
                        for part in msg_content:
                            if isinstance(part, dict) and "type" in part:
                                if part["type"] == "text":
                                    content_items.append({"type": "text", "text": part["text"]})
                                elif part["type"] == "image_url":
                                    content_items.append({"type": "image_url", "image_url": part["image_url"]})
                            else:
                                logger.warning(f"Invalid part in message content: {part}")
                        obj["content"] = content_items
                    else:
                        logger.warning(f"Unexpected content type: {type(msg_content)}")
                        obj["content"] = str(msg_content)

                formatted.append(obj)
            return formatted

        def chat(
            self,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs,
        ) -> Dict[str, Any]:
            logger.debug(f"Sending messages: {json.dumps(messages, indent=2)[:2000]}")
            body: Dict[str, Any] = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "temperature": 0.1 if temperature is None else temperature,  # Lower temperature for more consistent reasoning
                "max_tokens": max_tokens or 4096,  # Increased for complex reasoning
                "top_p": 0.9,  # Add top_p for better token selection
                **kwargs
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"

            r = self.client.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")
            data = r.json()
            choice = (data.get("choices") or [{}])[0].get("message", {})
            output = {
                "output": choice.get("content", "") or "",
            }
            if "tool_calls" in choice and choice["tool_calls"]:
                output["tool_calls"] = choice["tool_calls"]
            return output

        def safe_chat(
            self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, max_attempts: int = 3, **kwargs
        ) -> Dict[str, Any]:
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return self.chat(messages, tools, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break
                    wait = (2**attempt) + random.random()
                    print(f"[retry {attempt}/{max_attempts}] {e}. retrying in {wait:.1f}s")
                    if "Too many tokens" in str(e):
                        print("Rate limit exceeded. Sleeping for 1 minute.")
                        time.sleep(60)
                        continue
                    time.sleep(wait)
            raise last_exc  # surface after retries

        @staticmethod
        def create_text_message(role: str, content: str) -> Dict[str, Any]:
            return {"role": role, "content": content}

        @staticmethod
        def create_multimodal_message(
            role: str,
            text: str,
            images: List[Image.Image],
            image_size=1024
        ) -> Dict[str, Any]:
            content = [{"type": "text", "text": text}]
            for image in images:
                image_url = f"data:image/jpeg;base64,{AdvancedReasoningDetection.image_to_base64(image, image_size)}"
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            return {"role": role, "content": content}

    def __init__(self,
                 model_name: str,
                 confidence_threshold: float = 0.3,
                 grid_rows: int = 16,
                 grid_cols: int = 16,
                 tile_overlap: float = 0.3,
                 max_side_px: int = 1536,
                 max_total_tiles: int = 48,
                 iou_nms: float = 0.3,
                 refine_max: int = 24,
                 refine_pad: float = 0.3,
                 problem_statement: str = "",
                 sample_images: List[Image.Image] = None,
                 image_size=1536,
                 upscale_image=False):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.grid_rows = max(1, grid_rows)
        self.grid_cols = max(1, grid_cols)
        self.tile_overlap = max(0.0, min(0.49, tile_overlap))
        self.max_side_px = max_side_px
        self.max_total_tiles = max_total_tiles
        self.iou_nms = iou_nms
        self.refine_max = refine_max
        self.refine_pad = max(0.0, min(0.4, refine_pad))
        self.problem_statement = problem_statement
        self.sample_images = sample_images or []
        self.image_size = image_size
        self.upscale_image = upscale_image
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY env var not set")
        self.agent = self.BaseAgent(model=model_name, api_key=self.api_key)
        print(f"🧭 Reasoning-ADV model={self.model_name} | grid {self.grid_rows}x{self.grid_cols} overlap {self.tile_overlap} | refine {self.refine_max}")

        # Define tools for tool-calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "coarse_detection",
                    "description": "Perform initial coarse detection scan on the full image to establish baseline detections and global context.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_grid_occupancy",
                    "description": "Perform spatial grid analysis to identify cells containing target objects. Use strategic grid sizes: 3x3 for broad regions, 6x6-8x8 for intermediate, or fine grids like 16x16 for detailed analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rows": {"type": "integer", "description": "Number of grid rows. Recommended: 3 for coarse, 6-8 for medium, 12-16 for fine analysis."},
                            "cols": {"type": "integer", "description": "Number of grid columns. Should match rows for square cells or adapt to image aspect ratio."}
                        },
                        "required": ["rows", "cols"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cells_to_tiles",
                    "description": "Convert promising grid cells to pixel-aligned tiles for detailed analysis. Automatically merges overlapping tiles to optimize processing efficiency.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rows": {"type": "integer", "description": "Grid rows used in grid analysis."},
                            "cols": {"type": "integer", "description": "Grid columns used in grid analysis."},
                            "cells": {
                                "type": "array",
                                "description": "Array of [row, col] coordinate pairs for positive grid cells.",
                                "items": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "items": {"type": "integer"}
                                }
                            },
                            "overlap": {"type": "number", "description": "Overlap factor (0.0-0.5) to ensure edge objects aren't missed. Default uses configured value."}
                        },
                        "required": ["rows", "cols", "cells"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_tile",
                    "description": "Perform focused object detection on a specific tile region with higher resolution analysis than coarse detection. Limited by tile budget.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x0": {"type": "integer", "description": "Left boundary of tile in image coordinates."},
                            "y0": {"type": "integer", "description": "Top boundary of tile in image coordinates."},
                            "x1": {"type": "integer", "description": "Right boundary of tile in image coordinates."},
                            "y1": {"type": "integer", "description": "Bottom boundary of tile in image coordinates."}
                        },
                        "required": ["x0", "y0", "x1", "y1"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "refine_detection",
                    "description": "Refine a specific detection with pixel-level precision analysis. Focus on uncertain detections or important objects. Limited by refinement budget.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Object class label to refine."},
                            "conf": {"type": "number", "description": "Current confidence score (0.0-1.0)."},
                            "box_xyxy": {
                                "type": "array",
                                "description": "Current bounding box coordinates [x0, y0, x1, y1] in image pixels.",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4
                            }
                        },
                        "required": ["label", "conf", "box_xyxy"]
                    }
                }
            }
        ]

    # ----- Utility Methods -----
    @staticmethod
    def image_to_base64(img: Image.Image, image_size=1024) -> str:
        """Convert PIL Image to base64 JPEG string (smaller than PNG)."""
        im = img.copy()
        im.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        im = im.convert("RGB")
        im.save(buffer, format="JPEG", quality=88, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0: return 0.0
        return inter / max(1e-6, ((ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter))

    @staticmethod
    def _nms(dets: List[Dict[str, Any]], iou_thr=0.5):
        keep = []
        pool = sorted(dets, key=lambda d: d["conf"], reverse=True)
        while pool:
            cur = pool.pop(0)
            keep.append(cur)
            pool = [d for d in pool if not (d["label"].lower()==cur["label"].lower() and AdvancedReasoningDetection._iou(d["box"], cur["box"]) > iou_thr)]
        return keep

    @staticmethod
    def _clip_box(x0, y0, x1, y1, W, H):
        x0 = max(0, min(W, int(x0))); x1 = max(0, min(W, int(x1)))
        y0 = max(0, min(H, int(y0))); y1 = max(0, min(H, int(y1)))
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        if x0 == x1 or y0 == y1: return None
        return (x0, y0, x1, y1)

    @staticmethod
    def _merge_overlapping_tiles(tiles, W, H, merge_threshold=0.7):
        """Merge overlapping tiles to reduce redundant processing while maintaining coverage."""
        if len(tiles) <= 1:
            return tiles
            
        merged = []
        used = [False] * len(tiles)
        
        for i, tile1 in enumerate(tiles):
            if used[i]:
                continue
                
            # Start with current tile
            x0, y0, x1, y1 = tile1
            merged_with = [i]
            
            # Look for tiles to merge with this one
            for j, tile2 in enumerate(tiles[i+1:], i+1):
                if used[j]:
                    continue
                    
                tx0, ty0, tx1, ty1 = tile2
                
                # Calculate overlap
                overlap_area = max(0, min(x1, tx1) - max(x0, tx0)) * max(0, min(y1, ty1) - max(y0, ty0))
                tile1_area = (x1 - x0) * (y1 - y0)
                tile2_area = (tx1 - tx0) * (ty1 - ty0)
                
                # Merge if significant overlap
                overlap_ratio = overlap_area / min(tile1_area, tile2_area)
                if overlap_ratio > merge_threshold:
                    # Expand bounding box to include both tiles
                    x0 = min(x0, tx0)
                    y0 = min(y0, ty0) 
                    x1 = max(x1, tx1)
                    y1 = max(y1, ty1)
                    merged_with.append(j)
            
            # Mark all merged tiles as used
            for idx in merged_with:
                used[idx] = True
                
            # Add the merged tile
            merged.append([x0, y0, x1, y1])
        
        return merged

    # ----- Parsing Methods -----
    @staticmethod
    def parse_detection_output(output_text):
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
            result = AdvancedReasoningDetection._try_parse_json_or_python(fence_content)
            if result is not None:
                return result
        cleaned = re.sub(r"^```.*?\n|```$", "", cleaned, flags=re.S)
        cleaned = re.sub(r"//.*", "", cleaned)
        cleaned = re.sub(r"\(\s*(\d+)\s*\)", r"[\1]", cleaned)
        candidates = AdvancedReasoningDetection._extract_all_structures(cleaned)
        dicts = [c for c in candidates if c["text"].strip().startswith("{")]
        lists = [c for c in candidates if c["text"].strip().startswith("[")]
        for candidate in dicts:
            result = AdvancedReasoningDetection._try_parse_json_or_python(candidate["text"])
            if result is not None:
                return result
        for candidate in lists:
            result = AdvancedReasoningDetection._try_parse_json_or_python(candidate["text"])
            if isinstance(result, list) and result and isinstance(result[0], dict):
                return result[0]
        return None

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
        converted = AdvancedReasoningDetection._convert_python_to_json(text)
        if converted != text:
            try:
                return json.loads(converted)
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    @staticmethod
    def _convert_python_to_json(text):
        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)
        text = re.sub(r"\bNone\b", "null", text)
        text = re.sub(r"'([^']*)'", r'"\1"', text)
        return text

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
                end_pos = AdvancedReasoningDetection._find_matching_bracket(text, i)
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

    # ----- LLM Call -----
    def _call_llm(self, messages: List[Dict[str, Any]], **kwargs) -> dict:
        """
        Calls the LLM and tries to parse the returned JSON. Includes fallbacks for
        providers/models that don't support 'json_schema' response_format.
        """
        def _invoke(**kw):
            return self.agent.safe_chat(messages, **kw)

        try:
            resp = _invoke(**kwargs)
        except Exception as e1:
            # Fallback to json_object
            if "response_format" in kwargs:
                try:
                    kw2 = dict(kwargs)
                    kw2["response_format"] = {"type": "json_object"}
                    resp = _invoke(**kw2)
                except Exception as e2:
                    # Final fallback: no response_format
                    try:
                        kw3 = dict(kwargs)
                        kw3.pop("response_format", None)
                        resp = _invoke(**kw3)
                    except Exception as e3:
                        logger.error(f"LLM call failed (all fallbacks): {e3}")
                        return {"detections": [], "positive_cells": []}
            else:
                logger.error(f"LLM call failed: {e1}")
                return {"detections": [], "positive_cells": []}

        content = resp.get("output", "")
        logger.debug(f"Raw LLM response: {content[:500]}...")
        if "tool_calls" in resp and resp["tool_calls"]:
            # If the assistant is asking to call tools, just return passthrough marker
            return {"__tool_calls__": resp["tool_calls"], "__content__": content}

        if not content.strip():
            logger.warning("Empty content from LLM")
            return {"detections": [], "positive_cells": []}

        # Try direct JSON parse
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                logger.warning(f"Expected dict, got {type(parsed)}")
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    parsed = parsed[0]
                else:
                    parsed = {"detections": [], "positive_cells": []}
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.warning(f"Raw content (first 500 chars): {content[:500]}")
            parsed = self.parse_detection_output(content)
            if parsed is None or isinstance(parsed, list):
                logger.warning(f"Parsed result invalid: {parsed}")
                return {"detections": [], "positive_cells": []}
            return parsed

    # ----- Prompt Methods -----
    def get_coarse_prompt(self, queries: List[str], W: int, H: int, confidence_threshold: float, problem_statement: str) -> str:
        query_str = ", ".join(queries) if queries else "objects"
        prompt = f"""You are an expert computer vision system performing initial object detection analysis.

TASK: Detect all instances of {query_str} in this image with maximum precision.

IMAGE SPECIFICATIONS:
- Dimensions: {W}x{H} pixels
- Confidence threshold: {confidence_threshold}

CONTEXT: {problem_statement}

DETECTION REQUIREMENTS:
1. Scan the entire image systematically from left to right, top to bottom
2. Look for objects at multiple scales (small, medium, large)
3. Consider partial objects, occluded objects, and objects in shadows/lighting variations
4. Pay special attention to object boundaries and ensure tight bounding boxes
5. Include objects with any reasonable probability of being the target class

OUTPUT FORMAT:
For each detection, provide:
- label: exact object class name from target list
- confidence: detection confidence (0.0 to 1.0, be conservative but not overly strict)
- box_xyxy: precise bounding box [x0, y0, x1, y1] in absolute pixels

QUALITY STANDARDS:
- Prefer recall over precision (better to include uncertain detections than miss objects)
- Ensure bounding boxes tightly fit the objects
- Minimum confidence should reflect actual detection certainty
- Consider context clues and object relationships

Return ONLY valid JSON with "detections" array. No explanatory text."""
        return prompt

    def get_grid_prompt(self, queries: List[str], grid_rows: int, grid_cols: int, W: int, H: int) -> str:
        query_str = ", ".join(queries) if queries else "objects"
        cell_width = W // grid_cols
        cell_height = H // grid_rows
        
        prompt = f"""You are performing advanced spatial reasoning for object localization.

TASK: Identify grid cells containing any part of {query_str} using systematic spatial analysis.

GRID SPECIFICATION:
- Grid size: {grid_rows}x{grid_cols} ({grid_rows * grid_cols} total cells)
- Cell dimensions: ~{cell_width}x{cell_height} pixels each
- Image dimensions: {W}x{H} pixels

SPATIAL ANALYSIS METHODOLOGY:
1. Mentally overlay the grid on the image
2. Systematically examine each cell for object presence
3. Mark a cell as positive if it contains ANY portion of the target object(s)
4. Consider object boundaries that span multiple cells
5. Include cells with partial objects, edges, or even small object parts
6. Use conservative threshold: when in doubt, include the cell

GRID COORDINATE SYSTEM:
- Rows: 0 to {grid_rows-1} (top to bottom)  
- Columns: 0 to {grid_cols-1} (left to right)
- Cell [0,0] is top-left, [{grid_rows-1},{grid_cols-1}] is bottom-right

DETECTION STRATEGY:
- Scan systematically: left-to-right, top-to-bottom
- Look for complete objects, partial objects, and object fragments  
- Consider objects at cell boundaries and overlapping regions
- Include cells with high probability of containing target objects

Return ONLY valid JSON with "positive_cells" array containing [row, column] coordinate pairs.
Example: {{"positive_cells": [[0,1], [2,3], [5,7]]}}"""
        return prompt

    def get_tile_prompt(self, queries: List[str], gx0: int, gy0: int, gx1: int, gy1: int, tile_w: int, tile_h: int, W: int, H: int) -> str:
        query_str = ", ".join(queries) if queries else "objects"
        
        prompt = f"""You are analyzing a focused tile region for precise object detection.

TASK: Detect all instances of {query_str} in this detailed image tile with maximum accuracy.

TILE CONTEXT:
- Tile region in full image: ({gx0}, {gy0}) to ({gx1}, {gy1})
- Tile dimensions: {tile_w}x{tile_h} pixels  
- Full image dimensions: {W}x{H} pixels
- Coverage: {((gx1-gx0)*(gy1-gy0)/(W*H)*100):.1f}% of total image area

DETECTION STRATEGY:
1. This tile provides higher resolution detail than coarse detection
2. Look for objects that may have been missed or poorly localized in coarse pass
3. Focus on precise boundary detection and small object discovery
4. Include partial objects at tile edges (they may complete in adjacent tiles)
5. Re-examine areas that appeared uncertain in broader analysis

ACCURACY REQUIREMENTS:
- Provide pixel-perfect bounding boxes for maximum precision
- Consider lighting, shadows, and perspective variations within the tile
- Detect objects at multiple scales within this focused region
- Include objects with confidence >= 0.3 (lower threshold for focused analysis)

COORDINATE SYSTEM:
- All bounding boxes must be in FULL IMAGE coordinates
- Convert tile-relative positions back to full image space
- Ensure boxes are within the full image bounds [0,0] to [{W},{H}]

OUTPUT FORMAT:
For each detection:
- label: exact object class name from: {query_str}
- confidence: detection confidence (0.0 to 1.0)
- box_xyxy: [x0, y0, x1, y1] in full image pixel coordinates

Return ONLY valid JSON with "detections" array. Be thorough but precise."""
        return prompt

    def get_refine_prompt(self, label: str, conf: float, x0: int, y0: int, x1: int, y1: int, orig_width: int, orig_height: int, X0: int, Y0: int, X1: int, Y1: int, crop_w: int, crop_h: int, W: int, H: int) -> str:
        prompt = f"""You are performing precision refinement of an object detection.

TASK: Refine the bounding box for this {label} object with pixel-level accuracy.

DETECTION CONTEXT:
- Object class: {label}
- Initial confidence: {conf:.3f}
- Original bounding box: ({x0}, {y0}) to ({x1}, {y1})
- Original box size: {orig_width}x{orig_height} pixels

CROP ANALYSIS CONTEXT:
- Crop region in full image: ({X0}, {Y0}) to ({X1}, {Y1})  
- Crop dimensions: {crop_w}x{crop_h} pixels
- Full image dimensions: {W}x{H} pixels
- Crop padding: {max(X0-x0, Y0-y0, x1-X1, y1-Y1)} pixels around original detection

REFINEMENT OBJECTIVES:
1. BOUNDARY PRECISION: Adjust bounding box to tightly encompass the object
2. CONFIDENCE CALIBRATION: Update confidence based on detailed analysis
3. OBJECT COMPLETENESS: Ensure the box captures the complete object
4. BACKGROUND EXCLUSION: Minimize background inclusion in the bounding box

ANALYSIS METHODOLOGY:
- Examine object edges and boundaries at high resolution  
- Consider object pose, orientation, and partial occlusion
- Account for shadows, reflections, or visual artifacts
- Verify object class certainty with detailed view
- Optimize box coordinates for minimal background inclusion

QUALITY STANDARDS:
- Box should tightly fit the object with minimal padding
- Include all visible parts of the object
- Exclude background pixels where possible
- Update confidence to reflect detection certainty at this resolution
- Ensure coordinates remain within full image bounds

COORDINATE REQUIREMENTS:
- Return coordinates in FULL IMAGE space (not crop-relative)
- All coordinates must be integers within [0,{W}] x [0,{H}]
- Ensure x0 < x1 and y0 < y1

OUTPUT FORMAT:
Single detection with:
- label: "{label}" (must match exactly)
- confidence: refined score (0.0 to 1.0) 
- box_xyxy: [x0, y0, x1, y1] in full image coordinates

Return ONLY valid JSON with "detections" array containing one refined detection."""
        return prompt

    # ----- Tool Execution Methods -----
    def _execute_tool(self, tool_call: Dict[str, Any], image: Image.Image, queries: List[str]) -> str:
        function_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        W, H = image.size
        if function_name == "coarse_detection":
            print("🛠️ Tool call: coarse_detection")
            prompt = self.get_coarse_prompt(queries, W, H, self.confidence_threshold, self.problem_statement)
            messages = [
                self.agent.create_text_message("user", "Output valid JSON only."),
                self.agent.create_multimodal_message("user", prompt, [image] + self.sample_images, self.image_size)
            ]
            resp = self._call_llm(messages, response_format={"type": "json_schema", "json_schema": self._schema_det()})
            if "__tool_calls__" in resp:
                # If assistant asked for more tools while inside tool execution (rare), ignore and return empty
                return json.dumps({"detections": []})
            return json.dumps(resp)

        elif function_name == "get_grid_occupancy":
            rows = args["rows"]
            cols = args["cols"]
            print(f"🛠️ Tool call: get_grid_occupancy ({rows}x{cols})")
            prompt = self.get_grid_prompt(queries, rows, cols, W, H)
            messages = [
                self.agent.create_text_message("user", "Output valid JSON only."),
                self.agent.create_multimodal_message("user", prompt, [image] + self.sample_images, self.image_size)
            ]
            resp = self._call_llm(messages, response_format={"type": "json_schema", "json_schema": self._schema_grid()})
            if "__tool_calls__" in resp:
                return json.dumps({"positive_cells": []})
            return json.dumps(resp)

        elif function_name == "cells_to_tiles":
            rows = int(args["rows"]); cols = int(args["cols"])
            overlap = float(args.get("overlap", self.tile_overlap))
            cells = args["cells"]
            print(f"🛠️ Tool call: cells_to_tiles ({rows}x{cols}) overlap={overlap:.2f}")
            cell_w, cell_h = W / cols, H / rows
            tiles = []
            
            # Improved tile generation with smart overlap and merging
            for r, c in cells:
                # Calculate base tile boundaries
                base_x0 = c * cell_w
                base_y0 = r * cell_h
                base_x1 = (c + 1) * cell_w
                base_y1 = (r + 1) * cell_h
                
                # Apply overlap padding
                overlap_w = overlap * cell_w
                overlap_h = overlap * cell_h
                
                x0 = max(0, int(base_x0 - overlap_w))
                y0 = max(0, int(base_y0 - overlap_h))
                x1 = min(W, int(base_x1 + overlap_w))
                y1 = min(H, int(base_y1 + overlap_h))
                
                # Ensure minimum tile size
                min_size = 32
                if x1 - x0 >= min_size and y1 - y0 >= min_size:
                    tiles.append([x0, y0, x1, y1])
            
            # Merge overlapping tiles to optimize processing
            merged_tiles = self._merge_overlapping_tiles(tiles, W, H)
            print(f"📦 Generated {len(tiles)} tiles, merged to {len(merged_tiles)} tiles")
            return json.dumps({"tiles": merged_tiles})

        elif function_name == "analyze_tile":
            x0, y0, x1, y1 = args["x0"], args["y0"], args["x1"], args["y1"]
            print(f"🛠️ Tool call: analyze_tile ({x0},{y0})-({x1},{y1})")
            tile_img = image.crop((x0, y0, x1, y1))
            tile_w, tile_h = tile_img.size
            prompt = self.get_tile_prompt(queries, x0, y0, x1, y1, tile_w, tile_h, W, H)
            messages = [
                self.agent.create_text_message("user", "Output valid JSON only."),
                self.agent.create_multimodal_message("user", prompt, [tile_img] + self.sample_images, self.image_size)
            ]
            resp = self._call_llm(messages, response_format={"type": "json_schema", "json_schema": self._schema_det()})
            if "__tool_calls__" in resp:
                return json.dumps({"detections": []})
            return json.dumps(resp)

        elif function_name == "refine_detection":
            label = args["label"]
            conf = args["conf"]
            box = args["box_xyxy"]
            x0, y0, x1, y1 = box
            print(f"🛠️ Tool call: refine_detection {label} ({x0},{y0})-({x1},{y1})")
            orig_width, orig_height = x1 - x0, y1 - y0
            pw = max(int(orig_width * self.refine_pad), 20)
            ph = max(int(orig_height * self.refine_pad), 20)
            X0 = max(0, x0 - pw); Y0 = max(0, y0 - ph)
            X1 = min(W, x1 + pw); Y1 = min(H, y1 + ph)
            crop = image.crop((X0, Y0, X1, Y1))
            crop_w, crop_h = crop.size
            prompt = self.get_refine_prompt(label, conf, x0, y0, x1, y1, orig_width, orig_height, X0, Y0, X1, Y1, crop_w, crop_h, W, H)
            messages = [
                self.agent.create_text_message("user", "Output valid JSON only."),
                self.agent.create_multimodal_message("user", prompt, [crop] + self.sample_images, self.image_size)
            ]
            resp = self._call_llm(messages, response_format={"type": "json_schema", "json_schema": self._schema_det()})
            if "__tool_calls__" in resp:
                return json.dumps({"detections": []})
            return json.dumps(resp)

        return json.dumps({"error": "Unknown tool"})

    # ----- Pipeline Stages (now as tools) -----
    def _parse_detections(self, resp: dict) -> List[Dict[str, Any]]:
        out = []
        for it in resp.get("detections", []):
            try:
                label = str(it["label"]).strip()
                conf = float(it.get("confidence", 1.0))
                x0, y0, x1, y1 = [int(round(float(v))) for v in it["box_xyxy"]]
                box = self._clip_box(x0, y0, x1, y1, self.current_W, self.current_H)
                if box and conf >= self.confidence_threshold:
                    out.append({"label": label, "conf": conf, "box": box})
            except Exception:
                continue
        return out

    def _to_yolo(self, dets: List[Dict[str,Any]], name_to_id: Dict[str,int], W: int, H: int):
        yolo = []
        for d in dets:
            lab = d["label"]
            cid = name_to_id.get(lab)
            if cid is None:
                for k, v in name_to_id.items():
                    if k.lower() == lab.lower():
                        cid = v; break
            if cid is None:
                continue
            x0, y0, x1, y1 = d["box"]
            xc = ((x0 + x1) / 2) / W; yc = ((y0 + y1) / 2) / H
            w = (x1 - x0) / W; h = (y1 - y0) / H
            yolo.append([cid, xc, yc, w, h])
        return yolo

    def _schema_det(self):
        return {
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
                                "box_xyxy": {
                                    "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4
                                },
                            },
                            "required": ["label", "confidence", "box_xyxy"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["detections"],
                "additionalProperties": False
            },
        }

    def _schema_grid(self):
        return {
            "name": "grid_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "positive_cells": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 2, "maxItems": 2,
                            "items": {"type": "integer"}
                        }
                    }
                },
                "required": ["positive_cells"],
                "additionalProperties": False
            }
        }

    def _schema_final(self):
        return {
            "name": "final_detections",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "final_detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "confidence": {"type": "number"},
                                "box_xyxy": {
                                    "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4
                                },
                            },
                            "required": ["label", "confidence", "box_xyxy"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["final_detections"],
                "additionalProperties": False
            }
        }

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        queries = class_names if class_names else []
        name_to_id = {n: cid for n, cid in zip(class_names, class_ids)}
        self.current_W, self.current_H = image.size  # Store for parsing
        print(f"\n🚀 Starting Advanced Reasoning Detection Pipeline with Tool Use")
        print(f"📐 Image: {self.current_W}x{self.current_H}, Target classes: {queries}")
        if self.upscale_image:
            try:
                image = edsr_enhance(image)
            except Exception as e:
                print(f"Warning: Image enhancement failed, using original image: {e}")
                print("Using original image...")
                # Continue with the original image
        # System prompt explaining the process and tools
        system_prompt = f"""You are an advanced spatial reasoning AI system specializing in precise object detection using systematic multi-stage analysis.

TARGET OBJECTS: {', '.join(queries)}
PROBLEM CONTEXT: {self.problem_statement}

DETECTION METHODOLOGY - Use tools in this strategic sequence:

STAGE 1 - COARSE ANALYSIS:
- Use coarse_detection on full image to establish baseline detections
- This provides global context and initial object candidates

STAGE 2 - SPATIAL GRID ANALYSIS:
- Use get_grid_occupancy with strategic grid sizes:
  * Start with 3x3 for broad region identification
  * Progress to {self.grid_rows}x{self.grid_cols} for fine-grained analysis
  * Consider 6x6 or 8x8 for intermediate analysis if needed
- Grid analysis helps locate objects missed in coarse detection

STAGE 3 - FOCUSED TILE EXAMINATION:
- Use cells_to_tiles to convert promising grid cells to pixel regions
- Set overlap ~{self.tile_overlap} to ensure edge objects aren't lost
- Use analyze_tile on high-confidence regions for detailed detection
- Prioritize tiles with multiple positive grid cells

STAGE 4 - PRECISION REFINEMENT:
- Use refine_detection on individual detections for pixel-level accuracy
- Focus refinement on uncertain detections or important objects
- Update confidence scores based on detailed analysis

BUDGET CONSTRAINTS:
- Maximum {self.max_total_tiles} analyze_tile calls - use strategically
- Maximum {self.refine_max} refine_detection calls - prioritize best candidates
- Balance coverage with depth of analysis

STRATEGIC DECISION MAKING:
- Adapt grid resolution based on object size and image complexity
- Use iterative refinement: coarse→grid→tile→refine
- Consider recursive analysis for complex regions
- Merge complementary detections from different stages

FINAL OUTPUT REQUIREMENTS:
- Once analysis is complete, output 'final_detections' JSON array
- Include: label, confidence, box_xyxy for each detection
- Apply reasoning-based NMS to eliminate redundant detections
- Ensure high recall while maintaining precision

IMAGE SPECIFICATIONS: {self.current_W}x{self.current_H} pixels

Execute this systematic approach to achieve superior detection performance."""

        messages = [
            self.agent.create_text_message("system", system_prompt),
            self.agent.create_multimodal_message("user", "Detect the objects in the image.", [image], self.image_size)
        ]

        max_iterations = 40  # Increased for more complex multi-stage reasoning
        all_detections: List[Dict[str, Any]] = []
        tile_calls = 0
        refine_calls = 0
        best_count = 0
        stale_iters = 0
        final_dets = None
        stage_progress = {"coarse": False, "grid": False, "tiles": False, "refine": False}

        for iteration in range(max_iterations):
            print(f"🔄 Agent iteration {iteration + 1}")
            # Ask the agent to think & possibly call tools
            resp = self.agent.safe_chat(messages, tools=self.tools)

            # Append assistant turn with tool calls (preserve tool_calls!)
            tool_calls = resp.get("tool_calls", [])
            messages.append({"role": "assistant", "content": resp.get("output", ""), "tool_calls": tool_calls})

            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call["id"]
                    tool_output = self._execute_tool(tool_call, image, queries)

                    # Return tool result with proper wiring
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_call["function"]["name"],
                        "content": tool_output
                    })

                    # Parse & collect detections from this tool output
                    try:
                        tool_resp = json.loads(tool_output)
                        dets = self._parse_detections(tool_resp)
                        before = len(all_detections)
                        all_detections.extend(dets)
                        after = len(all_detections)
                        if after > before:
                            print(f"➕ Collected {after - before} new proposals (total {after})")
                    except Exception as e:
                        logger.warning(f"Failed to parse tool output: {e}")

                # Update budgets and stage tracking based on which tools were called
                names = [tc["function"]["name"] for tc in tool_calls]
                tile_calls += sum(1 for n in names if n == "analyze_tile")
                refine_calls += sum(1 for n in names if n == "refine_detection")
                
                # Track stage progress
                if "coarse_detection" in names:
                    stage_progress["coarse"] = True
                if "get_grid_occupancy" in names:
                    stage_progress["grid"] = True
                if "analyze_tile" in names:
                    stage_progress["tiles"] = True
                if "refine_detection" in names:
                    stage_progress["refine"] = True

                # Budget management and progression guidance
                budget_msg = ""
                if tile_calls >= self.max_total_tiles:
                    budget_msg += f"Tile budget exhausted ({tile_calls}/{self.max_total_tiles}). "
                if refine_calls >= self.refine_max:
                    budget_msg += f"Refinement budget exhausted ({refine_calls}/{self.refine_max}). "
                    
                if budget_msg:
                    messages.append(self.agent.create_text_message("system", 
                        f"{budget_msg}Finalize detection with current results using 'final_detections'."))
                elif iteration > 20 and not any(stage_progress.values()):
                    messages.append(self.agent.create_text_message("system", 
                        "Consider starting with coarse_detection or get_grid_occupancy to begin systematic analysis."))

                # Track improvement & early finalize if stale
                if len(all_detections) > best_count:
                    best_count = len(all_detections); stale_iters = 0
                    print(f"📈 Improved: {best_count} total detections")
                else:
                    stale_iters += 1
                    if stale_iters >= 5:  # Increased patience for complex reasoning
                        messages.append(self.agent.create_text_message("system", 
                            f"Analysis complete with {len(all_detections)} detections. Finalize with 'final_detections'."))
                continue  # next iteration to let model react to tool outputs

            # No tool calls -> assistant tried to finalize
            final_content = resp.get("output", "")
            parsed_final = self.parse_detection_output(final_content)
            if parsed_final and "final_detections" in parsed_final:
                final_dets = parsed_final["final_detections"]
            else:
                # Fallback: ask for strict final schema using what we collected so far
                finalize_msgs = messages + [
                    self.agent.create_text_message("system",
                        "Format the full set of detections as 'final_detections' only, no prose."),
                    self.agent.create_text_message("user",
                        json.dumps({"detections":[{"label": d["label"], "confidence": d["conf"], "box_xyxy": d["box"]} for d in all_detections]}))
                ]
                resp2 = self.agent.safe_chat(
                    finalize_msgs,
                    response_format={"type":"json_schema","json_schema": self._schema_final()}
                )
                parsed2 = self.parse_detection_output(resp2.get("output",""))
                if parsed2 and "final_detections" in parsed2:
                    final_dets = parsed2["final_detections"]
                else:
                    # Last resort: use collected detections as-is
                    final_dets = [{"label": d["label"], "confidence": d["conf"], "box_xyxy": d["box"]} for d in all_detections]
            break  # exit loop after finalization

        if final_dets is None:
            print("⚠️ Max iterations reached or no explicit finalization; using collected proposals.")
            final_dets = [{"label": d["label"], "confidence": d["conf"], "box_xyxy": d["box"]} for d in all_detections]

        # Final NMS on collected/refined detections
        proposals = self._nms(
            [{"label": it["label"], "conf": float(it["confidence"]), "box": tuple(it["box_xyxy"])} for it in final_dets],
            self.iou_nms
        )
        print(f"🎯 Final after NMS: {len(proposals)} detections")

        yolo = self._to_yolo(proposals, name_to_id, self.current_W, self.current_H)
        labels = [d["label"] for d in proposals]
        print(f"🎉 Detection Complete: {len(proposals)} final detections")
        print(f"📋 Detected objects: {labels if labels else 'None'}")
        print(f"{'='*60}\n")
        return (len(yolo) > 0), yolo, (", ".join(labels) if labels else ", ".join(queries))
