from typing import List, Tuple
from PIL import Image
import os
import json
import time
import logging
import re
import ast
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv
from ..enhance import edsr_enhance

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiDetection:
    """
    Gemini-based object detection with automatic cost tracking.
    
    This class automatically tracks and logs the cost of each Gemini API call
    in Indian Rupees (INR) to the cost.txt file in the raw_data directory.
    
    Cost tracking features:
    - Extracts token usage from API responses
    - Calculates costs based on current Gemini pricing
    - Converts USD to INR using live exchange rates
    - Logs costs with timestamps to cost.txt
    """
    class Cell:
        def __init__(self, id: int, left: int, top: int, right: int, bottom: int):
            self.id = id
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom

    def __init__(self, model_name: str, object_to_detect: str,
                 problem_statement: str = "", sample_images: List[Image.Image] = None,
                 confidence_threshold: float = 0.8, multiple_predictions: bool = True, image_size=1024, upscale_image=False, golden_examples: List[Tuple[Image.Image, List[List[float]]]] = None):
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
        
        self.pricing = {
            'gemini-2.5-pro': {
                'input': {'<=200k': 1.25, '>200k': 2.50},
                'output': {'<=200k': 10.00, '>200k': 15.00}
            },
            'gemini-2.5-flash': {
                'input': 0.30,    # applies to text, image, video input
                'output': 2.50    # text output
            },
            'gemini-2.5-flash-lite': {
                'input': 0.10,
                'output': 0.40
            }
        }

    @staticmethod
    def get_system_prompt(normalization_factor: int) -> str:
        """Get the system prompt for Gemini object detection."""
        return ""

    @staticmethod
    def get_user_prompt(object_of_interest: str, normalization_factor: int) -> str:
        """Get the user prompt for Gemini object detection."""
        return f"""Detect all of the prominent items in the image that corresponds to {object_of_interest}. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-{normalization_factor}."""

    @staticmethod
    def normalize_label_for_matching(label):
        """Normalize labels for better matching by removing common variations."""
        if not label:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = label.lower().strip()
        
        # Remove common punctuation and special characters
        normalized = re.sub(r'[_\-\.\,\;\:\!\?\(\)\[\]{}]', ' ', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common words that might cause mismatches
        stop_words = ['the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with']
        words = normalized.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words).strip()

    @staticmethod
    def parse_detection_output(output_text):
        """Forgiving parse of dicts/lists, handles outer quotes, comments & code-fences."""
        if not output_text:
            return None
        cleaned = output_text.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1]
        
        # Clean up duplicate box_2d fields that Gemini sometimes generates
        cleaned = GeminiDetection._fix_duplicate_fields(cleaned)
        
        code_fence_match = re.search(
            r"```(?:json|python)?\s*\n(.*?)\n```", cleaned, re.DOTALL
        )
        if code_fence_match:
            fence_content = code_fence_match.group(1).strip()
            result = GeminiDetection._try_parse_json_or_python(fence_content)
            if result is not None:
                return GeminiDetection._validate_and_fix_detections(result)
        cleaned = re.sub(r"^```.*?\n|```$", "", cleaned, flags=re.S)
        cleaned = re.sub(r"//.*", "", cleaned)
        cleaned = re.sub(r"\(\s*(\d+)\s*\)", r"[\1]", cleaned)
        candidates = GeminiDetection._extract_all_structures(cleaned)
        dicts = [c for c in candidates if c["text"].strip().startswith("{")]
        lists = [c for c in candidates if c["text"].strip().startswith("[")]
        for candidate in dicts:
            result = GeminiDetection._try_parse_json_or_python(candidate["text"])
            if result is not None:
                return GeminiDetection._validate_and_fix_detections(result)
        for candidate in lists:
            result = GeminiDetection._try_parse_json_or_python(candidate["text"])
            if result is not None:
                return GeminiDetection._validate_and_fix_detections(result)
        return None

    @staticmethod
    def _fix_duplicate_fields(text):
        """Fix duplicate box_2d fields that Gemini sometimes generates."""
        import re
        # Pattern to match duplicate box_2d fields in JSON objects
        # Matches: "box_2d": [values], "other": "stuff", "box_2d": [values]
        pattern = r'("box_2d":\s*\[[^\]]+\]),([^}]*),\s*("box_2d":\s*\[[^\]]+\])'
        
        def replace_duplicate(match):
            first_box = match.group(1)
            middle_content = match.group(2)
            second_box = match.group(3)
            # Keep the first box_2d and remove the duplicate
            return f'{first_box},{middle_content}'
        
        return re.sub(pattern, replace_duplicate, text)
    
    @staticmethod
    def _validate_and_fix_detections(detections):
        """Validate and fix detection objects to ensure consistent format."""
        if not isinstance(detections, list):
            return detections
        
        fixed_detections = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
                
            # Ensure required fields exist
            if "box_2d" not in detection:
                continue
                
            # Add missing fields with defaults
            fixed_detection = {
                "box_2d": detection["box_2d"],
                "label": detection.get("label", "object"),  # Default label
                "confidence": detection.get("confidence", 0.5)  # Default confidence
            }
            
            # Validate box_2d format
            box_2d = fixed_detection["box_2d"]
            if isinstance(box_2d, list) and len(box_2d) == 4:
                # Ensure all coordinates are numbers
                try:
                    fixed_detection["box_2d"] = [float(coord) for coord in box_2d]
                    fixed_detections.append(fixed_detection)
                except (ValueError, TypeError):
                    continue  # Skip invalid box coordinates
        
        return fixed_detections

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
        converted = GeminiDetection._convert_python_to_json(text)
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
                end_pos = GeminiDetection._find_matching_bracket(text, i)
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
    def _bbox_to_yolo(x0, y0, x1, y1, W, H, class_id, conf=1.0):
        xc = (x0 + x1) / 2 / W
        yc = (y0 + y1) / 2 / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H
        return [class_id, xc, yc, w, h, conf]
    
    def get_usd_to_inr_exchange_rate(self):
        """Fetch current USD to INR exchange rate."""
        try:
            response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=10)
            data = response.json()
            return data['rates']['INR']
        except Exception as e:
            logger.warning(f"Failed to fetch exchange rate: {e}. Using fallback rate of 83.0")
            return 83.0  # Fallback exchange rate
    
    def calculate_gemini_cost(self, input_tokens, output_tokens):
        """Calculate the cost of Gemini API call in INR."""
        try:
            model_key = None
            # Find matching model in pricing
            for key in self.pricing.keys():
                if key in self.model_name.lower():
                    model_key = key
                    break
            
            if not model_key:
                # Default to flash pricing for unknown models
                model_key = 'gemini-1.5-flash'
                logger.warning(f"Unknown model {self.model_name}, using default flash pricing")
            
            pricing_info = self.pricing[model_key]
            
            # Calculate cost in USD
            if isinstance(pricing_info['input'], dict):
                # Pro model with tiered pricing
                input_tier = '<=128k' if input_tokens <= 128000 else '>128k'
                output_tier = '<=128k' if input_tokens <= 128000 else '>128k'
                input_cost_usd = (input_tokens / 1_000_000) * pricing_info['input'][input_tier]
                output_cost_usd = (output_tokens / 1_000_000) * pricing_info['output'][output_tier]
            else:
                # Flash model with flat pricing
                input_cost_usd = (input_tokens / 1_000_000) * pricing_info['input']
                output_cost_usd = (output_tokens / 1_000_000) * pricing_info['output']
            
            total_cost_usd = input_cost_usd + output_cost_usd
            
            # Convert USD to INR
            exchange_rate = self.get_usd_to_inr_exchange_rate()
            total_cost_inr = total_cost_usd * exchange_rate
            
            logger.info(f"Gemini API cost: {total_cost_usd:.6f} USD = {total_cost_inr:.4f} INR (tokens: {input_tokens}+{output_tokens})")
            return total_cost_inr
            
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0
    
    def log_cost_to_file(self, cost_inr, dataset_name="unknown"):
        """Log the cost to cost.txt file in the raw_data directory."""
        try:
            # Find the appropriate raw_data directory
            raw_data_path = "./data/raw_data"
            cost_file_path = os.path.join(raw_data_path, "cost.txt")
            
            # Create timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Append cost information
            with open(cost_file_path, 'a', encoding='utf-8') as file:
                file.write(f"{timestamp} | {dataset_name} | Gemini API ({self.model_name}) | ₹{cost_inr:.4f}\n")
            
            logger.info(f"Cost logged to {cost_file_path}: ₹{cost_inr:.4f}")
            
        except Exception as e:
            logger.error(f"Error logging cost to file: {e}")

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        try:
            # Support multi-class: use class_names as queries (fallback to object_to_detect if empty)
            queries = class_names if class_names else [self.object_to_detect]
            name_to_id = {n: cid for n, cid in zip(class_names, class_ids)}

            # Downscale to ~512 max side for cost/perf; keep original for rescale
            original_size = image.size
            im = image.copy()

            if self.upscale_image:
                # Try to enhance the image, fallback to original if enhancement fails
                try:
                    print("Upscaling image...")
                    im = edsr_enhance(im)
                except Exception as e:
                    print(f"Warning: Image enhancement failed, using original image: {e}")
                    print("Using original image...")
                    # Continue with the original image
            
            im.thumbnail([self.image_size, self.image_size], Image.Resampling.LANCZOS)

            # Prepare context with problem statement and sample images
            context = []
            if self.problem_statement:
                context.append(f"Context: {self.problem_statement}")
            context.extend(self.sample_images)
            
            # Add golden examples if available
            for golden_img, golden_anns in self.golden_examples:
                context.append(golden_img)
                golden_str = ", ".join([f"{class_names[int(ann[0])]} at [{ann[1]:.2f},{ann[2]:.2f},{ann[3]:.2f},{ann[4]:.2f}] (conf {ann[5]:.2f})" for ann in golden_anns])
                context.append(f"Example: {golden_str}")

            # Generate prompt with reasoning
            base_prompt = self.get_user_prompt(
                object_of_interest=", ".join(queries),
                normalization_factor=1000
            )
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
     "box_2d": [y_min, x_min, y_max, x_max]  # normalized 0..1000
   }}
Rules:
- Ensure y_min < y_max and x_min < x_max.
- Include detections with confidence >= {self.confidence_threshold}.
- Order by confidence descending.
- Return [] if no detections meet criteria.
"""

            contents = [im, *context, prompt]

            # Determine thinking budget based on model type
            # Set to -1 for pro models (unlimited thinking), 0 for others (no thinking)
            thinking_budget = -1 if "pro" in self.model_name.lower() else 0
            
            # Retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
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
                    
                    # Extract token usage and calculate cost
                    try:
                        if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
                            input_tokens = getattr(resp.usage_metadata, 'prompt_token_count', 0)
                            output_tokens = getattr(resp.usage_metadata, 'candidates_token_count', 0)
                            
                            if input_tokens > 0 or output_tokens > 0:
                                cost_inr = self.calculate_gemini_cost(input_tokens, output_tokens)
                                
                                # Try to determine dataset name from current working directory or context
                                dataset_name = "annotation_job"
                                try:
                                    cwd = os.getcwd()
                                    if "raw_data" in cwd:
                                        dataset_name = os.path.basename(cwd)
                                    elif self.object_to_detect:
                                        dataset_name = f"{self.object_to_detect}_detection"
                                except:
                                    pass
                                
                                self.log_cost_to_file(cost_inr, dataset_name)
                            else:
                                logger.warning("No token usage information available from Gemini response")
                        else:
                            logger.warning("No usage metadata available from Gemini response")
                    except Exception as cost_error:
                        logger.error(f"Error in cost tracking: {cost_error}")
                    
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error("Failed after %d retries: %s", max_retries, e)
                        return False, [], f"API error after retries: {e}"
                    time.sleep(1)

            if not getattr(resp, "text", ""):
                logger.warning("Gemini empty response")
                return False, [], ""

            # Debug raw response
            print("----------------Gemini Detection LOGGING REASONING ----------------")
            print(resp.text)
            print("----------------Gemini Detection LOGGING REASONING ----------------")

            # Parse response
            try:
                raw = self.parse_detection_output(resp.text)
                if not isinstance(raw, list):
                    logger.error("Expected list, got %s. Attempting to extract detections anyway.", type(raw))
                    # If we got a single dict, wrap it in a list
                    if isinstance(raw, dict) and "box_2d" in raw:
                        raw = [raw]
                    else:
                        return False, [], "Could not parse detections as list"
                
                if not raw:  # Empty list
                    logger.warning("No detections found in response")
                    return False, [], "No detections found"
                    
            except Exception as e:
                logger.error("Parse error: %s; text=%s", e, resp.text[:500])  # Limit logged text
                return False, [], f"Parse error: {e}"

            # Process detections into Cell objects
            converted_bounding_boxes = []
            labels = []
            skipped_count = 0

            logger.info(f"Processing {len(raw)} raw detections")

            for i, it in enumerate(raw):
                if not isinstance(it, dict):
                    skipped_count += 1
                    continue
                    
                label = it.get("label", "")
                # Flexible label matching - try exact match first, then fuzzy matching
                matched_label = None
                if label in name_to_id:
                    matched_label = label
                else:
                    # Try case-insensitive and normalized matching
                    label_normalized = label.lower().strip()
                    for class_name in name_to_id.keys():
                        class_name_normalized = class_name.lower().strip()
                        if label_normalized == class_name_normalized:
                            matched_label = class_name
                            break
                    
                    # If still no match, try deep normalization matching
                    if not matched_label:
                        label_deep_normalized = self.normalize_label_for_matching(label)
                        for class_name in name_to_id.keys():
                            class_name_deep_normalized = self.normalize_label_for_matching(class_name)
                            if label_deep_normalized == class_name_deep_normalized:
                                matched_label = class_name
                                break
                    
                    # If still no match, try partial matching (contains)
                    if not matched_label:
                        for class_name in name_to_id.keys():
                            class_name_normalized = class_name.lower().strip()
                            if (label_normalized in class_name_normalized or 
                                class_name_normalized in label_normalized):
                                matched_label = class_name
                                break
                
                if not matched_label:
                    # Provide detailed debug information for unmatched labels
                    debug_info = []
                    debug_info.append(f"Original: '{label}'")
                    debug_info.append(f"Normalized: '{label.lower().strip()}'")
                    debug_info.append(f"Deep normalized: '{self.normalize_label_for_matching(label)}'")
                    debug_info.append(f"Available classes: {list(name_to_id.keys())}")
                    print(f"⚠️  Label matching failed - {' | '.join(debug_info)}")
                    skipped_count += 1
                    continue
                conf = it.get("confidence", 1.0)
                try:
                    conf = float(conf)
                except:
                    conf = 1.0
                if conf < self.confidence_threshold:
                    continue
                box = it.get("box_2d")
                if not (isinstance(box, list) and len(box) == 4):
                    continue

                # Convert normalized (0..1000) to absolute coordinates
                abs_y1 = int(box[0] / 1000 * original_size[1])
                abs_x1 = int(box[1] / 1000 * original_size[0])
                abs_y2 = int(box[2] / 1000 * original_size[1])
                abs_x2 = int(box[3] / 1000 * original_size[0])
                if abs_y1 >= abs_y2 or abs_x1 >= abs_x2:
                    continue

                cell = self.Cell(id=i, left=abs_x1, top=abs_y1, right=abs_x2, bottom=abs_y2)
                converted_bounding_boxes.append(cell)
                labels.append(matched_label)

            # Apply multiple_predictions logic
            if not self.multiple_predictions and converted_bounding_boxes:
                converted_bounding_boxes = [converted_bounding_boxes[0]]
                labels = [labels[0]]

            # Convert Cell objects to YOLO format
            yolo = []
            for cell in converted_bounding_boxes:
                cid = name_to_id.get(labels[cell.id], 0)
                x0, y0, x1, y1 = cell.left, cell.top, cell.right, cell.bottom
                # Extract confidence from raw response
                conf = raw[cell.id].get("confidence", 1.0) if isinstance(raw, list) and cell.id < len(raw) else 1.0
                yolo.append(self._bbox_to_yolo(x0, y0, x1, y1, original_size[0], original_size[1], cid, conf))

            logger.info(f"Successfully processed {len(yolo)} detections, skipped {skipped_count}")
            return (len(yolo) > 0), yolo, (", ".join(labels) if labels else ", ".join(queries))

        except Exception as e:
            logger.exception("Gemini detector error")
            return False, [], f"Error processing detection: {e}"