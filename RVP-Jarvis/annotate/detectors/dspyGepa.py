# dspy_gepa_gemini_detection.py
from typing import List, Tuple, Optional
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

import dspy
from dspy.teleprompt import GEPA  # pip install dspy-ai gepa

# Optional: if you have an EDSR enhancer
try:
    from ..enhance import edsr_enhance
    _HAS_ENHANCE = True
except Exception:
    _HAS_ENHANCE = False

load_dotenv()

# Enhanced logging configuration
def setup_logger(name: str, level: str = None) -> logging.Logger:
    """Setup enhanced logger with proper formatting and handlers."""
    logger = logging.getLogger(name)
    
    # Set level from environment or default to INFO
    log_level = level or os.getenv('LOG_LEVEL', 'INFO')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter with more context
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler for debug logs
    log_dir = os.getenv('LOG_DIR', '/tmp')
    if os.path.exists(log_dir) and os.access(log_dir, os.W_OK):
        try:
            file_handler = logging.FileHandler(
                os.path.join(log_dir, 'dspy_gepa_debug.log')
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
    
    return logger

logger = setup_logger(__name__)


class Cell:
    def __init__(self, id: int, left: int, top: int, right: int, bottom: int):
        self.id = id
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class GeminiDSPy(dspy.LM):
    """
    A DSPy LM wrapper over Gemini that:
      - properly handles DSPy's compiled prompts with multimodal content
      - supports multiple images (current image + sample images)
      - exposes `last_usage` for cost tracking
      - integrates with GEPA optimization
    """
    def __init__(self, client: genai.Client, model_name: str, **kwargs):
        logger.info(f"Initializing GeminiDSPy with model: {model_name}")
        logger.debug(f"GeminiDSPy kwargs: {kwargs}")
        
        # Don't pass model name to avoid LiteLLM routing - use a custom provider name
        super().__init__(model=f"custom/{model_name}")
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
        self.last_usage = {}
        self.current_image = None  # Store current main image for multimodal calls
        self.current_images = []   # Store additional sample images
        
        logger.info(f"GeminiDSPy initialized successfully for model: {model_name}")

    def set_current_image(self, image):
        """Set the current main image for multimodal inference"""
        logger.debug(f"Setting current image: {type(image)} - {getattr(image, 'size', 'no size') if image else None}")
        self.current_image = image

    def set_current_images(self, images):
        """Set additional sample images for multimodal inference"""
        images = images or []
        logger.debug(f"Setting {len(images)} sample images")
        self.current_images = images
    
    def __call__(self, *args, **kwargs):
        prompt = kwargs.get("prompt") or kwargs.get("messages") or args[0] if args else ""
        # messages can be a list of dicts; just stringify them
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            prompt = "\n".join(m.get("content", "") for m in prompt)
        return self.basic_request(prompt, **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):
        if messages and isinstance(messages, list) and messages and isinstance(messages[0], dict):
            prompt = "\n".join(m.get("content", "") for m in messages)
        return self.basic_request(prompt or "", **kwargs)

    def basic_request(self, prompt, **kwargs):
        """
        Handle DSPy's compiled prompts and convert to multimodal Gemini format
        """
        import json as _json
        start_time = time.time()
        logger.debug(f"Starting basic_request with prompt type: {type(prompt)}")
        logger.debug(f"Request kwargs: {kwargs}")
        
        try:
            # Start with any sample images
            contents = list(self.current_images) if self.current_images else []
            logger.debug(f"Starting with {len(contents)} sample images")
            
            # Add the main image if available
            if self.current_image is not None:
                contents.append(self.current_image)
                logger.debug(f"Added main image, total contents: {len(contents)}")
            
            # Add the text prompt
            if isinstance(prompt, str):
                contents.append(prompt)
                logger.debug(f"Added string prompt, length: {len(prompt)}")
            elif isinstance(prompt, list):
                for i, p in enumerate(prompt):
                    # keep PIL.Image as-is; coerce everything else to str
                    if isinstance(p, Image.Image):
                        contents.append(p)
                        logger.debug(f"Added image prompt at index {i}")
                    else:
                        contents.append(str(p))
                        logger.debug(f"Added text prompt at index {i}, length: {len(str(p))}")
            else:
                contents.append(str(prompt))
                logger.debug(f"Added converted prompt, length: {len(str(prompt))}")

            # If no images, just use text
            if not contents or (len(contents) == 1 and isinstance(contents[0], str)):
                contents = prompt if isinstance(prompt, list) else [str(prompt)]
                logger.debug(f"Using text-only contents: {len(contents)} items")

            # Configure thinking budget
            thinking_budget = -1 if "pro" in self.model_name.lower() else 0
            logger.debug(f"Configured thinking budget: {thinking_budget} for model: {self.model_name}")
            
            # Enforce JSON so DSPy's JSONAdapter can parse it.
            schema = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "reasoning": types.Schema(type=types.Type.STRING),
                    "detections": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "label": types.Schema(type=types.Type.STRING),
                                "confidence": types.Schema(type=types.Type.NUMBER),
                                "box_2d": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.NUMBER)
                                ),
                            },
                            required=["label", "confidence", "box_2d"],
                        ),
                    ),
                },
                required=["detections"],
            )
            logger.debug("Created JSON schema for structured output")

            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 0.05),
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                response_mime_type="application/json",
                response_schema=schema,
                **self.kwargs
            )
            logger.debug(f"Created config with temperature: {config.temperature}, top_p: {config.top_p}")
            
            logger.info(f"Sending request to Gemini model: {self.model_name}")
            logger.debug(f"Request contents count: {len(contents)}")
            
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            
            request_time = time.time() - start_time
            logger.info(f"Gemini API request completed in {request_time:.2f}s")

            # Token usage for cost tracking
            self.last_usage = {}
            if getattr(resp, "usage_metadata", None):
                self.last_usage = {
                    "prompt_token_count": getattr(resp.usage_metadata, "prompt_token_count", 0),
                    "candidates_token_count": getattr(resp.usage_metadata, "candidates_token_count", 0),
                }
                logger.info(f"Token usage - Input: {self.last_usage['prompt_token_count']}, Output: {self.last_usage['candidates_token_count']}")
            else:
                logger.warning("No usage metadata available from Gemini response")

            raw_text = resp.text if hasattr(resp, "text") else str(resp)
            normalized = None

            def _try_json_load(s):
                try:
                    return _json.loads(s)
                except Exception:
                    return None

            # 1) If it’s valid JSON already
            obj = _try_json_load(raw_text)
            if obj is None:
                # 2) Try to extract JSON array/dict from prose (code fences, etc.)
                import re, ast
                m = re.search(r"```(?:json|python)?\s*(.*?)\s*```", raw_text, re.S)
                candidate = m.group(1) if m else raw_text
                # try as JSON
                obj = _try_json_load(candidate)
                if obj is None:
                    # try Python-ish -> JSON (True/False/None & single quotes)
                    candidate2 = re.sub(r"\bTrue\b", "true", candidate)
                    candidate2 = re.sub(r"\bFalse\b", "false", candidate2)
                    candidate2 = re.sub(r"\bNone\b", "null", candidate2)
                    candidate2 = re.sub(r"'([^']*)'", r'"\1"', candidate2)
                    obj = _try_json_load(candidate2)

            # 3) Wrap into the schema DSPy expects
            if obj is None:
                # give DSPy a valid object with required keys, but empty detections
                normalized = {"reasoning": "", "detections": []}
            elif isinstance(obj, list):
                normalized = {"reasoning": "", "detections": obj}
            elif isinstance(obj, dict):
                if "detections" in obj and "reasoning" not in obj:
                    obj["reasoning"] = ""
                if "detections" not in obj and "box_2d" in obj:
                    obj = {"reasoning": "", "detections": [obj]}
                if "detections" not in obj:
                    obj["detections"] = []
                if "reasoning" not in obj:
                    obj["reasoning"] = ""
                normalized = obj
            else:
                normalized = {"reasoning": "", "detections": []}

            # return JSON string so JSONAdapter sees both fields
            return [_json.dumps(normalized, ensure_ascii=False)]
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"GeminiDSPy basic_request failed after {total_time:.2f}s: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.debug(f"Full error details:", exc_info=True)
            return ["[]"]


class DetectionSignature(dspy.Signature):
    """Detect objects in an image based on queries."""
    context: str = dspy.InputField(desc="Problem statement, instructions, and queries")
    image = dspy.InputField(desc="PIL.Image input")
    reasoning: str = dspy.OutputField(desc="brief reasoning / steps")
    detections: str = dspy.OutputField(
        desc="JSON array of detections: "
             "[{'label': str, 'confidence': float, 'box_2d': [ymin, xmin, ymax, xmax]}] "
             "with coordinates normalized to 0..1000"
    )


class DetectionModule(dspy.Module):
    """
    Uses DSPy Chain-of-Thought with a multimodal LM.
    This module will be optimized by GEPA and use the learned prompts/demos.
    """
    def __init__(self, lm: GeminiDSPy, model_name: str):
        logger.info(f"Initializing DetectionModule with model: {model_name}")
        super().__init__()
        self.lm = lm
        self.model_name = model_name
        self.detect = dspy.ChainOfThought(DetectionSignature)
        self.sample_images = []  # Will be set by DSPyGepaGeminiDetection
        logger.debug("DetectionModule initialized with ChainOfThought")

    def forward(self, image: Image.Image, context: str):
        """
        Forward pass that uses DSPy's optimized prompts and demos.
        GEPA will optimize this flow and we'll use the learned prompts.
        """
        logger.debug(f"DetectionModule forward pass started")
        logger.debug(f"Image size: {image.size if image else 'None'}")
        logger.debug(f"Context length: {len(context)}")
        logger.debug(f"Sample images count: {len(self.sample_images)}")
        
        # Set the current main image and sample images in the LM
        self.lm.set_current_image(image)
        self.lm.set_current_images(self.sample_images)
        
        try:
            # Use DSPy's optimized flow with placeholder string for image
            # DSPy will use this placeholder while LM handles actual image
            logger.debug("Calling DSPy detect with placeholder image")
            result = self.detect(image="(image)", context=context)
            
            logger.debug(f"DetectionModule forward pass completed")
            logger.debug(f"Result type: {type(result)}")
            if hasattr(result, 'detections'):
                logger.debug(f"Detections length: {len(str(result.detections))}")
            
            return result
        finally:
            # Clear both image and sample images after use
            logger.debug("Clearing images from LM")
            self.lm.set_current_image(None)
            self.lm.set_current_images([])


class DSPyGepaGeminiDetection:
    """
    Unified DSPy + GEPA + Gemini detector with:
      - real multimodal flow through DSPy LM wrapper (so GEPA matters)
      - IoU-based GEPA metric
      - robust parsing + label normalization
      - deterministic token usage + INR cost logging
    """

    PRICING = {
        # Gemini 2.5
        'gemini-2.5-pro': {
            'tier_tokens': 200_000,
            'input': {'<=tier': 1.25, '>tier': 2.50},   # USD / 1M tokens
            'output': {'<=tier': 10.00, '>tier': 15.00}
        },
        'gemini-2.5-flash': {
            'input': 0.30,
            'output': 2.50
        },
        'gemini-2.5-flash-lite': {
            'input': 0.10,
            'output': 0.40
        },
    }

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
        cost_log_dir: Optional[str] = None
    ):
        logger.info(f"Initializing DSPyGepaGeminiDetection")
        logger.info(f"Model: {model_name}")
        logger.info(f"Object to detect: {object_to_detect}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"Image size: {image_size}")
        logger.info(f"Multiple predictions: {multiple_predictions}")
        logger.info(f"Upscale image: {upscale_image}")
        logger.debug(f"Problem statement length: {len(problem_statement)}")
        logger.debug(f"Sample images count: {len(sample_images or [])}")
        
        self.model_name = model_name
        self.object_to_detect = object_to_detect
        self.problem_statement = problem_statement
        self.sample_images = sample_images or []
        self.confidence_threshold = confidence_threshold
        self.multiple_predictions = multiple_predictions
        self.image_size = image_size
        self.upscale_image = upscale_image

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is not set")
            raise RuntimeError("GEMINI_API_KEY is not set.")
        
        logger.debug("Creating Gemini client with API key")
        self.client = genai.Client(api_key=api_key)
        logger.info("Gemini client created successfully")

        # Pricing map selection helper
        self._pricing_key = self._resolve_pricing_key(model_name)
        logger.debug(f"Resolved pricing key: {self._pricing_key}")

        # cost log path
        self.cost_log_dir = cost_log_dir or os.getenv("COST_LOG_DIR") \
            or "./data/raw_data"
        logger.debug(f"Cost log directory: {self.cost_log_dir}")

        # DSPy LM + settings - configure properly for full DSPy/GEPA usage
        logger.debug("Creating DSPy LM wrapper")
        self.dspy_lm = GeminiDSPy(self.client, self.model_name)
        
        logger.debug("Configuring DSPy settings")
        dspy.settings.configure(lm=self.dspy_lm, use_litellm=False)

        # DSPy module that will be optimized by GEPA
        logger.debug("Creating detection module")
        self.detection_module = DetectionModule(self.dspy_lm, self.model_name)
        self.detection_module.sample_images = self.sample_images  # Assign sample images
        self.optimized_module = None
        
        logger.info(f"DSPyGepaGeminiDetection initialization completed successfully")

    # ---------- Prompts ----------

    @staticmethod
    def get_user_prompt(object_of_interest: str, normalization_factor: int = 1000) -> str:
        return (
            f"Detect all prominent items corresponding to '{object_of_interest}'. "
            f"Return ONLY a JSON array of objects, each with: "
            f"label (one of the requested classes), confidence (0..1), and "
            f"box_2d = [ymin, xmin, ymax, xmax] normalized to 0..{normalization_factor}. "
            f"No prose. No extra keys."
        )

    # ---------- GEPA optimization ----------

    @staticmethod
    def create_training_example(image: Image.Image, context: str, detections_json: str) -> dspy.Example:
        """
        Helper to create a training example for GEPA optimization.
        
        Args:
            image: PIL Image (actual image, not placeholder)
            context: Context string with problem statement and queries
            detections_json: JSON string of ground truth detections
            
        Returns:
            dspy.Example for GEPA training
        """
        return dspy.Example(
            image=image,  # Pass actual PIL.Image for GEPA training
            context=context,
            detections=detections_json
        ).with_inputs("image", "context")

    def optimize_with_gepa(self, trainset: List[dspy.Example], metric_fn=None):
        """
        trainset items should follow DetectionSignature fields:
          dspy.Example(image=<PIL.Image>, context=<str>, detections=<str JSON>)
        """
        logger.info(f"Starting GEPA optimization with {len(trainset)} training examples")
        logger.debug(f"Using custom metric: {metric_fn is not None}")
        
        def iou(box1, box2):
            try:
                y1_min, x1_min, y1_max, x1_max = box1
                y2_min, x2_min, y2_max, x2_max = box2
                inter_xmin = max(x1_min, x2_min)
                inter_ymin = max(y1_min, y2_min)
                inter_xmax = min(x1_max, x2_max)
                inter_ymax = min(y1_max, y2_max)
                if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
                    return 0.0
                inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                union_area = box1_area + box2_area - inter_area
                return inter_area / union_area if union_area > 0 else 0.0
            except Exception:
                return 0.0

        def metric(example, pred, trace=None):
            try:
                gt = json.loads(example.detections)
                pd = json.loads(pred)
                
                logger.debug(f"Evaluating metric - GT: {len(gt)} items, Pred: {len(pd)} items")
                
                if not gt or not pd:
                    logger.debug("Empty ground truth or predictions")
                    return 0.0
                    
                total = 0.0
                matched = set()
                
                for i, p in enumerate(pd):
                    if 'label' not in p or 'box_2d' not in p:
                        logger.debug(f"Prediction {i} missing required fields")
                        continue
                        
                    best_iou = 0.0
                    best_idx = -1
                    
                    for j, g in enumerate(gt):
                        if j in matched:
                            continue
                        if 'label' not in g or 'box_2d' not in g:
                            continue
                        if str(p['label']).lower() != str(g['label']).lower():
                            continue
                            
                        v = iou(p['box_2d'], g['box_2d'])
                        if v > best_iou:
                            best_iou = v
                            best_idx = j
                    
                    if best_iou > 0.5:
                        total += best_iou
                        matched.add(best_idx)
                        logger.debug(f"Matched prediction {i} with GT {best_idx}, IoU: {best_iou:.3f}")
                
                final_score = total / max(len(gt), 1)
                logger.debug(f"Final metric score: {final_score:.3f} ({len(matched)}/{len(gt)} matched)")
                return final_score
                
            except Exception as e:
                logger.warning(f"Metric evaluation error: {e}")
                logger.debug(f"Error details:", exc_info=True)
                return 0.0

        chosen_metric = metric_fn or metric
        logger.debug(f"Selected metric function: {'custom' if metric_fn else 'default IoU-based'}")

        # GEPA optimization - DSPy is already configured globally
        logger.info("Starting GEPA compilation process")
        
        try:
            with dspy.settings.context(lm=self.dspy_lm, use_litellm=False):
                gepa = GEPA(
                    metric=chosen_metric,
                    max_bootstraps=5,
                    max_labeled_demos=4,
                    teacher_settings={"lm": self.dspy_lm},
                )
                logger.debug("GEPA optimizer created with max_bootstraps=5, max_labeled_demos=4")
                
                logger.info("Compiling detection module with GEPA (max_iters=10)")
                self.optimized_module = gepa.compile(
                    self.detection_module,
                    trainset=trainset,
                    max_iters=10,
                )
                
            # Ensure optimized module also has sample images
            if hasattr(self.optimized_module, 'sample_images'):
                self.optimized_module.sample_images = self.sample_images
                logger.debug(f"Assigned {len(self.sample_images)} sample images to optimized module")
            
            logger.info("GEPA optimization completed successfully")
            
        except Exception as e:
            logger.error(f"GEPA optimization failed: {e}")
            logger.debug(f"GEPA optimization error details:", exc_info=True)
            raise

    # ---------- Core detection ----------

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        start_time = time.time()
        logger.info(f"Starting detection process")
        logger.info(f"Image size: {image.size}")
        logger.info(f"Class names: {class_names}")
        logger.info(f"Class IDs: {class_ids}")
        
        try:
            queries = class_names if class_names else [self.object_to_detect]
            name_to_id = {n: cid for n, cid in zip(class_names, class_ids)}
            logger.debug(f"Queries: {queries}")
            logger.debug(f"Name to ID mapping: {name_to_id}")

            original_size = image.size
            im = image.copy()
            logger.debug(f"Original image size: {original_size}")

            if self.upscale_image and _HAS_ENHANCE:
                try:
                    logger.info("Upscaling image with EDSR...")
                    enhance_start = time.time()
                    im = edsr_enhance(im)
                    enhance_time = time.time() - enhance_start
                    logger.info(f"Image enhanced in {enhance_time:.2f}s, new size: {im.size}")
                except Exception as e:
                    logger.warning(f"Enhancement failed, continuing with original: {e}")
            elif self.upscale_image and not _HAS_ENHANCE:
                logger.warning("Image upscaling requested but EDSR enhancer not available")

            # Resize for cost/perf
            pre_resize_size = im.size
            im.thumbnail([self.image_size, self.image_size], Image.Resampling.LANCZOS)
            logger.debug(f"Image resized from {pre_resize_size} to {im.size}")

            # Prepare context (GEPA sees this verbatim)
            context_str = (
                f"Context: {self.problem_statement}\n"
                f"Objects: {', '.join(queries)}\n"
                f"Threshold: {self.confidence_threshold}\n"
                f"{self.get_user_prompt(', '.join(queries), normalization_factor=1000)}"
            )
            logger.debug(f"Context string length: {len(context_str)}")
            logger.debug(f"Context preview: {context_str[:300]}..." if len(context_str) > 300 else f"Full context: {context_str}")

            # Use DSPy with GEPA optimization - this is where the magic happens!
            # The optimized_module contains learned prompts and demos from GEPA
            module_to_use = self.optimized_module if self.optimized_module else self.detection_module
            logger.info(f"Using {'optimized' if self.optimized_module else 'base'} DSPy module for inference")
            
            # Add sample images to context if available
            enhanced_context = context_str
            if self.sample_images:
                enhanced_context = f"Sample images provided for reference.\n{context_str}"
                logger.debug(f"Enhanced context with {len(self.sample_images)} sample images")
            
            # Run the DSPy module - this will use optimized prompts and demos
            logger.info("Running DSPy detection module")
            inference_start = time.time()
            result = module_to_use(image=im, context=enhanced_context)
            inference_time = time.time() - inference_start
            logger.info(f"DSPy inference completed in {inference_time:.2f}s")
            
            # Extract the response text from DSPy result
            resp_text = result.detections if hasattr(result, 'detections') else str(result)
            logger.debug(f"Extracted response text length: {len(resp_text)}")
            logger.debug(f"Response preview: {resp_text[:200]}..." if len(resp_text) > 200 else f"Full response: {resp_text}")
            
            # Get token usage from the LM wrapper
            input_tokens = self.dspy_lm.last_usage.get("prompt_token_count", 0)
            output_tokens = self.dspy_lm.last_usage.get("candidates_token_count", 0)
            logger.debug(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            if input_tokens > 0 or output_tokens > 0:
                cost_inr = self._calculate_cost_in_inr(input_tokens, output_tokens)
                dataset_name = self._infer_dataset_name()
                logger.info(f"Calculated cost: ₹{cost_inr:.4f} for dataset: {dataset_name}")
                self._log_cost(cost_inr, dataset_name=dataset_name)
            else:
                logger.warning("No token usage information available for cost calculation")

            if not resp_text:
                logger.warning("Empty model response received")
                return False, [], "Empty model response."

            # Parse
            logger.debug("Parsing detection output")
            parse_start = time.time()
            raw = self._parse_detection_output(resp_text)
            parse_time = time.time() - parse_start
            logger.debug(f"Parsing completed in {parse_time:.3f}s")
            
            if not isinstance(raw, list):
                if isinstance(raw, dict) and "box_2d" in raw:
                    raw = [raw]
                    logger.debug("Converted single detection dict to list")
                else:
                    logger.error(f"Could not parse detections as list, got type: {type(raw)}")
                    return False, [], "Could not parse detections as list."
            
            if not raw:
                logger.info("No detections found in parsed output")
                return False, [], "No detections found."
            
            logger.info(f"Parsed {len(raw)} raw detections")

            # Post-process detections
            logger.debug("Starting post-processing of detections")
            converted_bounding_boxes: List[Cell] = []
            labels: List[str] = []
            skipped = 0

            # For flexible label normalization
            def _match_label(lbl: str) -> Optional[str]:
                lbl_l = (lbl or "").strip().lower()
                for cname in name_to_id.keys():
                    cn_l = cname.lower()
                    if lbl_l == cn_l or lbl_l in cn_l or cn_l in lbl_l:
                        return cname
                return None

            for i, it in enumerate(raw):
                logger.debug(f"Processing detection {i}: {it}")
                
                if not isinstance(it, dict):
                    logger.debug(f"Detection {i} is not a dict, skipping")
                    skipped += 1
                    continue
                    
                label = _match_label(it.get("label", ""))
                if label is None:
                    logger.debug(f"Detection {i} has no matching label for '{it.get('label', '')}'")
                    skipped += 1
                    continue
                    
                # confidence
                try:
                    conf = float(it.get("confidence", 1.0))
                except Exception:
                    logger.debug(f"Detection {i} has invalid confidence, using 1.0")
                    conf = 1.0
                    
                if conf < self.confidence_threshold:
                    logger.debug(f"Detection {i} confidence {conf:.3f} below threshold {self.confidence_threshold}")
                    continue
                    
                # box
                box = it.get("box_2d")
                if not (isinstance(box, list) and len(box) == 4):
                    logger.debug(f"Detection {i} has invalid box format: {box}")
                    continue
                    
                abs_y1 = int(box[0] / 1000 * original_size[1])
                abs_x1 = int(box[1] / 1000 * original_size[0])
                abs_y2 = int(box[2] / 1000 * original_size[1])
                abs_x2 = int(box[3] / 1000 * original_size[0])
                
                if abs_y1 >= abs_y2 or abs_x1 >= abs_x2:
                    logger.debug(f"Detection {i} has invalid box coordinates: ({abs_x1}, {abs_y1}, {abs_x2}, {abs_y2})")
                    continue
                    
                logger.debug(f"Detection {i} accepted: label='{label}', conf={conf:.3f}, box=({abs_x1}, {abs_y1}, {abs_x2}, {abs_y2})")
                cell = Cell(id=i, left=abs_x1, top=abs_y1, right=abs_x2, bottom=abs_y2)
                converted_bounding_boxes.append(cell)
                labels.append(label)

            logger.info(f"Post-processing complete: {len(converted_bounding_boxes)} valid detections, {skipped} skipped")
            
            if not self.multiple_predictions and converted_bounding_boxes:
                logger.debug("Multiple predictions disabled, keeping only first detection")
                converted_bounding_boxes = [converted_bounding_boxes[0]]
                labels = [labels[0]]

            # Convert to YOLO format
            logger.debug("Converting detections to YOLO format")
            yolo = []
            for cell in converted_bounding_boxes:
                cid = name_to_id.get(labels[cell.id], 0)
                x0, y0, x1, y1 = cell.left, cell.top, cell.right, cell.bottom
                yolo_box = self._bbox_to_yolo(x0, y0, x1, y1, original_size[0], original_size[1], cid)
                yolo.append(yolo_box)
                logger.debug(f"YOLO box: class_id={cid}, center=({yolo_box[1]:.3f}, {yolo_box[2]:.3f}), size=({yolo_box[3]:.3f}, {yolo_box[4]:.3f})")

            success = len(yolo) > 0
            result_labels = ", ".join(labels) if labels else ", ".join(queries)
            total_time = time.time() - start_time
            
            logger.info(f"Detection completed in {total_time:.2f}s")
            logger.info(f"Result: success={success}, {len(yolo)} detections, labels='{result_labels}'")
            
            return success, yolo, result_labels

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Detector failed after {total_time:.2f}s: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.debug(f"Full detector error details:", exc_info=True)
            return False, [], f"Error: {e}"

    # ---------- Helpers ----------

    @staticmethod
    def _bbox_to_yolo(x0, y0, x1, y1, W, H, class_id):
        xc = (x0 + x1) / 2 / W
        yc = (y0 + y1) / 2 / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H
        return [class_id, xc, yc, w, h]

    def _resolve_pricing_key(self, model_name: str) -> str:
        logger.debug(f"Resolving pricing key for model: {model_name}")
        m = model_name.lower()
        for k in self.PRICING.keys():
            if k in m:
                logger.debug(f"Found matching pricing key: {k}")
                return k
        # fallback
        logger.debug(f"No matching pricing key found, using fallback: gemini-2.5-flash")
        return 'gemini-2.5-flash'

    def _usd_to_inr(self) -> float:
        logger.debug("Fetching USD to INR exchange rate")
        try:
            r = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=10)
            if r.status_code != 200:
                logger.warning(f"Exchange rate API returned status {r.status_code}")
                return 83.0
            
            data = r.json()
            rate = float(data['rates']['INR'])
            logger.debug(f"Fetched exchange rate: 1 USD = {rate} INR")
            return rate
        except requests.exceptions.Timeout:
            logger.warning("Exchange rate API request timed out; using fallback rate 83.0")
            return 83.0
        except Exception as e:
            logger.warning(f"FX fetch failed: {e}; using fallback rate 83.0")
            logger.debug(f"Exchange rate fetch error details:", exc_info=True)
            return 83.0

    def _calculate_cost_in_inr(self, input_tokens: int, output_tokens: int) -> float:
        logger.debug(f"Calculating cost for {input_tokens} input tokens, {output_tokens} output tokens")
        try:
            pricing = self.PRICING[self._pricing_key]
            logger.debug(f"Using pricing model: {self._pricing_key}")
            
            if 'tier_tokens' in pricing:
                tier = pricing['tier_tokens']
                in_key = '<=tier' if input_tokens <= tier else '>tier'
                out_key = '<=tier' if input_tokens <= tier else '>tier'
                input_rate = pricing['input'][in_key]
                output_rate = pricing['output'][out_key]
                logger.debug(f"Tiered pricing: input_rate={input_rate} ({in_key}), output_rate={output_rate} ({out_key})")
                input_cost_usd = (input_tokens / 1_000_000) * input_rate
                output_cost_usd = (output_tokens / 1_000_000) * output_rate
            else:
                input_rate = pricing['input']
                output_rate = pricing['output']
                logger.debug(f"Flat pricing: input_rate={input_rate}, output_rate={output_rate}")
                input_cost_usd = (input_tokens / 1_000_000) * input_rate
                output_cost_usd = (output_tokens / 1_000_000) * output_rate
            
            total_usd = input_cost_usd + output_cost_usd
            exchange_rate = self._usd_to_inr()
            total_inr = total_usd * exchange_rate
            
            logger.debug(f"Cost breakdown: input=${input_cost_usd:.6f}, output=${output_cost_usd:.6f}, total=${total_usd:.6f}")
            logger.debug(f"Total cost in INR: ₹{total_inr:.4f} (rate: {exchange_rate})")
            
            return total_inr
        except Exception as e:
            logger.error(f"Cost calculation error: {e}")
            logger.debug(f"Cost calculation error details:", exc_info=True)
            return 0.0

    def _infer_dataset_name(self) -> str:
        logger.debug("Inferring dataset name from current context")
        try:
            cwd = os.getcwd()
            logger.debug(f"Current working directory: {cwd}")
            
            if "raw_data" in cwd:
                dataset_name = os.path.basename(cwd)
                logger.debug(f"Inferred dataset name from cwd: {dataset_name}")
                return dataset_name
                
            if self.object_to_detect:
                dataset_name = f"{self.object_to_detect}_detection"
                logger.debug(f"Inferred dataset name from object: {dataset_name}")
                return dataset_name
        except Exception as e:
            logger.debug(f"Error inferring dataset name: {e}")
            
        logger.debug("Using default dataset name: annotation_job")
        return "annotation_job"

    def _log_cost(self, cost_inr: float, dataset_name: str = "unknown"):
        logger.debug(f"Logging cost: ₹{cost_inr:.4f} for dataset: {dataset_name}")
        try:
            os.makedirs(self.cost_log_dir, exist_ok=True)
            cost_file_path = os.path.join(self.cost_log_dir, "cost.txt")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = f"{timestamp} | {dataset_name} | Gemini API ({self.model_name}) | ₹{cost_inr:.4f}\n"
            
            with open(cost_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            logger.info(f"Cost logged to {cost_file_path}: ₹{cost_inr:.4f}")
            logger.debug(f"Cost log entry: {log_entry.strip()}")
            
        except PermissionError:
            logger.error(f"Permission denied writing to cost log: {cost_file_path}")
        except OSError as e:
            logger.error(f"OS error writing cost log: {e}")
        except Exception as e:
            logger.error(f"Unexpected error logging cost: {e}")
            logger.debug(f"Cost logging error details:", exc_info=True)

    # ---- Parsing utilities (robust to messy LLM output) ----

    def _parse_detection_output(self, output_text):
        logger.debug(f"Parsing detection output, length: {len(output_text) if output_text else 0}")
        
        if not output_text:
            logger.debug("Empty output text, returning None")
            return None
            
        cleaned = output_text.strip()
        logger.debug(f"Stripped output length: {len(cleaned)}")
        
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1]
            logger.debug("Removed surrounding quotes from output")
        
        cleaned = self._fix_duplicate_fields(cleaned)
        logger.debug("Fixed duplicate fields in output")
        
        code_fence_match = re.search(r"```(?:json|python)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
        if code_fence_match:
            fence = code_fence_match.group(1).strip()
            logger.debug(f"Found code fence, content length: {len(fence)}")
            r = self._try_parse_json_or_python(fence)
            if r is not None:
                logger.debug(f"Successfully parsed code fence content")
                return self._validate_and_fix_detections(r)
            else:
                logger.debug("Failed to parse code fence content")

        cleaned = re.sub(r"^```.*?\n|```$", "", cleaned, flags=re.S)
        cleaned = re.sub(r"//.*", "", cleaned)
        cleaned = re.sub(r"\(\s*(\d+)\s*\)", r"[\1]", cleaned)
        logger.debug("Applied regex cleaning to output")

        candidates = self._extract_all_structures(cleaned)
        logger.debug(f"Extracted {len(candidates)} structure candidates")
        
        dicts = [c for c in candidates if c["text"].strip().startswith("{")]
        lists = [c for c in candidates if c["text"].strip().startswith("[")]
        logger.debug(f"Found {len(dicts)} dict candidates and {len(lists)} list candidates")
        
        for i, c in enumerate(dicts + lists):
            logger.debug(f"Trying to parse candidate {i}: {c['text'][:100]}...")
            r = self._try_parse_json_or_python(c["text"])
            if r is not None:
                logger.debug(f"Successfully parsed candidate {i}")
                return self._validate_and_fix_detections(r)
            else:
                logger.debug(f"Failed to parse candidate {i}")
                
        logger.debug("All parsing attempts failed, returning None")
        return None

    @staticmethod
    def _fix_duplicate_fields(text):
        pattern = r'("box_2d":\s*\[[^\]]+\]),([^}]*),\s*("box_2d":\s*\[[^\]]+\])'
        def repl(m):
            first_box = m.group(1)
            middle = m.group(2)
            # drop the duplicate trailing box_2d
            return f"{first_box},{middle}"
        return re.sub(pattern, repl, text)
    
    @staticmethod
    def _validate_and_fix_detections(dets):
        logger.debug(f"Validating detections, input type: {type(dets)}")
        
        if not isinstance(dets, list):
            logger.debug("Input is not a list, returning as-is")
            return dets
            
        logger.debug(f"Validating {len(dets)} detection items")
        out = []
        skipped = 0
        
        for i, d in enumerate(dets):
            if not isinstance(d, dict):
                logger.debug(f"Detection {i} is not a dict, skipping")
                skipped += 1
                continue
                
            if "box_2d" not in d:
                logger.debug(f"Detection {i} missing box_2d field, skipping")
                skipped += 1
                continue
                
            try:
                box = [float(x) for x in d["box_2d"]]
            except Exception as e:
                logger.debug(f"Detection {i} has invalid box_2d values: {e}")
                skipped += 1
                continue
                
            if len(box) != 4:
                logger.debug(f"Detection {i} box_2d has {len(box)} values, expected 4")
                skipped += 1
                continue
                
            validated_detection = {
                "box_2d": box,
                "label": d.get("label", "object"),
                "confidence": float(d.get("confidence", 0.5))
            }
            out.append(validated_detection)
            logger.debug(f"Detection {i} validated: {validated_detection}")
        
        logger.debug(f"Validation complete: {len(out)} valid detections, {skipped} skipped")
        return out

    @staticmethod
    def _try_parse_json_or_python(text):
        if not text:
            logger.debug("Empty text provided to parser")
            return None
            
        logger.debug(f"Trying to parse text of length {len(text)}")
        
        # Try JSON first
        try:
            result = json.loads(text)
            logger.debug("Successfully parsed as JSON")
            return result
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        # Try Python literal_eval
        try:
            obj = ast.literal_eval(text)
            result = json.loads(json.dumps(obj))
            logger.debug("Successfully parsed as Python literal and converted to JSON")
            return result
        except Exception as e:
            logger.debug(f"Python literal_eval failed: {e}")
        
        # Try converting Python-like syntax to JSON
        converted = DSPyGepaGeminiDetection._convert_python_to_json(text)
        if converted != text:
            logger.debug("Attempting to parse converted Python-to-JSON text")
            try:
                result = json.loads(converted)
                logger.debug("Successfully parsed converted Python-to-JSON text")
                return result
            except Exception as e:
                logger.debug(f"Converted text parsing failed: {e}")
        
        logger.debug("All parsing attempts failed")
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
        cands = []
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            if text[i] in "{[":
                start = i
                end = DSPyGepaGeminiDetection._find_matching_bracket(text, i)
                if end is not None:
                    cands.append({"text": text[start:end+1], "start": start, "end": end})
                    i = end + 1
                else:
                    i += 1
            else:
                i += 1
        return cands

    @staticmethod
    def _find_matching_bracket(text, start_idx):
        if start_idx >= len(text):
            return None
        open_ch = text[start_idx]
        close_ch = "}" if open_ch == "{" else ("]" if open_ch == "[" else None)
        if not close_ch:
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
