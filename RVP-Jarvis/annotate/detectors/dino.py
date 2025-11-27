# backends/grounding_dino_hf.py
from typing import List, Tuple, Dict, Union
from PIL import Image
import logging, time
from ..detector import BaseDetector
from ..detector import get_device

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from torchvision.ops import nms
    from collections import defaultdict
except Exception as _e:
    AutoProcessor = AutoModelForZeroShotObjectDetection = None
    _import_error = _e

class GroundingDinoHFDetection(BaseDetector):
    """
    Grounding DINO loaded purely from Hugging Face via transformers.
    Example model ids: 
      - 'IDEA-Research/grounding-dino-tiny'
      - 'IDEA-Research/grounding-dino-base'
    """
    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        object_to_detect: str = "",
        problem_statement: str = "",
        sample_images: List[Image.Image] = None,
        confidence_threshold: Union[float, Dict[str, float]] = 0.45,
        text_threshold: float = 0.3,
        nms_iou: Union[float, Dict[str, float]] = 0.5,
        device: str = None,
    ):
        if AutoProcessor is None:
            logger.exception("Transformers/torch import failed", exc_info=_import_error)
            raise ImportError("Install transformers, torch, accelerate, torchvision")

        self.model_name = model_name
        self.object_to_detect = object_to_detect
        self.problem_statement = problem_statement
        self.sample_images = sample_images or []
        self.confidence_threshold = confidence_threshold
        self.text_threshold = float(text_threshold)
        self.nms_iou = nms_iou
        device_str = get_device(device)
        self.device = torch.device(device_str)

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device).eval()
        
        # Half precision on GPU - disabled due to dtype compatibility issues
        self.use_half_precision = False
        # if self.device.type == "cuda":
        #     self.model.half()
        #     self.use_half_precision = True
        
        # cuDNN benchmark for speed on repeated shapes
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        logger.info("GroundingDINO(HF) loaded in %.2fs (model=%s, device=%s)",
                    time.time()-t0, self.model_name, self.device)

    @staticmethod
    def _bbox_to_yolo(x0, y0, x1, y1, W, H, cid, conf=None):
        xc = (x0 + x1) / 2 / W
        yc = (y0 + y1) / 2 / H
        w = (x1 - x0) / W
        h = (y1 - y0) / H
        out = [cid, xc, yc, w, h]
        return out if conf is None else out + [float(conf)]

    @staticmethod
    def _clip_xyxy(x0, y0, x1, y1, W, H):
        x0 = max(0.0, min(W, float(x0)))
        y0 = max(0.0, min(H, float(y0)))
        x1 = max(0.0, min(W, float(x1)))
        y1 = max(0.0, min(H, float(y1)))
        return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        try:
            queries = [q.strip() for q in (class_names if class_names else [self.object_to_detect]) if q.strip()]
            if not queries:
                logger.warning("GroundingDINO(HF): empty queries"); return False, [], "No queries provided"
            name_to_id: Dict[str, int] = {q: cid for q, cid in zip(queries, class_ids)}

            im = image.convert("RGB")
            W, H = im.size
            # Transformers expects a list of lists for zero-shot detection
            text_labels = [queries]

            t0 = time.time()
            inputs = self.processor(images=im, text=text_labels, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
            logger.info("GroundingDINO(HF) forward %.2fs", time.time() - t0)

            # Post-process with compatible parameters (manual box threshold filtering)
            t1 = time.time()
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=self.text_threshold,
                target_sizes=[(H, W)],
            )[0]
            logger.info("GroundingDINO(HF) post-process %.2fs (raw=%d)", time.time() - t1, len(results.get("scores", [])))

            # Group by label for per-class filtering and NMS
            class_boxes = defaultdict(list)
            class_scores = defaultdict(list)
            labels_key = "text_labels" if "text_labels" in results else "labels"
            for box, score, label in zip(results["boxes"], results["scores"], results[labels_key]):
                # Handle both string labels (new) and integer IDs (old)
                if isinstance(label, str):
                    # New format: direct string label
                    label_text = label
                else:
                    # Old format: integer ID, map back to string
                    if int(label) < len(queries):
                        label_text = queries[int(label)]
                    else:
                        continue
                
                thresh = self.confidence_threshold.get(label_text, self.confidence_threshold) if isinstance(self.confidence_threshold, dict) else self.confidence_threshold
                if float(score) < thresh:
                    continue
                
                class_boxes[label_text].append(box)
                class_scores[label_text].append(score)

            yolo, labels_out = [], []
            for label in class_boxes:
                if not class_boxes[label]:
                    continue
                cb = torch.stack(class_boxes[label])  # [M, 4]
                cs = torch.tensor(class_scores[label], device=cb.device)
                
                iou_thr = self.nms_iou.get(label, self.nms_iou) if isinstance(self.nms_iou, dict) else self.nms_iou
                keep = nms(cb, cs, iou_thr)
                
                kept_boxes = cb[keep]
                kept_scores = cs[keep]
                
                for kb, ks in zip(kept_boxes, kept_scores):
                    x0, y0, x1, y1 = kb.tolist()
                    clipped = self._clip_xyxy(x0, y0, x1, y1, W, H)
                    if not clipped:
                        continue
                    cid = name_to_id.get(label)
                    if cid is None:
                        continue
                    yolo.append(self._bbox_to_yolo(*clipped, W, H, cid, ks))
                    labels_out.append(label)

            return (len(yolo) > 0), yolo, (", ".join(labels_out) if labels_out else ", ".join(queries))

        except Exception as e:
            logger.exception("GroundingDINO(HF) detector error")
            return False, [], f"Error processing GroundingDINO(HF) detection: {e}"