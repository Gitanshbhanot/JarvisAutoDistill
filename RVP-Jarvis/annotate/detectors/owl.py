# owl.py
from typing import List, Tuple, Dict
from PIL import Image
import logging, time
from dotenv import load_dotenv
from ..detector import BaseDetector
from ..detector import get_device

load_dotenv()
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception as _e:
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None
    _import_error = _e
else:
    _import_error = None

class OWLDetection(BaseDetector):
    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble",
                 object_to_detect: str = "", problem_statement: str = "",
                 sample_images: List[Image.Image] = None,
                 confidence_threshold: float = 0.30,
                 device: str = None):
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            logger.exception("Failed to import transformers/torch", exc_info=_import_error)
            raise ImportError("Install: pip install transformers accelerate torch torchvision pillow")

        self.model_name = model_name
        self.object_to_detect = object_to_detect
        self.problem_statement = problem_statement
        self.sample_images = sample_images or []
        self.confidence_threshold = float(confidence_threshold)
        self.device = get_device(device)

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        logger.info("OWL loaded in %.2fs (model=%s, device=%s)", time.time() - t0, self.model_name, self.device)

    @staticmethod
    def _bbox_to_yolo(x0,y0,x1,y1,W,H,cid):
        return [cid, (x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H]

    @staticmethod
    def _clip_xyxy(x0,y0,x1,y1,W,H):
        x0,y0,x1,y1 = map(int,(max(0,min(W,x0)),max(0,min(H,y0)),max(0,min(W,x1)),max(0,min(H,y1))))
        return None if x0>=x1 or y0>=y1 else (x0,y0,x1,y1)

    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        try:
            queries = [q.strip() for q in (class_names if class_names else [self.object_to_detect]) if q.strip()]
            if not queries:
                logger.warning("OWL: empty queries"); return False, [], "No queries provided"
            name_to_id: Dict[str,int] = {q: cid for q,cid in zip(class_names, class_ids)}

            im = image.convert("RGB"); W,H = im.size
            t0 = time.time()
            inputs = self.processor(text=[queries], images=im, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logger.info("OWL forward %.2fs", time.time()-t0)

            t1 = time.time()
            # Use the new recommended method instead of deprecated post_process_object_detection
            results = self.processor.post_process_grounded_object_detection(
                outputs, 
                target_sizes=[(H, W)]
            )[0]
            logger.info("OWL post-process %.2fs (raw=%d)", time.time()-t1, len(results.get("scores", [])))

            yolo, labels_out = [], []
            for box, score, lid in zip(results["boxes"], results["scores"], results["labels"]):
                # Handle score - might be tensor or float depending on method
                s = float(score.item()) if hasattr(score, 'item') else float(score)
                if s < self.confidence_threshold: 
                    continue
                    
                # Handle box coordinates - might be tensor or list
                if hasattr(box, 'tolist'):
                    x0, y0, x1, y1 = map(int, box.tolist())
                else:
                    x0, y0, x1, y1 = map(int, box)
                    
                clipped = self._clip_xyxy(x0, y0, x1, y1, W, H)
                if not clipped: 
                    continue
                    
                # Handle label id - might be tensor or int
                label_idx = int(lid.item()) if hasattr(lid, 'item') else int(lid)
                if label_idx >= len(queries):
                    continue  # Safety check for out-of-bounds
                    
                label = queries[label_idx]
                cid = name_to_id.get(label)
                if cid is None: 
                    continue
                    
                x0, y0, x1, y1 = clipped
                yolo.append(self._bbox_to_yolo(x0, y0, x1, y1, W, H, cid))
                labels_out.append(label)

            return (len(yolo)>0), yolo, (", ".join(labels_out) if labels_out else ", ".join(queries))
        except Exception as e:
            logger.exception("OWL detector error"); return False, [], f"Error processing OWL detection: {e}"
