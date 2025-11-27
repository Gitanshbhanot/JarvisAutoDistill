from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image
from dotenv import load_dotenv
import logging
import torch

load_dotenv()

logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    @abstractmethod
    def detector(self, image: Image.Image, class_names: List[str], class_ids: List[int]) -> Tuple[bool, List[List[float]], str]:
        """
        Detect objects in an image and return YOLO-compatible annotations.
        Returns: (success, annotations, label_str)
        - success: bool indicating if detection was successful
        - annotations: List of [class_id, x_center, y_center, width, height] (normalized)
        - label_str: String describing detected objects
        """
        pass

def get_device(preferred: str = None) -> str:
    """
    Determine best device to use for inference.
    - CUDA if available
    - else MPS (Apple Silicon GPU) if available
    - else CPU
    """
    # If user explicitly requests a device
    if preferred:
        if preferred == "cuda" and torch.cuda.is_available():
            logger.info("Using preferred device: CUDA (GPU)")
            return "cuda"
        elif preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("Using preferred device: MPS (Apple Silicon GPU)")
            return "mps"
        elif preferred == "cpu":
            logger.info("Using preferred device: CPU")
            return "cpu"
        else:
            logger.warning("Preferred device '%s' not available. Falling back.", preferred)

    # Auto-detect
    if torch.cuda.is_available():
        logger.info("Auto-detected CUDA GPU")
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        logger.info("Auto-detected Apple MPS GPU")
        return "mps"
    else:
        logger.info("No GPU found, using CPU")
        return "cpu"