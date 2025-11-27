# Requires: pip install opencv-contrib-python pillow
import cv2
import numpy as np
import os
import threading
from PIL import Image

# Thread-safe model caching
_EDSR_SR = None
_EDSR_MODEL_LOADED = None
_MODEL_LOCK = threading.Lock()

def edsr_enhance(
    pil_img: Image.Image,
    model_path: str = None,
    scale: int = 4,
    supersample_x8: bool = True,
    cleanup: bool = True,
    skip_if_multithreaded: bool = True
) -> Image.Image:
    """
    Enhance a PIL image using EDSR super-resolution (x4 by default) and return a PIL image.
    - If supersample_x8=True, applies EDSR x4 twice (≈x8) and downsamples back to the x4 size
      with Lanczos for extra crisp edges (often outperforms single x4).
    - If cleanup=True, runs a very mild denoise + unsharp to reduce mush and keep edges tight.

    Args:
        pil_img: Input PIL.Image in any mode (converted to RGB internally).
        model_path: Path to EDSR_x4.pb (OpenCV dnn_superres model). If None, uses default path.
        scale: SR scale (normally 4 for EDSR_x4.pb).
        supersample_x8: Apply two x4 passes then downscale to the x4 size.
        cleanup: Apply conservative post-processing (denoise + light sharpen).
        skip_if_multithreaded: If True, skip enhancement in multi-threaded contexts to avoid conflicts.

    Returns:
        PIL.Image.Image (RGB)
    """
    if not hasattr(cv2, "dnn_superres"):
        raise ImportError(
            "cv2.dnn_superres not found. Install contrib build: `pip install opencv-contrib-python`"
        )
    
    # Skip enhancement in multi-threaded contexts to avoid thread safety issues
    if skip_if_multithreaded and threading.active_count() > 1:
        return pil_img

    # Set default model path if not provided
    if model_path is None:
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "upscaleModels", "EDSR_x4.pb")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"EDSR model not found at: {model_path}")

    global _EDSR_SR, _EDSR_MODEL_LOADED
    
    # Thread-safe lazy-load + cache the model
    with _MODEL_LOCK:
        if _EDSR_SR is None or _EDSR_MODEL_LOADED != (model_path, scale):
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(model_path)
                sr.setModel("edsr", scale)
                _EDSR_SR = sr
                _EDSR_MODEL_LOADED = (model_path, scale)
            except Exception as e:
                # If model loading fails, don't cache anything
                _EDSR_SR = None
                _EDSR_MODEL_LOADED = None
                raise e

    # PIL (RGB) -> OpenCV (BGR)
    rgb = pil_img.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    # Check minimum dimensions (some models require minimum sizes)
    h, w = bgr.shape[:2]
    if h < 4 or w < 4:
        raise ValueError(f"Image too small for upscaling: {w}x{h}. Minimum size is 4x4 pixels.")

    # Single x4
    edsr4 = _EDSR_SR.upsample(bgr)

    # Optional x8 supersample trick
    if supersample_x8:
        up8 = _EDSR_SR.upsample(edsr4)  # second x4 pass
        h4, w4 = edsr4.shape[:2]
        edsr4 = cv2.resize(up8, (w4, h4), interpolation=cv2.INTER_LANCZOS4)
        del up8  # Clear intermediate variable

    # Optional conservative cleanup (kept subtle to avoid halos)
    if cleanup:
        den = cv2.fastNlMeansDenoisingColored(edsr4, None, 3, 3, 7, 21)
        blur = cv2.GaussianBlur(den, (0, 0), 0.7)
        edsr4 = cv2.addWeighted(den, 1.10, blur, -0.10, 0)
        # Clear intermediate variables to save memory
        del den, blur

    # Back to PIL
    out_rgb = cv2.cvtColor(edsr4, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(out_rgb)
    
    # Clear intermediate variables to save memory
    del bgr, edsr4, out_rgb
    
    return result


def cleanup_edsr_model():
    """
    Clean up the cached EDSR model to free memory.
    Useful when switching between different model configurations or when done processing.
    """
    global _EDSR_SR, _EDSR_MODEL_LOADED
    with _MODEL_LOCK:
        _EDSR_SR = None
        _EDSR_MODEL_LOADED = None
