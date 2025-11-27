import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import tempfile
from pathlib import Path

class YOLOInference:
    def __init__(self):
        app_dir = Path(__file__).parent.parent
        self.models_path = app_dir / "models"
        print(f"🔍 YOLOInference: models_path = {self.models_path}")
        
    def get_available_models(self):
        """Get list of available trained models"""
        print(f"🔍 Checking for models in: {self.models_path}")
        print(f"🔍 Directory exists: {self.models_path.exists()}")
        
        if not self.models_path.exists():
            print(f"❌ Models directory not found: {self.models_path}")
            return []
        
        model_files = list(self.models_path.glob("*.pt"))
        print(f"🔍 Found {len(model_files)} .pt files: {[f.name for f in model_files]}")
        
        models = []
        for model_file in model_files:
            timestamp = model_file.stem
            models.append({
                'name': f"Model {timestamp}",
                'path': str(model_file),
                'timestamp': timestamp
            })
        return models
    
    def get_model_classes(self, model_path: str):
        """Get class names from a YOLO model"""
        try:
            model = YOLO(model_path)
            if hasattr(model, 'names') and model.names:
                return list(model.names.values())
            else:
                # Fallback: try to get from model.model.names
                if hasattr(model.model, 'names'):
                    return list(model.model.names)
                else:
                    return ["Unknown classes"]
        except Exception as e:
            return [f"Error loading classes: {str(e)}"]
    
    def run_inference(self, model_path: str, image: Image.Image, conf_threshold: float = 0.25):
        """
        Run YOLO inference on an image
        Returns: (annotated_image, detections_info)
        """
        try:
            # Load model
            model = YOLO(model_path)
            
            # Run inference
            results = model(image, conf=conf_threshold)
            
            # Get the first result (single image)
            result = results[0]
            
            # Create annotated image
            annotated_image = self._draw_detections(image.copy(), result)
            
            # Extract detection info
            detections_info = self._extract_detection_info(result)
            
            return annotated_image, detections_info
            
        except Exception as e:
            raise Exception(f"Inference failed: {str(e)}")
    
    def _draw_detections(self, image: Image.Image, result):
        """Draw bounding boxes and labels on image"""
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Get detections
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            class_names = result.names
            
            # Colors for different classes
            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
                "#00FFFF", "#FFA500", "#800080", "#FFC0CB", "#A52A2A"
            ]
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box
                class_name = class_names[int(cls)]
                color = colors[int(cls) % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label background
                label = f"{class_name}: {conf:.2f}"
                bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
                
                # Draw label text
                draw.text((x1, y1), label, fill="white", font=font)
        
        return image
    
    def _extract_detection_info(self, result):
        """Extract detection information as text"""
        info = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            class_names = result.names
            
            info.append(f"**Detections Found: {len(boxes)}**")
            info.append("")
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box
                class_name = class_names[int(cls)]
                
                info.append(f"**Detection {i+1}:**")
                info.append(f"- Class: {class_name}")
                info.append(f"- Confidence: {conf:.3f}")
                info.append(f"- Bounding Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                info.append(f"- Size: {x2-x1:.1f} x {y2-y1:.1f} pixels")
                info.append("")
        else:
            info.append("**No detections found**")
        
        return "\n".join(info)

def test_model_on_image(model_path: str, image: Image.Image, conf_threshold: float = 0.25):
    """
    Convenience function for testing a model on an image
    Returns: (annotated_image, detection_info)
    """
    inference = YOLOInference()
    return inference.run_inference(model_path, image, conf_threshold) 