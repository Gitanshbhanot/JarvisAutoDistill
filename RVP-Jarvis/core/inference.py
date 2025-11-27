import os
import tempfile
from pathlib import Path
from PIL import Image
from test.inference import YOLOInference
import gradio as gr

def get_available_models_for_testing():
    """Get available models for testing"""
    # Ensure we're in the correct directory
    app_dir = Path(__file__).parent.parent
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 App directory: {app_dir}")
    
    inference_engine = YOLOInference()
    models = inference_engine.get_available_models()
    print(f"🔍 Found {len(models)} models: {models}")
    if models:
        model_names = [f"{model['name']} ({model['timestamp']})" for model in models]
        print(f"📋 Model names: {model_names}")
        return model_names, models
    return [], []

def get_model_classes_info(model_choice, model_data):
    """Get model classes for selected model"""
    if not model_choice or not model_data:
        return "No model selected"
    
    try:
        # Find the selected model
        selected_idx = model_choice
        if isinstance(selected_idx, str):
            # If it's a string, find the index
            model_options, _ = get_available_models_for_testing()
            selected_idx = model_options.index(selected_idx) if selected_idx in model_options else 0
        
        if selected_idx < len(model_data):
            model_path = model_data[selected_idx]['path']
            inference_engine = YOLOInference()
            model_classes = inference_engine.get_model_classes(model_path)
            return f"**Model Classes:** {', '.join(model_classes)}"
    except Exception as e:
        return f"Could not load model classes: {str(e)}"
    
    return "No model information available"

def run_model_inference(model_choice, model_data, uploaded_image, confidence_threshold, uploaded_model_file):
    """Run model inference on uploaded image"""
    selected_model_path = None
    
    # Determine model source
    if uploaded_model_file is not None:
        # Use uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_model_file)
            selected_model_path = tmp_file.name

    elif model_choice and model_data:
        # Use existing model
        try:
            selected_idx = 0
            if isinstance(model_choice, str):
                model_options, _ = get_available_models_for_testing()
                selected_idx = model_options.index(model_choice) if model_choice in model_options else 0
            elif isinstance(model_choice, int):
                selected_idx = model_choice
            
            if selected_idx < len(model_data):
                selected_model_path = model_data[selected_idx]['path']

        except Exception as e:
            return None, f"❌ Error selecting model: {str(e)}"
    
    if not selected_model_path:
        return None, "❌ Please select a model or upload a model file first."
    
    if uploaded_image is None:
        return None, "❌ Please upload an image for testing."
    
    try:
        # Load image
        test_image = Image.open(uploaded_image)
        
        # Run inference
        inference_engine = YOLOInference()
        annotated_image, detection_info = inference_engine.run_inference(
            selected_model_path, 
            test_image, 
            confidence_threshold
        )
        
        # Clean up temporary file if it was uploaded
        if uploaded_model_file is not None and os.path.exists(selected_model_path):
            os.unlink(selected_model_path)
        
        return annotated_image, f"✅ Detection completed!\n\n{detection_info}"
        
    except Exception as e:
        # Clean up temporary file on error
        if uploaded_model_file is not None and os.path.exists(selected_model_path):
            os.unlink(selected_model_path)
        return None, f"❌ Detection failed: {str(e)}"

def download_model(model_choice, model_data):
    """Download selected model"""
    if not model_choice or not model_data:
        return None
    
    try:
        selected_idx = 0
        if isinstance(model_choice, str):
            model_options, _ = get_available_models_for_testing()
            selected_idx = model_options.index(model_choice) if model_choice in model_options else 0
        elif isinstance(model_choice, int):
            selected_idx = model_choice
        
        if selected_idx < len(model_data):
            model_path = model_data[selected_idx]['path']
            timestamp = model_data[selected_idx]['timestamp']
            
            if Path(model_path).exists():
                # Return the file path for download
                return str(model_path)
    except Exception as e:
        pass
    
    return None

def refresh_models():
    """Refresh available models and return updated dropdown choices"""
    models, model_data = get_available_models_for_testing()
    model_update = gr.update(choices=models, value=models[0] if models else None)
    download_update = gr.update(choices=models, value=models[0] if models else None)
    return models, model_data, model_update, download_update 