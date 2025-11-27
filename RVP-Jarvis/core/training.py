from pathlib import Path
from finetune.train import train_yolo_model, get_available_models

def get_available_annotated_datasets():
    """Get list of available annotated datasets"""
    annotated_data_path = Path("data/annotated_data")
    available_datasets = []
    if annotated_data_path.exists():
        for folder in annotated_data_path.iterdir():
            if folder.is_dir() and (folder / "dataset.yaml").exists():
                available_datasets.append(folder.name)
    return available_datasets

def train_model(dataset_name, epochs, batch_size):
    """Train model on selected dataset"""
    if not dataset_name:
        return "❌ Please select a dataset first."
    
    try:
        success = train_yolo_model(dataset_name, epochs=epochs)
        if success:
            return f"✅ Training completed for dataset {dataset_name}!"
        else:
            return f"❌ Training failed for dataset {dataset_name}!"
    except Exception as e:
        return f"❌ Training error: {str(e)}" 