import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
import fcntl
import torch
import threading
import time
import signal

class YOLOTrainer:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.state_file = self.project_root / "data" / "trainer_state.json"
        self.state_file.parent.mkdir(exist_ok=True)
        self._initialize_state()
        self.annotated_data_path = self.project_root / "data" / "annotated_data"
        self.models_path = self.project_root / "models"
        self.models_path.mkdir(exist_ok=True)
        
    def _initialize_state(self):
        """Initialize state file if it doesn't exist."""
        default_state = {"cancelled": False}
        if not self.state_file.exists():
            with open(self.state_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(default_state, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _read_state(self):
        """Read state from JSON file with locking."""
        try:
            with open(self.state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                state = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return state
        except (json.JSONDecodeError, FileNotFoundError):
            self._initialize_state()
            return self._read_state()

    def _write_state(self, state):
        """Write state to JSON file with locking."""
        with open(self.state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(state, f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @property
    def cancelled(self):
        return self._read_state().get("cancelled", False)

    @cancelled.setter
    def cancelled(self, value):
        state = self._read_state()
        state["cancelled"] = value
        self._write_state(state)

    def train_model(self, timestamp: str, epochs: int = 100, img_size: int = 640, batch_size: int = 16) -> bool:
        """
        Train YOLO model on annotated data for specific timestamp.
        """
        try:
            dataset_path = self.annotated_data_path / timestamp
            dataset_yaml = dataset_path / "dataset.yaml"
            
            if not dataset_path.exists():
                print(f"Error: Annotated data folder {dataset_path} does not exist")
                return False
                
            if not dataset_yaml.exists():
                print(f"Error: Dataset YAML file {dataset_yaml} does not exist")
                return False
            
            # Validate train/val images
            train_path = dataset_path / "images" / "train"
            val_path = dataset_path / "images" / "val"
            if not train_path.exists() or not any(train_path.glob("*.[jJ][pP][gG]") or train_path.glob("*.[pP][nN][gG]")):
                print(f"Error: No training images found in {train_path}")
                return False
            if not val_path.exists() or not any(val_path.glob("*.[jJ][pP][gG]") or val_path.glob("*.[pP][nN][gG]")):
                print(f"Warning: No validation images found in {val_path}, proceeding with training only")
            
            model = YOLO('yolo11n.pt')
            print(f"🎯 Starting training for timestamp {timestamp}...")
            print(f"📊 Dataset: {dataset_yaml}")
            print(f"📁 Current working directory: {os.getcwd()}")
            print(f"📁 Dataset path: {dataset_path}")
            print(f"📁 Dataset YAML: {dataset_yaml}")
            print(f"⚙️  Configuration: {epochs} epochs, {img_size}px images, batch size: {batch_size}")
            
            # Clean up any existing runs directory before training
            runs_dir = Path("runs")
            if runs_dir.exists():
                try:
                    shutil.rmtree(runs_dir)
                    print(f"🧹 Cleaned up existing runs directory before training")
                except Exception as e:
                    print(f"⚠️  Could not clean up existing runs directory: {e}")
            
            # Train and save directly to models directory
            model_save_path = self.models_path / f"{timestamp}.pt"
            runs_project = self.models_path / timestamp / "runs"
            runs_project.mkdir(parents=True, exist_ok=True)
            
            # Change to dataset directory for training
            original_cwd = os.getcwd()
            os.chdir(str(dataset_path))
            print(f"📁 Changed to dataset directory: {dataset_path}")
            print(f"📁 Current working directory: {os.getcwd()}")
            
            # Verify relative paths exist
            train_path = Path("images/train")
            val_path = Path("images/val")
            print(f"✅ Train path exists: {train_path.exists()}")
            print(f"✅ Val path exists: {val_path.exists()}")
            
            # List files in train and val directories
            if train_path.exists():
                train_files = list(train_path.glob("*.jpg")) + list(train_path.glob("*.jpeg")) + list(train_path.glob("*.png"))
                print(f"📁 Train images: {len(train_files)} files")
                for f in train_files[:3]:
                    print(f"  - {f.name}")
            
            if val_path.exists():
                val_files = list(val_path.glob("*.jpg")) + list(val_path.glob("*.jpeg")) + list(val_path.glob("*.png"))
                print(f"📁 Val images: {len(val_files)} files")
                for f in val_files[:3]:
                    print(f"  - {f.name}")
            
            # Check for cancellation before starting training
            if self.cancelled:
                print("Training cancelled by user before starting")
                self.cancelled = False
                return False
            
            # Train model with device check
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            
            # Create a custom callback to check for cancellation during training
            def check_cancellation(trainer):
                if self.cancelled:
                    print("Training cancelled by user during training")
                    trainer.stop_training = True
                    self.cancelled = False
                    return True
                return False
            
            # Train model with cancellation monitoring
            training_pid = None
            monitor_thread = None
            training_completed = threading.Event()
            
            def monitor_cancellation():
                """Monitor for cancellation requests during training"""
                while not training_completed.is_set():
                    if self.cancelled:
                        print("🛑 Training cancellation detected during training")
                        # Try to gracefully stop the training process
                        if training_pid:
                            try:
                                os.kill(training_pid, signal.SIGTERM)
                                print(f"🛑 Sent SIGTERM to training process {training_pid}")
                            except ProcessLookupError:
                                pass  # Process already ended
                        self.cancelled = False
                        return
                    time.sleep(1)  # Check every second
            
            try:
                # Start the monitoring thread
                monitor_thread = threading.Thread(target=monitor_cancellation, daemon=True)
                monitor_thread.start()
                training_pid = os.getpid()  # Get current process ID
                
                model.train(
                    data=str(dataset_yaml),
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    device=device,
                    project=str(runs_project),
                    name="train",
                    exist_ok=True
                )
                
                # Signal training completion
                training_completed.set()
                
                # Additional check after training completion
                if self.cancelled:
                    print("Training cancelled by user after completion")
                    self.cancelled = False
                    return False
                    
            except Exception as e:
                # Signal training completion
                training_completed.set()
                
                # Check if cancellation caused the exception
                if self.cancelled:
                    print("Training cancelled by user (via exception)")
                    self.cancelled = False
                    return False
                else:
                    raise e
            finally:
                # Ensure monitoring thread stops
                training_completed.set()
                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=2)
            
            # Check for cancellation after training
            if self.cancelled:
                print("Training cancelled by user")
                self.cancelled = False
                return False
            
            # Restore original working directory
            os.chdir(original_cwd)
            print(f"📁 Restored working directory: {original_cwd}")
            
            # Try to copy the best model from runs directory
            runs_best_model = runs_project / "train" / "weights" / "best.pt"
            runs_last_model = runs_project / "train" / "weights" / "last.pt"
            
            print(f"🔍 Checking for saved models...")
            print(f"📁 Best model path: {runs_best_model}")
            print(f"📁 Last model path: {runs_last_model}")
            print(f"✅ Best model exists: {runs_best_model.exists()}")
            print(f"✅ Last model exists: {runs_last_model.exists()}")
            
            if runs_best_model.exists():
                shutil.copy2(runs_best_model, model_save_path)
                print(f"✅ Copied best model from runs directory to: {model_save_path}")
            elif runs_last_model.exists():
                shutil.copy2(runs_last_model, model_save_path)
                print(f"✅ Copied last model from runs directory to: {model_save_path}")
            else:
                model.save(str(model_save_path))
                print(f"✅ Saved current model to: {model_save_path}")
            
            return True
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
            self.cancelled = False
            return False
    
    def list_available_models(self):
        """
        List all trained models.
        """
        model_files = list(self.models_path.glob("*.pt"))
        if not model_files:
            print("No trained models found")
            return []
        
        models = []
        for model_file in model_files:
            timestamp = model_file.stem
            model_info = {'timestamp': timestamp, 'path': str(model_file)}
            models.append(model_info)
        
        return models

def train_yolo_model(timestamp: str, epochs: int = 100, batch_size: int = 16) -> bool:
    """
    Main function to train YOLO model with automatic model download.
    """
    trainer = YOLOTrainer()
    return trainer.train_model(timestamp, epochs, batch_size=batch_size)

def get_available_models():
    """
    Get list of available trained models.
    """
    trainer = YOLOTrainer()
    return trainer.list_available_models()