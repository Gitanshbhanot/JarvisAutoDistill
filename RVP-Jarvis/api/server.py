import io
import os
import sys
import json
import base64
import threading
from dataclasses import dataclass
from typing import Optional, Any
from io import BytesIO
import zipfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from pathlib import Path
import fcntl
import shutil
from typing import Dict
from google import genai
from shutil import copytree, ignore_patterns
from google.genai import types
from dotenv import load_dotenv
import cv2

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.state import app_state
from core.data_processing import process_uploaded_zip, start_annotation, get_current_status
from core.training import get_available_annotated_datasets
from finetune.train import train_yolo_model
from test.inference import YOLOInference
from core.database import (
    get_available_annotated_datasets_for_viewing,
    load_dataset_info,
    view_annotated_image,
    save_golden_set,
    load_golden_set,
)
from finetune.train import YOLOTrainer

load_dotenv()
# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

@dataclass
class TrainingTask:
    dataset: str
    epochs: int
    batch_size: int
    status: str = "pending"
    message: str = ""

class ServerState:
    def __init__(self) -> None:
        self.state_file = PROJECT_ROOT / "data" / "server_state.json"
        self.state_file.parent.mkdir(exist_ok=True)
        self._initialize_state()

    def _initialize_state(self):
        """Initialize state file if it doesn't exist."""
        default_state = {"training_task": None}
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

    def set_training(self, task: Optional[TrainingTask]) -> None:
        state = self._read_state()
        state["training_task"] = None if task is None else {
            "dataset": task.dataset,
            "epochs": task.epochs,
            "batch_size": task.batch_size,
            "status": task.status,
            "message": task.message
        }
        self._write_state(state)

    def get_training(self) -> Optional[TrainingTask]:
        state = self._read_state()
        task = state.get("training_task")
        if task is None:
            return None
        return TrainingTask(
            dataset=task["dataset"],
            epochs=task["epochs"],
            batch_size=task["batch_size"],
            status=task["status"],
            message=task["message"]
        )

    def update_training_status(self, status: str, message: str) -> None:
        state = self._read_state()
        if state["training_task"] is not None:
            state["training_task"]["status"] = status
            state["training_task"]["message"] = message
            self._write_state(state)

server_state = ServerState()

app = Flask(__name__)
CORS(app)

@app.route("/api/health", methods=["GET"])
def health() -> Any:
    return jsonify({"ok": True})

@app.route("/api/enhance_prompt", methods=["POST"])
def api_enhance_prompt() -> Any:
    """
    Enhance the provided problem statement using Gemini.
    Returns a single plain-text improved statement.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        problem_statement = (data.get("problem_statement") or "").strip()
        if not problem_statement:
            return jsonify({"ok": False, "error": "problem_statement is required"}), 400

        # Keep guidance inside the prompt to avoid SDK struct quirks.
        prompt = (
            "You rewrite problem statements for object-detection datasets. "
            "Be specific about target classes, context, edge cases, and desired outputs. "
            "Keep it professional and under 150 words. Output plain text only.\n\n"
            "Improve this problem statement:\n\n"
            f"{problem_statement}"
        )

        model_id = os.getenv("GEMINI_ENHANCE_MODEL", "gemini-2.5-flash-lite")

        cfg = types.GenerateContentConfig(
            temperature=0.3,
            top_p=0.8,
            max_output_tokens=512,
            response_mime_type="text/plain",
        )

        resp = gemini_client.models.generate_content(
            model=model_id,
            contents=[prompt],   # <-- pass plain string(s)
            config=cfg,
        )

        # ---- Robust extraction (handles empty .text) ----
        def _extract_text(r) -> str:
            t = getattr(r, "text", None)
            if t:
                return t.strip()
            texts = []
            for cand in (getattr(r, "candidates", None) or []):
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if not parts:
                    continue
                for p in parts:
                    pt = getattr(p, "text", None)
                    if pt:
                        texts.append(pt)
            return "\n".join(texts).strip()

        enhanced = _extract_text(resp)

        # Debug useful metadata if empty
        if not enhanced:
            finish = None
            try:
                finish = getattr(resp.candidates[0], "finish_reason", None)
            except Exception:
                pass
            print(
                "❌ Empty Gemini response",
                "finish_reason:", finish,
                "prompt_feedback:", getattr(resp, "prompt_feedback", None),
                "usage:", getattr(resp, "usage_metadata", None),
            )
            return jsonify({"ok": False, "error": "Empty response from Gemini"}), 502

        return jsonify({"ok": True, "enhanced_prompt": enhanced})

    except Exception as e:
        print(f"Error in /api/enhance_prompt: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/upload_zip", methods=["POST"])
def api_upload_zip() -> Any:
    """
    Upload and process a ZIP file, streaming to disk.
    """
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "file is required (multipart/form-data)"}), 400
        object_name = request.form.get("object_name", type=str)
        if not object_name:
            return jsonify({"ok": False, "error": "object_name is required"}), 400

        file_storage = request.files["file"]
        if file_storage.filename == "":
            return jsonify({"ok": False, "error": "empty filename"}), 400
        
        if not file_storage.filename.lower().endswith('.zip'):
            return jsonify({"ok": False, "error": "file must be a .zip"}), 400

        problem_statement = request.form.get("problem_statement", type=str)
        model_name = request.form.get("model_name", "gemini-2.5-flash-lite")
        
        # Validate problem statement for reasoning models
        if model_name and model_name.startswith("reasoning:"):
            if not problem_statement or not problem_statement.strip():
                return jsonify({"ok": False, "error": "Problem statement is required for reasoning models"}), 400
        
        samples = request.files.getlist("samples")
        confidence_str = request.form.get("confidence_threshold", "0.8")
        try:
            confidence_threshold = float(confidence_str)
            if not 0 <= confidence_threshold <= 1:
                raise ValueError
        except ValueError:
            confidence_threshold = 0.8

        dataset_name = request.form.get("dataset_name", type=str)
        tmp_dir = PROJECT_ROOT / "data" / "_tmp_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / file_storage.filename

        with open(tmp_path, "wb") as f:
            chunk_size = 8192
            while True:
                chunk = file_storage.stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            if zip_ref.testzip() is not None:
                tmp_path.unlink(missing_ok=True)
                return jsonify({"ok": False, "error": "invalid ZIP file"}), 400

        success_msg, status_msg = process_uploaded_zip(str(tmp_path), object_name, dataset_name)
        if "Error" in success_msg or "❌" in success_msg:
            tmp_path.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": success_msg}), 400

        if not app_state.current_timestamp:
            tmp_path.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": "failed to set dataset_id"}), 500

        dataset_folder = PROJECT_ROOT / "data" / "raw_data" / app_state.current_timestamp
        metadata_dir = dataset_folder / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        if problem_statement and problem_statement.strip():
            with open(metadata_dir / "problem_statement.txt", "w") as f:
                f.write(problem_statement.strip())

        with open(metadata_dir / "confidence_threshold.txt", "w") as f:
            f.write(str(confidence_threshold))

        with open(metadata_dir / "model_name.txt", "w") as f:
            f.write(model_name)

        success_msg += f"\nConfidence Threshold: {confidence_threshold}\n"
        success_msg += f"\nModel Name: {model_name}\n"
        success_msg += f"\nProblem Statement: {'Provided' if problem_statement else 'None'}\n"

        samples_dir = dataset_folder / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        saved_samples = []
        for sample in samples:
            if sample.filename and len(saved_samples) < 3:
                sample_path = samples_dir / sample.filename
                with open(sample_path, "wb") as f:
                    chunk_size = 8192
                    while True:
                        chunk = sample.stream.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                saved_samples.append(sample.filename)
        
        success_msg += f"Samples: {len(saved_samples)} uploaded\n"

        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "message": success_msg,
            "status": status_msg,
            "dataset_id": app_state.current_timestamp,
        })
    except Exception as e:
        print(f"Error in /api/upload_zip: {str(e)}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/start_annotation", methods=["POST"])
def api_start_annotation() -> Any:
    """
    Start async annotation task in a background thread.
    UPDATED: Parses provider and model name.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        object_name = data.get("object_name")
        image_size = data.get("image_size")
        upscale_image = data.get("upscale_image") == "true"
        dataset_folder = PROJECT_ROOT / "data" / "raw_data" / app_state.current_timestamp
        metadata_dir = dataset_folder / "metadata"
        if not object_name:
            return jsonify({"ok": False, "error": "object_name is required"}), 400

        confidence_threshold = data.get("confidence_threshold")
        if confidence_threshold is not None:
            try:
                confidence_threshold = float(confidence_threshold)
                if not 0 <= confidence_threshold <= 1:
                    return jsonify({"ok": False, "error": "confidence_threshold must be between 0 and 1"}), 400
            except (ValueError, TypeError):
                return jsonify({"ok": False, "error": "Invalid confidence_threshold, must be a number between 0 and 1"}), 400
        else:
            conf_file = metadata_dir / "confidence_threshold.txt"
            try:
                confidence_threshold = float(conf_file.read_text().strip())
            except (FileNotFoundError, ValueError):
                confidence_threshold = 0.8

        model_name = data.get("model_name")
        if model_name is not None:
            model_name = model_name.strip()
        else:
            model_name_file = metadata_dir / "model_name.txt"
            try:
                model_name = model_name_file.read_text().strip()
            except (FileNotFoundError, ValueError):
                model_name = "gemini:gemini-2.5-flash-lite"
        
        # Validate problem statement for reasoning models
        if model_name and model_name.startswith("reasoning:"):
            problem_statement_file = metadata_dir / "problem_statement.txt"
            try:
                problem_statement = problem_statement_file.read_text().strip()
                if not problem_statement:
                    return jsonify({"ok": False, "error": "Problem statement is required for reasoning models"}), 400
            except FileNotFoundError:
                return jsonify({"ok": False, "error": "Problem statement is required for reasoning models"}), 400

        with open(metadata_dir / "image_size.txt", "w") as f:
            f.write(str(image_size))
        
        with open(metadata_dir / "upscale_image.txt", "w") as f:
            f.write(str(upscale_image))

        def run_annotation() -> None:
            try:
                print(f"🚀 Starting annotation for dataset: {app_state.current_timestamp}")
                print(f"🔧 Using model: {model_name}")
                print(f"🎯 Object name: {object_name}")
                print(f"📊 Confidence threshold: {confidence_threshold}")
                
                message, status = start_annotation(object_name, confidence_threshold=confidence_threshold, model_name=model_name, image_size=image_size, upscale_image=upscale_image)
                
                print(f"📄 Annotation result message: {message}")
                print(f"📊 Annotation status: {app_state.annotation_status}")
                
                # Don't override status if it was already set in start_annotation
                if app_state.annotation_status not in ["completed", "failed", "cancelled"]:
                    app_state.annotation_status = "completed" if "successfully" in message.lower() else "failed"
                
                app_state.annotation_progress.append(message)
                print(f"✅ Annotation thread completed with status: {app_state.annotation_status}")
                
                # DON'T clear current_timestamp - let it persist for status polling
                # DON'T clear progress - let frontend see the results
                app_state.cancelled = False
                
            except Exception as e:
                print(f"❌ Error in annotation thread: {str(e)}")
                import traceback
                traceback.print_exc()
                
                app_state.annotation_status = "failed"
                app_state.annotation_progress.append(f"❌ Error: {str(e)}")
                app_state.cancelled = False
                # DON'T clear current_timestamp and progress here either

        # Don't set status to "running" here - let start_annotation() handle it
        threading.Thread(target=run_annotation, daemon=True).start()
        return jsonify({
            "ok": True,
            "status": "started",
            "dataset_id": app_state.current_timestamp,
            "confidence_threshold": confidence_threshold
        })
    except Exception as e:
        print(f"Error in /api/start_annotation: {str(e)}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/annotation/status", methods=["GET"])
def api_annotation_status() -> Any:
    try:
        status_text = get_current_status()
        return jsonify({
            "ok": True,
            "dataset_id": app_state.current_timestamp,
            "annotation_status": app_state.annotation_status,
            "current": app_state.annotation_current,
            "total": app_state.annotation_total,
            "progress": app_state.annotation_progress,
            "status_text": status_text,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/annotation/cancel", methods=["POST"])
def api_annotation_cancel() -> Any:
    """
    Cancel the current annotation task.
    """
    try:
        if app_state.annotation_status not in ["pending", "running"]:
            return jsonify({"ok": False, "error": "No active annotation to cancel"}), 400
        
        # Store the current timestamp before clearing it
        current_dataset = app_state.current_timestamp
        
        app_state.cancelled = True
        app_state.annotation_status = "cancelled"
        app_state.annotation_progress.append("❌ Annotation cancelled by user")
        
        # Cleanup temporary uploads
        tmp_dir = PROJECT_ROOT / "data" / "_tmp_uploads"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        
        # Clean up raw and annotated data for the current dataset
        if current_dataset:
            raw_data_dir = PROJECT_ROOT / "data" / "raw_data" / current_dataset
            annotated_data_dir = PROJECT_ROOT / "data" / "annotated_data" / current_dataset
            
            if raw_data_dir.exists():
                shutil.rmtree(raw_data_dir)
                print(f"🧹 Cleaned up raw data directory: {raw_data_dir}")
                
            if annotated_data_dir.exists():
                shutil.rmtree(annotated_data_dir)
                print(f"🧹 Cleaned up annotated data directory: {annotated_data_dir}")
        
        # Clear app_state
        app_state.current_timestamp = None
        app_state.annotation_progress = []
        app_state.annotation_total = 0
        app_state.annotation_current = 0
        return jsonify({"ok": True, "message": "Annotation cancelled"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/datasets", methods=["GET"])
def api_datasets() -> Any:
    try:
        datasets = get_available_annotated_datasets()
        return jsonify({"ok": True, "datasets": datasets})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/datasets/<dataset>/download", methods=["GET"])
def api_download_dataset(dataset: str) -> Any:
    try:
        raw_dir = PROJECT_ROOT / "data" / "raw_data" / dataset
        annotated_dir = PROJECT_ROOT / "data" / "annotated_data" / dataset

        if not (raw_dir.exists() or annotated_dir.exists()):
            return jsonify({"ok": False, "error": "Dataset not found"}), 404

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            if raw_dir.exists():
                for root, _, files in os.walk(raw_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = f"raw/{file_path.relative_to(raw_dir)}"
                        zf.write(file_path, arcname)

            if annotated_dir.exists():
                for root, _, files in os.walk(annotated_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = f"annotated/{file_path.relative_to(annotated_dir)}"
                        zf.write(file_path, arcname)

        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{dataset}.zip",
            mimetype="application/zip"
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/datasets/<dataset>/annotated_video", methods=["GET"])
def api_annotated_video(dataset: str) -> Any:
    """
    Generate and return a stitched video of all annotated frames in the dataset.
    Query params:
    - fps: frames per second (default: 5)
    """
    try:
        fps = request.args.get("fps", 5, type=int)
        if fps <= 0:
            fps = 5

        images, class_names = load_dataset_info(dataset)
        if not images:
            return jsonify({"ok": False, "error": "No images in dataset"}), 404

        # Sort images by relative path for consistent order
        images.sort()

        frames = []
        for img_rel in images:
            annotated_np, info_text = view_annotated_image(
                dataset,
                img_rel,
                class_names,
                include_classes=None  # Include all classes
            )
            if annotated_np is not None:
                # Convert RGB to BGR for cv2
                annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)
                frames.append(annotated_bgr)

        if not frames:
            return jsonify({"ok": False, "error": "No annotated frames"}), 404

        # Assume all frames have the same size
        height, width, _ = frames[0].shape

        # Create temporary file for video
        import tempfile
        tmp_path = Path(tempfile.mktemp(suffix='.mp4'))

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))

            for frame in frames:
                video.write(frame)

            video.release()

            # Load video data into memory
            with open(tmp_path, 'rb') as f:
                video_data = f.read()

            return send_file(
                BytesIO(video_data),
                as_attachment=True,
                download_name=f"{dataset}_annotated.mp4",
                mimetype="video/mp4"
            )
        finally:
            # Clean up temp file
            if tmp_path.exists():
                os.unlink(str(tmp_path))

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def api_train() -> Any:
    """
    Start a training task in a background thread.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        dataset = data.get("dataset")
        if not dataset:
            return jsonify({"ok": False, "error": "dataset is required"}), 400

        epochs = data.get("epochs", 100)
        batch_size = data.get("batch_size", 16)
        try:
            epochs = int(epochs)
            batch_size = int(batch_size)
            if epochs <= 0 or batch_size <= 0:
                raise ValueError
        except ValueError:
            return jsonify({"ok": False, "error": "epochs and batch_size must be positive integers"}), 400

        if server_state.get_training():
            return jsonify({"ok": False, "error": "Another training task is already running"}), 400

        task = TrainingTask(dataset=dataset, epochs=epochs, batch_size=batch_size)
        task.status = "running"
        task.message = f"Training started for dataset {dataset}"
        server_state.set_training(task)

        def run_training():
            try:
                success = train_yolo_model(dataset, epochs=epochs, batch_size=batch_size)
                
                # Check if training was cancelled during execution
                current_task = server_state.get_training()
                if current_task and current_task.status == "cancelled":
                    print(f"Training was cancelled for dataset {dataset}")
                    return  # Don't update status if already cancelled
                
                server_state.update_training_status(
                    "completed" if success else "failed",
                    f"Training {'completed' if success else 'failed'} for dataset {dataset}"
                )
                server_state.set_training(None)  # Clear state
            except Exception as e:
                # Check if training was cancelled during execution
                current_task = server_state.get_training()
                if current_task and current_task.status == "cancelled":
                    print(f"Training was cancelled for dataset {dataset} (exception case)")
                    return  # Don't update status if already cancelled
                    
                server_state.update_training_status("failed", f"Training failed: {str(e)}")
                server_state.set_training(None)  # Clear state

        threading.Thread(target=run_training, daemon=True).start()
        return jsonify({"ok": True, "message": "Training started"})
    except Exception as e:
        print(f"Error in /api/train: {str(e)}")
        server_state.set_training(None)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/train/status", methods=["GET"])
def api_train_status() -> Any:
    """
    Get the current training task status.
    """
    try:
        task = server_state.get_training()
        if not task:
            return jsonify({"ok": True, "status": "idle", "dataset": "", "message": "No training task running"})
        return jsonify({
            "ok": True,
            "status": task.status,
            "dataset": task.dataset,
            "message": task.message,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/train/cancel", methods=["POST"])
def api_train_cancel() -> Any:
    """
    Cancel the current training task.
    """
    try:
        task = server_state.get_training()
        if not task or task.status not in ["pending", "running"]:
            return jsonify({"ok": False, "error": "No active training to cancel"}), 400
        
        # Signal cancellation to the trainer process via trainer state file
        trainer = YOLOTrainer()
        trainer.cancelled = True
        print(f"🛑 Set trainer cancellation flag for dataset: {task.dataset}")
        
        task.status = "cancelled"
        task.message = "Training cancelled by user"
        server_state.set_training(None)
        runs_dir = PROJECT_ROOT / "models" / task.dataset / "runs"
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
        return jsonify({"ok": True, "message": "Training cancelled"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/models", methods=["GET"])
def api_models() -> Any:
    try:
        inference_engine = YOLOInference()
        models = inference_engine.get_available_models()
        return jsonify({"ok": True, "models": models})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/models/<timestamp>", methods=["DELETE"])
def api_delete_model(timestamp: str) -> Any:
    """
    Delete a trained model and its runs directory.
    """
    try:
        model_path = PROJECT_ROOT / "models" / f"{timestamp}.pt"
        runs_dir = PROJECT_ROOT / "models" / timestamp / "runs"
        deleted = False

        if model_path.exists():
            model_path.unlink()
            deleted = True

        if runs_dir.exists():
            def _delete_runs() -> None:
                try:
                    shutil.rmtree(runs_dir)
                except Exception:
                    pass
            threading.Thread(target=_delete_runs, daemon=True).start()
            deleted = True

        if not deleted:
            return jsonify({"ok": False, "error": "model not found"}), 404

        server_state.set_training(None)  # Clear state
        return jsonify({"ok": True, "message": "Model deleted; runs cleanup in background"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/models/<timestamp>/download", methods=["GET"])
def api_download_model(timestamp: str) -> Any:
    try:
        model_path = PROJECT_ROOT / "models" / f"{timestamp}.pt"
        if not model_path.exists():
            return jsonify({"ok": False, "error": "model not found"}), 404
        return send_file(str(model_path), as_attachment=True, download_name=f"{timestamp}.pt")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/train/<dataset>/cleanup", methods=["POST"])
def api_train_cleanup(dataset: str) -> Any:
    """
    Cleanup runs data for a failed training job.
    """
    try:
        runs_dir = PROJECT_ROOT / "models" / dataset / "runs"
        if runs_dir.exists():
            def _cleanup() -> None:
                try:
                    shutil.rmtree(runs_dir)
                except Exception:
                    pass
            threading.Thread(target=_cleanup, daemon=True).start()
        server_state.set_training(None)  # Clear state
        return jsonify({"ok": True, "message": "Cleanup started"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/inference", methods=["POST"])
def api_inference() -> Any:
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "image file is required"}), 400

        image_file = request.files["image"]
        conf = float(request.form.get("confidence", 0.25))
        model_timestamp = request.form.get("model_timestamp")
        model_file = request.files.get("model_file")

        image = Image.open(image_file.stream).convert("RGB")

        model_path: Optional[str] = None
        tmp_model_path: Optional[Path] = None
        if model_file is not None and model_file.filename:
            tmp_dir = PROJECT_ROOT / "data" / "_tmp_uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_model_path = tmp_dir / model_file.filename
            model_file.save(str(tmp_model_path))
            model_path = str(tmp_model_path)
        elif model_timestamp:
            candidate = PROJECT_ROOT / "models" / f"{model_timestamp}.pt"
            if candidate.exists():
                model_path = str(candidate)

        if not model_path:
            return jsonify({"ok": False, "error": "no model provided. Provide model_timestamp or upload model_file"}), 400

        engine = YOLOInference()
        annotated_image, detection_info = engine.run_inference(model_path, image, conf)

        if isinstance(annotated_image, Image.Image):
            b64_image = pil_to_base64(annotated_image)
            width, height = annotated_image.size
        else:
            img = Image.fromarray(annotated_image)
            b64_image = pil_to_base64(img)
            width, height = img.size

        if tmp_model_path is not None:
            try:
                tmp_model_path.unlink(missing_ok=True)
            except Exception:
                pass

        return jsonify({
            "ok": True,
            "confidence": conf,
            "detection_info": detection_info,
            "image": {
                "base64": b64_image,
                "format": "PNG",
                "width": width,
                "height": height,
            },
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/db/datasets", methods=["GET"])
def api_db_datasets() -> Any:
    try:
        datasets = get_available_annotated_datasets_for_viewing()
        return jsonify({"ok": True, "datasets": datasets})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/db/datasets/<dataset>/images", methods=["GET"])
def api_db_dataset_images(dataset: str) -> Any:
    try:
        source = (request.args.get("source") or "annotated").lower()
        if source not in ("annotated", "raw"):
            return jsonify({"ok": False, "error": "invalid source"}), 400

        if source == "annotated":
            images, class_names = load_dataset_info(dataset)
            labels_by_image: Dict[str, Any] = {}
            dataset_path = PROJECT_ROOT / "data" / "annotated_data" / dataset
            for img_rel in images:
                if img_rel.startswith("train/"):
                    label_path = dataset_path / "labels" / "train" / f"{img_rel.split('/')[-1].rsplit('.', 1)[0]}.txt"
                elif img_rel.startswith("val/"):
                    label_path = dataset_path / "labels" / "val" / f"{img_rel.split('/')[-1].rsplit('.', 1)[0]}.txt"
                else:
                    labels_by_image[img_rel] = []
                    continue
                classes_present = []
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    try:
                                        cls_id = int(parts[0])
                                        if cls_id not in classes_present:
                                            classes_present.append(cls_id)
                                    except ValueError:
                                        pass
                    except Exception:
                        pass
                labels_by_image[img_rel] = classes_present
            return jsonify({"ok": True, "images": images, "class_names": class_names, "labels_by_image": labels_by_image, "source": source})
        else:
            raw_root = PROJECT_ROOT / "data" / "raw_data" / dataset
            if not raw_root.exists():
                return jsonify({"ok": True, "images": [], "class_names": [], "labels_by_image": {}, "source": source})
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            raw_images = []
            for path in raw_root.rglob('*'):
                if path.is_file() and path.suffix.lower() in image_extensions:
                    rel = str(path.relative_to(raw_root)).replace('\\', '/')
                    raw_images.append(rel)
            return jsonify({"ok": True, "images": raw_images, "class_names": [], "labels_by_image": {}, "source": source})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/db/datasets/<dataset>/image", methods=["GET"])
def api_db_view_image(dataset: str) -> Any:
    try:
        image_choice = request.args.get("path")
        if not image_choice:
            return jsonify({"ok": False, "error": "query param 'path' is required, e.g., train/foo.jpg"}), 400

        source = (request.args.get("source") or "annotated").lower()
        selected_classes = request.args.getlist("classes")
        if len(selected_classes) == 1 and "," in selected_classes[0]:
            selected_classes = [c.strip() for c in selected_classes[0].split(",") if c.strip()]

        if source == "raw":
            raw_root = PROJECT_ROOT / "data" / "raw_data" / dataset
            raw_image_path = raw_root / image_choice
            if not raw_image_path.exists():
                return jsonify({"ok": False, "error": f"raw image not found: {raw_image_path}"}), 404
            original_pil = Image.open(raw_image_path)
            b64_original = pil_to_base64(original_pil)
            ow, oh = original_pil.size
            return jsonify({
                "ok": True,
                "original": {"base64": b64_original, "format": "PNG", "width": ow, "height": oh},
                "source": source,
                "selected_classes": selected_classes,
            })

        _, class_names = load_dataset_info(dataset)
        valid_set = set(class_names)
        selected_classes = [c for c in selected_classes if c in valid_set]

        annotated_np, info_text = view_annotated_image(
            dataset,
            image_choice,
            class_names,
            include_classes=selected_classes if selected_classes else None,
        )
        if annotated_np is None:
            return jsonify({"ok": False, "error": info_text}), 400

        dataset_path = PROJECT_ROOT / "data" / "annotated_data" / dataset
        if image_choice.startswith("train/"):
            original_image_path = dataset_path / "images" / "train" / image_choice.split("/")[1]
        else:
            original_image_path = dataset_path / "images" / "val" / image_choice.split("/")[1]

        original_pil = Image.open(original_image_path)
        b64_original = pil_to_base64(original_pil)
        orig_w, orig_h = original_pil.size

        img = Image.fromarray(annotated_np)
        b64_image = pil_to_base64(img)
        width, height = img.size

        # Add annotations to response
        dataset_path = PROJECT_ROOT / "data" / "annotated_data" / dataset
        if image_choice.startswith("train/"):
            img_rel = image_choice.split("/")[1]
            label_path = dataset_path / "labels" / "train" / f"{img_rel.rsplit('.', 1)[0]}.txt"
        else:
            img_rel = image_choice.split("/")[1]
            label_path = dataset_path / "labels" / "val" / f"{img_rel.rsplit('.', 1)[0]}.txt"
        
        annotations = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        ann = [int(parts[0])] + [float(p) for p in parts[1:]]
                        if len(ann) == 5: 
                            ann.append(1.0)  # Add default confidence
                        annotations.append(ann)

        # Filter by selected classes
        if selected_classes:
            _, class_names = load_dataset_info(dataset)
            class_ids = [class_names.index(c) for c in selected_classes if c in class_names]
            annotations = [ann for ann in annotations if ann[0] in class_ids]

        return jsonify({
            "ok": True,
            "info": info_text,
            "annotated": {"base64": b64_image, "format": "PNG", "width": width, "height": height},
            "original": {"base64": b64_original, "format": "PNG", "width": orig_w, "height": orig_h},
            "source": source,
            "selected_classes": selected_classes,
            "annotations": annotations,  # [[cid, xc, yc, w, h, conf], ...]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/datasets/<dataset>", methods=["DELETE"])
def api_delete_dataset(dataset: str) -> Any:
    """
    Delete a dataset by timestamp name from both annotated and raw folders.
    """
    try:
        annotated_dir = PROJECT_ROOT / "data" / "annotated_data" / dataset
        raw_dir = PROJECT_ROOT / "data" / "raw_data" / dataset
        deleted = {"annotated": False, "raw": False}

        if annotated_dir.exists() and annotated_dir.is_dir():
            shutil.rmtree(annotated_dir)
            deleted["annotated"] = True

        if raw_dir.exists() and raw_dir.is_dir():
            shutil.rmtree(raw_dir)
            deleted["raw"] = True

        if not deleted["annotated"] and not deleted["raw"]:
            return jsonify({"ok": False, "error": "dataset not found"}), 404

        return jsonify({"ok": True, "deleted": deleted})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/datasets/<dataset>/reannotate", methods=["POST"])
def api_reannotate_dataset(dataset: str) -> Any:
    """
    Create a new dataset for re-annotation based on an existing one.
    """
    try:
        data = request.json
        new_name = data.get("new_name")
        if not new_name:
            return jsonify({"ok": False, "error": "new_name required"}), 400
        
        golden_selections = data.get("golden_selections", {})
        
        old_raw = PROJECT_ROOT / "data" / "raw_data" / dataset
        new_raw = PROJECT_ROOT / "data" / "raw_data" / new_name
        if new_raw.exists():
            return jsonify({"ok": False, "error": "new dataset exists"}), 409
        
        # Copy raw data excluding annotated directories
        shutil.copytree(old_raw, new_raw, ignore=ignore_patterns('annotated*'))
        
        # Copy specific metadata files
        metadata_old = old_raw / "metadata"
        metadata_new = new_raw / "metadata"
        for file in ["problem_statement.txt", "confidence_threshold.txt", "model_name.txt", "image_size.txt", "upscale_image.txt"]:
            if (metadata_old / file).exists():
                shutil.copy(metadata_old / file, metadata_new / file)
        
        # Save golden set
        save_golden_set(new_name, golden_selections)
        
        # Load class names and start annotation
        _, class_names = load_dataset_info(dataset)

        # Load confidence_threshold
        conf_file = metadata_new / "confidence_threshold.txt"
        try:
            confidence_threshold = float(conf_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            confidence_threshold = 0.8

        # Load model_name
        model_name_file = metadata_new / "model_name.txt"
        try:
            model_name = model_name_file.read_text().strip()
        except (FileNotFoundError, ValueError):
            model_name = "gemini:gemini-2.5-flash-lite"
        
        # Validate problem statement for reasoning models
        if model_name and model_name.startswith("reasoning:"):
            problem_statement_file = metadata_new / "problem_statement.txt"
            try:
                problem_statement = problem_statement_file.read_text().strip()
                if not problem_statement:
                    return jsonify({"ok": False, "error": "Problem statement is required for reasoning models"}), 400
            except FileNotFoundError:
                return jsonify({"ok": False, "error": "Problem statement is required for reasoning models"}), 400

        image_size_file = metadata_new / "image_size.txt"
        try:
            image_size = int(image_size_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            image_size = 512
            
        upscale_image_file = metadata_new / "upscale_image.txt"
        try:
            upscale_image = bool(upscale_image_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            upscale_image = False

        app_state.current_timestamp = new_name
        app_state.annotation_status = "pending"
        app_state.annotation_progress = []
        app_state.annotation_total = 0
        app_state.annotation_current = 0
        app_state.completed_timestamp = None

        def run_annotation() -> None:
            try:
                print(f"🚀 Starting annotation for dataset: {app_state.current_timestamp}")
                print(f"🔧 Using model: {model_name}")
                print(f"🎯 Object name: {class_names}")
                print(f"📊 Confidence threshold: {confidence_threshold}")
                
                message, status = start_annotation(class_names, confidence_threshold=confidence_threshold, model_name=model_name, image_size=image_size, upscale_image=upscale_image)
                
                print(f"📄 Annotation result message: {message}")
                print(f"📊 Annotation status: {app_state.annotation_status}")
                
                # Don't override status if it was already set in start_annotation
                if app_state.annotation_status not in ["completed", "failed", "cancelled"]:
                    app_state.annotation_status = "completed" if "successfully" in message.lower() else "failed"
                
                app_state.annotation_progress.append(message)
                print(f"✅ Annotation thread completed with status: {app_state.annotation_status}")
                
                # DON'T clear current_timestamp - let it persist for status polling
                # DON'T clear progress - let frontend see the results
                app_state.cancelled = False
                
            except Exception as e:
                print(f"❌ Error in annotation thread: {str(e)}")
                import traceback
                traceback.print_exc()
                
                app_state.annotation_status = "failed"
                app_state.annotation_progress.append(f"❌ Error: {str(e)}")
                app_state.cancelled = False
                # DON'T clear current_timestamp and progress here either

        threading.Thread(target=run_annotation, daemon=True).start()
        
        return jsonify({"ok": True, "new_dataset": new_name})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7002))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)