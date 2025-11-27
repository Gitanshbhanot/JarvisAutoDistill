import os
import shutil
from pathlib import Path
from typing import List, Sequence, Dict
from PIL import Image
import yaml
import time
from annotate.detectors.gemini import GeminiDetection
from annotate.detectors.openRouter import OpenRouterDetection
from annotate.detectors.dino import GroundingDinoHFDetection
from annotate.detectors.owl import OWLDetection
from annotate.detectors.reasoning import AdvancedReasoningDetection
from annotate.detectors.sahiTiler import SAHIDetector
from annotate.detectors.twoStage import TwoStageDetector
# from annotate.detectors.dspyGepa import DSPyGepaGeminiDetection
# from annotate.detectors.samGemini import SAMGeminiDetection
from annotate.detector import BaseDetector, get_device
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from core.state import app_state

class DataAnnotator:
    def __init__(self, model_name: str = "gemini:gemini-2.5-flash-lite"):
        self.model_name = model_name
        self.raw_data_path = Path("data/raw_data")
        self.annotated_data_path = Path("data/annotated_data")
        self.progress_callback = None
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def process_timestamp_folder(self, timestamp: str, object_names: Sequence[str], confidence_threshold: float = 0.8, model_name: str = None, image_size=1024, upscale_image=False, golden_examples=None) -> bool:
        """
        Process a specific timestamp folder and annotate all images in parallel batches.
        """
        try:
            timestamp_raw_path = self.raw_data_path / timestamp
            timestamp_annotated_path = self.annotated_data_path / timestamp
            if not timestamp_raw_path.exists():
                print(f"Error: Raw data folder {timestamp_raw_path} does not exist")
                return False
            images_train_path = timestamp_annotated_path / "images" / "train"
            images_val_path = timestamp_annotated_path / "images" / "val"
            labels_train_path = timestamp_annotated_path / "labels" / "train"
            labels_val_path = timestamp_annotated_path / "labels" / "val"
            for path in [images_train_path, images_val_path, labels_train_path, labels_val_path]:
                path.mkdir(parents=True, exist_ok=True)
            
            self._create_dataset_yaml(timestamp_annotated_path, object_names)
            
            class_names = [name.strip() for name in object_names if str(name).strip()]
            if not class_names:
                print("Error: No valid class names provided")
                return False
            
            metadata_dir = self.raw_data_path / timestamp / "metadata"
            problem_file = metadata_dir / "problem_statement.txt"
            problem_statement = problem_file.read_text().strip() if problem_file.exists() else ""

            conf_file = metadata_dir / "confidence_threshold.txt"
            if conf_file.exists():
                try:
                    confidence_threshold = float(conf_file.read_text().strip())
                except ValueError:
                    pass

            # Use passed model_name parameter, fallback to stored model_name, then metadata file
            if model_name is None:
                model_file = metadata_dir / "model_name.txt"
                model_name = model_file.read_text().strip() if model_file.exists() else self.model_name

            samples_dir = self.raw_data_path / timestamp / "samples"
            sample_paths = list(samples_dir.glob('*')) if samples_dir.exists() else []
            sample_images = []
            for p in sample_paths:
                if p.is_file() and p.suffix.lower() in self.image_extensions:
                    try:
                        sample_images.append(Image.open(p).convert('RGB'))
                    except Exception as e:
                        print(f"Warning: Failed to load sample {p}: {e}")

            # NEW: Select detector based on provider
            print(f"🔍 Processing with model_name: {model_name}")
            provider, model = model_name.split(":", 1) if ":" in model_name else ("gemini", model_name)
            print(f"🔧 Detected provider: {provider}, model: {model}")
            
            if provider == "gemini":
                print(f"🟢 Using Gemini detector with model: {model}")
                detector = GeminiDetection(
                    model_name=model,
                    object_to_detect=class_names[0],
                    problem_statement=problem_statement,
                    sample_images=sample_images,
                    confidence_threshold=confidence_threshold,
                    image_size=image_size,
                    upscale_image=upscale_image,
                    golden_examples=golden_examples or []
                )
            elif provider == "openrouter":
                print(f"🟡 Using OpenRouter detector with model: {model}")
                try:
                    detector = OpenRouterDetection(
                        model_name=model,
                        object_to_detect=class_names[0],
                        problem_statement=problem_statement,
                        sample_images=sample_images,
                        confidence_threshold=confidence_threshold,
                        image_size=image_size,
                        upscale_image=upscale_image
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize OpenRouter detector: {e}")
                    return False
            elif provider == "groundingdino":
                print(f"🟣 Using Grounding DINO")
                try:
                     detector = GroundingDinoHFDetection(
                        model_name=model,  # e.g. "IDEA-Research/grounding-dino-tiny" or "...-base"
                        object_to_detect=class_names[0],
                        problem_statement=problem_statement,
                        sample_images=sample_images,
                        confidence_threshold=confidence_threshold,
                        text_threshold=0.30,
                        device=get_device()
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize Grounding DINO detector: {e}")
                    return False
            elif provider == "owl":
                print(f"🟣 Using OWL detector with model: {model}")
                try:
                    detector = OWLDetection(
                        model_name=model,  # e.g. "google/owlv2-base-patch16-ensemble"
                        confidence_threshold=confidence_threshold,
                        device=get_device()
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize OWL detector: {e}")
                    return False
            elif provider == "reasoning":
                print(f"🟣 Using Reasoning detector with model: {model}")
                try:
                    detector = AdvancedReasoningDetection(
                        model_name=model,
                        confidence_threshold=confidence_threshold,
                        problem_statement=problem_statement,
                        sample_images=sample_images,
                        image_size=image_size,
                        upscale_image=upscale_image
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize Reasoning detector: {e}")
                    return False
            elif provider == "sahi":
                print(f"🟢 Using SAHI tiling detector with model: {model}")
                try:
                    detector = SAHIDetector(
                        model_name=model,
                        object_to_detect=class_names[0],
                        problem_statement=problem_statement,
                        sample_images=sample_images,
                        confidence_threshold=confidence_threshold,
                        image_size=image_size,
                        upscale_image=upscale_image
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize SAHI detector: {e}")
                    return False
            elif provider == "twostage":
                print(f"🟢 Using Two-Stage detector with model: {model}")
                try:
                    detector = TwoStageDetector(
                        model_name=model,
                        object_to_detect=class_names[0],
                        problem_statement=problem_statement,
                        sample_images=sample_images,
                        confidence_threshold=confidence_threshold,
                        image_size=image_size,
                        upscale_image=upscale_image
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize Two-Stage detector: {e}")
                    return False
            # elif provider == "dspyGepa":
            #     print(f"🟢 Using DSPy & GEPA detector with model: {model}")
            #     try:
            #         detector = DSPyGepaGeminiDetection(
            #             model_name=model,
            #             object_to_detect=class_names[0],
            #             problem_statement=problem_statement,
            #             sample_images=sample_images,
            #             confidence_threshold=confidence_threshold,
            #             image_size=image_size,
            #             upscale_image=upscale_image
            #         )
            #     except Exception as e:
            #         print(f"❌ Failed to initialize DSPy & GEPA detector: {e}")
            #         return False
            # elif provider == "samgemini":
            #     print(f"🟢 Using SAM-enhanced Gemini detector with model: {model}")
            #     try:
            #         detector = SAMGeminiDetection(
            #             model_name=model,
            #             object_to_detect=class_names[0],
            #             problem_statement=problem_statement,
            #             sample_images=sample_images,
            #             confidence_threshold=confidence_threshold,
            #             image_size=image_size,
            #             upscale_image=upscale_image,
            #             golden_examples=golden_examples or []
            #         )
            #     except Exception as e:
            #         print(f"❌ Failed to initialize SAM-enhanced Gemini detector: {e}")
            #         return False
            else:
                print(f"❌ Error: Unknown provider '{provider}' in model name '{model_name}'")
                return False

            image_files = []
            for file_path in timestamp_raw_path.rglob('*'):
                if file_path.suffix.lower() in self.image_extensions:
                    image_files.append(file_path)
            if not image_files:
                print(f"No image files found in {timestamp_raw_path}")
                return False
            print(f"Processing {len(image_files)} images for objects: {', '.join(class_names)}")
            
            batch_size = 10
            split_index = int(len(image_files) * 0.8)
            train_files = image_files[:split_index]
            val_files = image_files[split_index:]
            annotated_count = 0
            total_files = len(image_files)
            current_index = 0

            start_time = time.time()
            for batch_start in range(0, len(train_files), batch_size):
                if app_state.cancelled:
                    print("Annotation cancelled by user")
                    app_state.annotation_status = "cancelled"
                    app_state.annotation_progress.append("❌ Annotation cancelled by user")
                    app_state.current_timestamp = None
                    app_state.annotation_progress = []
                    app_state.annotation_total = 0
                    app_state.annotation_current = 0
                    app_state.cancelled = False
                    return False
                batch_files = train_files[batch_start:batch_start + batch_size]
                success = self._process_image_batch(
                    batch_files,
                    class_names,
                    detector,
                    images_train_path,
                    labels_train_path,
                    current_index + 1,
                    total_files,
                )
                annotated_count += sum(1 for _ in batch_files if success.get(_.name, False))
                current_index += len(batch_files)

            for batch_start in range(0, len(val_files), batch_size):
                if app_state.cancelled:
                    print("Annotation cancelled by user")
                    app_state.annotation_status = "cancelled"
                    app_state.annotation_progress.append("❌ Annotation cancelled by user")
                    app_state.current_timestamp = None
                    app_state.annotation_progress = []
                    app_state.annotation_total = 0
                    app_state.annotation_current = 0
                    app_state.cancelled = False
                    return False
                batch_files = val_files[batch_start:batch_start + batch_size]
                success = self._process_image_batch(
                    batch_files,
                    class_names,
                    detector,
                    images_val_path,
                    labels_val_path,
                    current_index + 1,
                    total_files,
                )
                annotated_count += sum(1 for _ in batch_files if success.get(_.name, False))
                current_index += len(batch_files)
            
            print(f"Successfully annotated {annotated_count}/{len(image_files)} images")
            print(f"Dataset saved to: {timestamp_annotated_path}")
            print(f"Total annotation time: {time.time() - start_time:.2f} seconds")
            
            return annotated_count > 0
                
        except KeyboardInterrupt:
            print(f"⚠️ Annotation interrupted by user (Ctrl+C), saving partial progress")
            self._create_dataset_yaml(timestamp_annotated_path, object_names)
            print(f"Successfully saved partial dataset with {current_index} images processed")
            return current_index > 0
        except Exception as e:
            print(f"Error processing timestamp folder {timestamp}: {e}")
            return False

    def _process_image_batch(self, batch_files: List[Path], class_names: Sequence[str], detector: 'BaseDetector', 
                            images_output_path: Path, labels_output_path: Path, 
                            batch_start_index: int, total_files: int) -> Dict[str, bool]:
        """
        Process a batch of images using parallel single-image API calls.
        """
        try:
            batch_start_time = time.time()
            print(f"🔍 Processing batch of {len(batch_files)} images in parallel")
            batch_success = {img_file.name: False for img_file in batch_files}
            images = []
            image_names = []
            for img_file in batch_files:
                try:
                    load_start = time.time()
                    image = Image.open(img_file).convert('RGB')
                    images.append(image)
                    image_names.append(img_file.name)
                    print(f"📏 Loaded image: {img_file.name}, size: {image.size}, time: {time.time() - load_start:.2f}s")
                except Exception as e:
                    print(f"❌ Failed to load image {img_file.name}: {e}")
                    if self.progress_callback:
                        self.progress_callback(img_file.name, False, batch_start_index, total_files)
                        batch_start_index += 1
                    batch_success[img_file.name] = False
                    continue

            def process_image(img_file, image, img_name):
                try:
                    api_start = time.time()
                    print(f"🔍 Processing {img_name} with {detector.model_name if hasattr(detector, 'model_name') else detector.__class__.__name__} detector...")
                    success, yolo_annotations, label = detector.detector(
                        image,
                        class_names,
                        class_ids=list(range(len(class_names)))
                    )
                    api_time = time.time() - api_start
                    if not success or not yolo_annotations:
                        print(f"❌ No detections for {img_name}, API time: {api_time:.2f}s")
                        return img_name, False
                    
                    print(f"✅ Collected {len(yolo_annotations)} annotations for {img_name}, API time: {api_time:.2f}s")
                    save_start = time.time()
                    output_image_path = images_output_path / f"{img_file.stem}.jpg"
                    image.save(output_image_path, "JPEG", quality=95)
                    print(f"💾 Saved image to: {output_image_path}, time: {time.time() - save_start:.2f}s")
                    output_label_path = labels_output_path / f"{img_file.stem}.txt"
                    with open(output_label_path, 'w') as f:
                        for annotation in yolo_annotations:
                            if len(annotation) == 6:
                                line = f"{annotation[0]} {annotation[1]:.6f} {annotation[2]:.6f} {annotation[3]:.6f} {annotation[4]:.6f} {annotation[5]:.6f}\n"
                            else:
                                line = f"{annotation[0]} {annotation[1]:.6f} {annotation[2]:.6f} {annotation[3]:.6f} {annotation[4]:.6f}\n"
                            f.write(line)
                    print(f"💾 Saved annotation to: {output_label_path}, time: {time.time() - save_start:.2f}s")
                    print(f"✅ Successfully processed: {img_name}")
                    return img_name, True
                except Exception as e:
                    print(f"❌ Error processing {img_name}: {e}, API time: {api_time:.2f}s" if 'api_time' in locals() else f"❌ Error processing {img_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    return img_name, False

            with ThreadPoolExecutor(max_workers=min(len(batch_files), 4)) as executor:
                futures = [
                    executor.submit(process_image, img_file, image, img_name)
                    for img_file, image, img_name in zip(batch_files, images, image_names)
                ]
                future_to_name = {f: name for f, name in zip(futures, image_names)}
                for future in as_completed(futures):
                    img_name = future_to_name.get(future, "<unknown>")
                    try:
                        result = future.result(timeout=30)
                        if isinstance(result, tuple) and len(result) == 2:
                            img_name, success = result
                        else:
                            success = bool(result)
                        batch_success[img_name] = success
                        if self.progress_callback:
                            self.progress_callback(img_name, success, batch_start_index, total_files)
                            batch_start_index += 1
                    except TimeoutError:
                        print(f"❌ Timeout processing image {img_name}")
                        batch_success[img_name] = False
                        if self.progress_callback:
                            self.progress_callback(img_name, False, batch_start_index, total_files)
                            batch_start_index += 1
                    except Exception as e:
                        print(f"❌ Exception in thread for image {img_name}: {e}")
                        batch_success[img_name] = False
                        if self.progress_callback:
                            self.progress_callback(img_name, False, batch_start_index, total_files)
                            batch_start_index += 1

            time.sleep(0.75)
            print(f"Batch completed, total time: {time.time() - batch_start_time:.2f}s")
            
            # Clean up EDSR model cache after each batch to prevent memory buildup
            try:
                from .enhance import cleanup_edsr_model
                cleanup_edsr_model()
            except ImportError:
                pass  # Enhancement not available
            
            return batch_success

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            for img_file in batch_files:
                if self.progress_callback:
                    self.progress_callback(img_file.name, False, batch_start_index, total_files)
                    batch_start_index += 1
            return {img_file.name: False for img_file in batch_files}
    
    def _create_dataset_yaml(self, dataset_path: Path, class_names: Sequence[str]):
        """
        Create dataset.yaml file for YOLO training.
        """
        yaml_content = {
            'train': "images/train",
            'val': "images/val",
            'nc': len(list(class_names)),
            'names': list(class_names)
        }
        
        yaml_file = dataset_path / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created dataset.yaml: {yaml_file}")
        print(f"Dataset YAML content:")
        print(f"  train: {yaml_content['train']}")
        print(f"  val: {yaml_content['val']}")
        print(f"  nc: {yaml_content['nc']}")
        print(f"  names: {yaml_content['names']}")
        
        with open(yaml_file, 'r') as f:
            print(f"Actual file content:")
            print(f.read())

def annotate_data(timestamp: str, object_names, progress_callback=None, confidence_threshold=0.8, model_name="gemini:gemini-2.5-flash-lite", image_size=1024, upscale_image=False, golden_examples=None) -> bool:
    """Annotate a dataset for one or multiple object names."""
    if isinstance(object_names, str):
        class_names = [name.strip() for name in object_names.split(',') if name.strip()]
    else:
        class_names = [str(name).strip() for name in object_names if str(name).strip()]

    annotator = DataAnnotator(model_name=model_name)
    annotator.progress_callback = progress_callback
    return annotator.process_timestamp_folder(timestamp, class_names, confidence_threshold=confidence_threshold, model_name=model_name, image_size=image_size, upscale_image=upscale_image, golden_examples=golden_examples)