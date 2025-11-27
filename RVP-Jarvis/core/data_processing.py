import zipfile
import os
import time
import re
from pathlib import Path
from .state import app_state
from annotate.main import annotate_data
from PIL import Image
from .database import load_golden_set

def process_uploaded_zip(file, object_name, dataset_name=None, upscale_images=False):
    """
    Process uploaded zip file with pre-resizing for images.
    """
    if file is None:
        return "Please upload a zip file first.", ""
    
    if not object_name:
        return "Please enter an object name first.", ""
    
    if dataset_name and dataset_name.strip():
        safe_name = re.sub(r'[^\w\-_.]', '_', dataset_name.strip())
        if not safe_name:
            safe_name = str(int(time.time()))
    else:
        safe_name = str(int(time.time()))

    app_state.current_timestamp = None
    app_state.annotation_status = "pending"
    app_state.annotation_progress = []
    app_state.annotation_total = 0
    app_state.annotation_current = 0
    app_state.completed_timestamp = None
    
    app_state.current_timestamp = safe_name
    app_state.annotation_status = "pending"
    
    current_dataset_name = app_state.current_timestamp
    
    raw_data_dir = Path("data/raw_data")
    dataset_folder = raw_data_dir / current_dataset_name
    dataset_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = time.time()
        with zipfile.ZipFile(file, 'r') as zip_ref:
            file_list = []
            for file_path in zip_ref.namelist():
                if (not file_path.startswith('__MACOSX/') and 
                    not file_path.startswith('._') and 
                    not file_path.startswith('.DS_Store') and
                    not file_path.endswith('/') and
                    not file_path.startswith('.')):
                    file_list.append(file_path)
            
            if not file_list:
                return "❌ No valid files found in the zip archive (only system files detected)", ""
            
            extracted_files = []
            for file_path in file_list:
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    with zip_ref.open(file_path) as source:
                        img_start = time.time()
                        try:
                            image = Image.open(source).convert('RGB')
                            if upscale_images:
                                # Upscale to 2x resolution, max 2048x2048
                                new_size = (min(image.width * 2, 2048), min(image.height * 2, 2048))
                                image = image.resize(new_size, Image.Resampling.LANCZOS)
                                print(f"✅ Upscaled {file_path} to {new_size}")
                            output_path = dataset_folder / file_path
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            image.save(output_path, "JPEG", quality=95)
                            extracted_files.append(file_path)
                            print(f"✅ Extracted and resized {file_path}, time: {time.time() - img_start:.2f}s")
                        except Exception as e:
                            print(f"❌ Failed to process {file_path}: {e}")
                            continue
                else:
                    with zip_ref.open(file_path) as source:
                        output_path = dataset_folder / file_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(source.read())
                        extracted_files.append(file_path)
                        print(f"✅ Extracted {file_path}")
            
            time_taken = time.time() - start_time
            extracted_count = len(extracted_files)
            return (
                f"✅ Successfully extracted {extracted_count} files to {dataset_folder}\nTime taken: {time_taken:.2f}s",
                f"Extracted {extracted_count}/{len(file_list)} files"
            )
    except Exception as e:
        return f"❌ Error processing zip file: {str(e)}", ""
    
def get_current_status():
    """Get the current annotation status."""
    if app_state.annotation_status == "pending":
        return f"Waiting to start annotation..."
    elif app_state.annotation_status == "running":
        return f"Annotation in progress... ({app_state.annotation_current}/{app_state.annotation_total})"
    elif app_state.annotation_status == "completed":
        return f"Annotation completed! ({app_state.annotation_current}/{app_state.annotation_total})"
    elif app_state.annotation_status == "failed":
        return f"Annotation failed! ({app_state.annotation_current}/{app_state.annotation_total})"
    elif app_state.annotation_status == "cancelled":
        return f"Annotation cancelled! ({app_state.annotation_current}/{app_state.annotation_total})"
    return "Unknown status"

def start_annotation(object_name, confidence_threshold=0.8, model_name="gemini:gemini-2.5-flash-lite", image_size=1024, upscale_image=False, golden_examples=None):
    """
    Start annotation process.
    """
    print(f"🔍 start_annotation called with:")
    print(f"   object_name: {object_name}")
    print(f"   confidence_threshold: {confidence_threshold}")
    print(f"   model_name: {model_name}")
    print(f"   image_size: {image_size}")
    print(f"   upscale_image: {upscale_image}")
    print(f"   current_timestamp: {app_state.current_timestamp}")
    print(f"   annotation_status: {app_state.annotation_status}")
    
    if not app_state.current_timestamp:
        error_msg = "❌ No dataset uploaded. Please upload a zip file first."
        print(f"❌ {error_msg}")
        return error_msg, get_current_status()
    
    if not object_name:
        error_msg = "❌ Please enter an object name first."
        print(f"❌ {error_msg}")
        return error_msg, get_current_status()
    
    if app_state.annotation_status not in ["pending"]:
        error_msg = f"❌ Annotation already started or completed. Current status: {app_state.annotation_status}"
        print(f"❌ {error_msg}")
        return error_msg, get_current_status()
    
    app_state.annotation_status = "running"
    
    app_state.cancelled = False
    app_state.annotation_progress = []
    app_state.annotation_total = 0
    app_state.annotation_current = 0
    
    def progress_callback(image_name, success, current, total):
        """Callback to track annotation progress."""
        app_state.annotation_current = current
        app_state.annotation_total = total
        if success:
            app_state.annotation_progress.append(f"✅ {image_name}")
        else:
            app_state.annotation_progress.append(f"❌ {image_name}")
    
    try:
        # Load golden examples if not provided
        if golden_examples is None:
            golden = load_golden_set(app_state.current_timestamp)
            golden_list = []
            raw_data_path = Path("data/raw_data")
            for img_path, anns in golden.items():
                full_path = raw_data_path / app_state.current_timestamp / img_path
                if full_path.exists():
                    img = Image.open(full_path).convert('RGB')
                    golden_list.append((img, anns))
            golden_examples = golden_list
        
        print(f"🚀 Calling annotate_data with:")
        print(f"   timestamp: {app_state.current_timestamp}")
        print(f"   object_name: {object_name}")
        print(f"   model_name: {model_name}")
        print(f"   golden_examples: {len(golden_examples)} examples")
        
        success = annotate_data(app_state.current_timestamp, object_name, progress_callback, confidence_threshold=confidence_threshold, model_name=model_name, image_size=image_size, upscale_image=upscale_image, golden_examples=golden_examples)
        
        print(f"📊 annotate_data returned: {success}")
        print(f"📊 Final progress: {app_state.annotation_current}/{app_state.annotation_total}")
        
        app_state.completed_timestamp = time.time()
        if app_state.cancelled:
            progress_text = "\n".join(app_state.annotation_progress)
            result = f"❌ Annotation cancelled by user!\n\n📊 Progress ({app_state.annotation_current}/{app_state.annotation_total}):\n{progress_text}"
        elif success:
            app_state.annotation_status = "completed"
            progress_text = "\n".join(app_state.annotation_progress)
            result = f"✅ Annotation completed successfully!\n\n📊 Progress ({app_state.annotation_current}/{app_state.annotation_total}):\n{progress_text}"
        else:
            app_state.annotation_status = "failed"
            progress_text = "\n".join(app_state.annotation_progress)
            result = f"❌ Annotation failed!\n\n📊 Progress ({app_state.annotation_current}/{app_state.annotation_total}):\n{progress_text}"
        
        print(f"📄 Final result: {result}")
        return result, get_current_status()
    except Exception as e:
        app_state.completed_timestamp = time.time()
        if not app_state.cancelled:
            app_state.annotation_status = "failed"
        progress_text = "\n".join(app_state.annotation_progress)
        return f"❌ Annotation error: {str(e)}\n\n📊 Progress ({app_state.annotation_current}/{app_state.annotation_total}):\n{progress_text}", get_current_status()
