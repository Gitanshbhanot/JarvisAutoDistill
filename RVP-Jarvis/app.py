import gradio as gr
import sys
import subprocess
import threading
import os
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import UI components
from ui.instructions import create_instructions_section
from ui.styles import get_app_styles, get_app_header

# Import core functionality
from core.data_processing import process_uploaded_zip, get_current_status, start_annotation
from core.training import get_available_annotated_datasets, train_model
from core.inference import (
    get_available_models_for_testing, get_model_classes_info, 
    run_model_inference, download_model, refresh_models
)
from core.database import (
    get_available_annotated_datasets_for_viewing, load_dataset_info, 
    view_annotated_image
)

# Create Gradio interface with custom CSS for compact single-column layout
with gr.Blocks(title="Jarvis - AI-Powered Object Detection System", theme=gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
), css="""
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 10px !important;
}
.block {
    max-width: 100% !important;
    margin-bottom: 10px !important;
    padding: 8px !important;
}
.theme-toggle-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 5px 0 15px 0;
}
.theme-toggle-switch {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border: none;
    border-radius: 25px;
    padding: 8px 20px;
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
}
.theme-toggle-switch:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
}
#theme-detector {
    display: none;
}
/* Reduce spacing and margins */
.gradio-row {
    margin: 8px 0 !important;
    gap: 10px !important;
}
.gradio-column {
    padding: 5px !important;
}
.gradio-group {
    margin-bottom: 15px !important;
    padding: 10px !important;
}
.gradio-accordion {
    margin-bottom: 15px !important;
}
.gradio-tabs {
    margin-top: 10px !important;
}
.gradio-tab-item {
    padding: 15px !important;
}
.gradio-textbox, .gradio-dropdown, .gradio-file, .gradio-number, .gradio-slider {
    margin-bottom: 8px !important;
}
.gradio-button {
    margin: 5px 0 !important;
}
.gradio-markdown {
    margin: 8px 0 !important;
    padding: 5px 0 !important;
}
/* Compact form elements */
.gradio-textbox label, .gradio-dropdown label, .gradio-file label {
    margin-bottom: 3px !important;
}
/* Terminal output styling */
#log_output textarea {
    font-family: 'Courier New', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.4 !important;
    background-color: #1e1e1e !important;
    color: #d4d4d4 !important;
    white-space: pre !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
}
#log_output {
    background-color: #1e1e1e !important;
}
.gradio-textbox:has(#log_output) {
    background-color: #1e1e1e !important;
}
""", js="""
function() {
    console.log('🎨 Starting system theme detection...');
    
    let detectedTheme = 'light'; // Default fallback
    
    try {
        // Method 1: Check prefers-color-scheme media query
        if (window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            const lightModeQuery = window.matchMedia('(prefers-color-scheme: light)');
            
            console.log('Dark mode query matches:', darkModeQuery.matches);
            console.log('Light mode query matches:', lightModeQuery.matches);
            
            if (darkModeQuery.matches) {
                detectedTheme = 'dark';
                console.log('✅ Detected DARK mode via media query');
            } else if (lightModeQuery.matches) {
                detectedTheme = 'light';
                console.log('✅ Detected LIGHT mode via media query');
            }
            
            // Listen for real-time changes
            darkModeQuery.addEventListener('change', (e) => {
                const newTheme = e.matches ? 'dark' : 'light';
                console.log('🔄 System theme changed to:', newTheme);
                window.detectedSystemTheme = newTheme;
                
                // Try to trigger theme update (if function exists)
                if (window.updateThemeFromSystem) {
                    window.updateThemeFromSystem(newTheme);
                }
            });
        }
        
        // Method 2: Check if we're in an iframe or special context
        if (window.self !== window.top) {
            console.log('📱 Running in iframe/embedded context');
        }
        
        // Method 3: Additional browser-specific checks
        if (navigator.userAgent.includes('Dark')) {
            detectedTheme = 'dark';
            console.log('✅ Detected DARK mode via user agent');
        }
        
        // Method 4: Check for dark mode class on html/body (some browsers/extensions add this)
        const htmlElement = document.documentElement;
        const bodyElement = document.body;
        
        if (htmlElement && (htmlElement.classList.contains('dark') || htmlElement.classList.contains('dark-mode'))) {
            detectedTheme = 'dark';
            console.log('✅ Detected DARK mode via HTML class');
        }
        
        if (bodyElement && (bodyElement.classList.contains('dark') || bodyElement.classList.contains('dark-mode'))) {
            detectedTheme = 'dark';
            console.log('✅ Detected DARK mode via body class');
        }
        
        // Method 5: Time-based fallback
        const hour = new Date().getHours();
        const timeBased = (hour >= 18 || hour < 6) ? 'dark' : 'light';
        console.log('⏰ Time-based suggestion:', timeBased, '(current hour:', hour + ')');
        
        // If no clear detection, use time-based as fallback
        if (!window.matchMedia || (!window.matchMedia('(prefers-color-scheme: dark)').matches && !window.matchMedia('(prefers-color-scheme: light)').matches)) {
            console.log('⚠️  No media query support or no preference set, using time-based fallback');
            detectedTheme = timeBased;
        }
        
    } catch (error) {
        console.error('❌ Error in theme detection:', error);
        detectedTheme = 'light'; // Safe fallback
    }
    
    // Store globally for access
    window.detectedSystemTheme = detectedTheme;
    
    // Try to communicate with Python side
    try {
        // Create a hidden element to store the theme
        let themeElement = document.getElementById('detected-theme-value');
        if (!themeElement) {
            themeElement = document.createElement('div');
            themeElement.id = 'detected-theme-value';
            themeElement.style.display = 'none';
            themeElement.setAttribute('data-theme', detectedTheme);
            document.body.appendChild(themeElement);
        } else {
            themeElement.setAttribute('data-theme', detectedTheme);
        }
        
        console.log('💾 Stored detected theme in DOM:', detectedTheme);
    } catch (error) {
        console.error('❌ Error storing theme:', error);
    }
    
    console.log('🎯 Final detected theme:', detectedTheme);
    return detectedTheme;
}
""") as demo:
    
    # Global state for models and theme
    available_models_state = gr.State([])
    model_data_state = gr.State([])
    theme_state = gr.State("light")  # Default to light, will be updated
    
    # Hidden textbox to receive JS theme detection result
    theme_detector = gr.Textbox(value="light", visible=False, elem_id="theme-detector")
    
    # Create initial content with light theme (will be updated)
    app_styles = gr.HTML(get_app_styles("light"))
    app_header = gr.HTML(get_app_header("light"))
    
    # Theme toggle at the top center
    with gr.Row(elem_classes=["theme-toggle-container"]):
        theme_toggle_btn = gr.Button("🌙 Switch to Dark Mode", variant="primary", size="sm", elem_classes=["theme-toggle-switch"])
    
    # Create instructions section initially
    instructions_group = gr.Group()
    with instructions_group:
        create_instructions_section("light")

    
    # Main workflow tabs
    with gr.Tabs():
        
        # Tab 1: Annotate Data
        with gr.TabItem("🏷️ Annotate Data"):
            gr.Markdown("""
            ### Create annotated datasets for object detection
            Upload your images, specify what object to detect, and let AI automatically annotate your images.
            """)
            
            object_name_input = gr.Textbox(
                label="Object to Annotate", 
                placeholder="Enter object name (e.g., car, person, building)"
            )
            gr.Markdown("*What object would you like to detect?*")
            
            zip_file_input = gr.File(
                label="Upload Zip File",
                type="filepath",
                file_types=[".zip"]
            )
            gr.Markdown("*Upload a zip file containing your images*")
            
            process_zip_btn = gr.Button("📁 Process Zip File", variant="primary", size="lg")
            
            current_status = gr.Markdown(value="No active dataset", label="📊 Current Dataset Status")
            
            start_annotation_btn = gr.Button("🏷️ Start Annotation", variant="secondary", size="lg")
            
            go_to_finetune_btn = gr.Button("🚀 Go to Fine-tune Model", variant="primary", size="lg")
            
            refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
        
        # Tab 2: Fine-tune Model
        with gr.TabItem("🚀 Fine-tune Model"):
            gr.Markdown("""
            ### Train and fine-tune object detection models
            Select an annotated dataset and train a custom YOLO model with your preferred parameters.
            """)
            
            dataset_dropdown = gr.Dropdown(
                choices=get_available_annotated_datasets(),
                label="Select Dataset"
            )
            gr.Markdown("*Choose from previously annotated datasets*")
            
            with gr.Row():
                epochs_input = gr.Number(
                    value=100, 
                    minimum=10, 
                    maximum=500, 
                    label="Number of Epochs"
                )
                batch_size_input = gr.Number(
                    value=16, 
                    minimum=1, 
                    maximum=64, 
                    label="Batch Size"
                )
            
            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
            training_output = gr.Textbox(label="📋 Training Output", lines=8, interactive=False)
        
        # Tab 3: Run Inference
        with gr.TabItem("🔍 Run Inference"):
            gr.Markdown("""
            ### Test your trained models on new images
            Upload an image and see how your model performs with adjustable confidence threshold.
            """)
            
            model_source_radio = gr.Radio(
                choices=["📁 Use existing model", "📤 Upload model file"],
                value="📁 Use existing model",
                label="Model Source"
            )
            
            # Existing model selection
            model_dropdown = gr.Dropdown(
                label="Select Model",
                visible=True,
                choices=[]  # Start empty, will be populated by load_initial_data
            )
            
            # Upload model file
            model_file_input = gr.File(
                label="Upload Model File (.pt)",
                type="binary",
                file_types=[".pt"],
                visible=False
            )
            
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )
            
            model_classes_info = gr.Markdown(value="No model selected", label="Model Information")
            
            refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
            
            test_image_input = gr.Image(
                label="Upload Test Image",
                type="filepath"
            )
            
            run_detection_btn = gr.Button("🔍 Run Detection", variant="primary", size="lg")
            
            detection_output_image = gr.Image(label="🎯 Detection Results")
            
            detection_info = gr.Textbox(label="📋 Detection Details", lines=8, interactive=False)
        
        # Tab 4: JarvisDB
        with gr.TabItem("🗄️ JarvisDB"):
            gr.Markdown("""
            ### Database of all your datasets and models
            Browse, view, and manage your annotated datasets and trained models.
            """)
            
            with gr.Tabs():
                # Sub-tab: Datasets
                with gr.TabItem("📊 Datasets"):
                    gr.Markdown("### Browse your annotated datasets")
                    
                    view_dataset_dropdown = gr.Dropdown(
                        label="Select Dataset",
                        choices=get_available_annotated_datasets_for_viewing(),
                        value=get_available_annotated_datasets_for_viewing()[0] if get_available_annotated_datasets_for_viewing() else None
                    )
                    gr.Markdown("*Choose an annotated dataset to view*")
                    
                    # Initialize with images from the first dataset if available
                    initial_images = []
                    initial_class_names = []
                    if get_available_annotated_datasets_for_viewing():
                        initial_dataset = get_available_annotated_datasets_for_viewing()[0]
                        initial_images, initial_class_names = load_dataset_info(initial_dataset)
                    
                    view_image_dropdown = gr.Dropdown(
                        label="Select Image",
                        choices=initial_images,
                        value=initial_images[0] if initial_images else None
                    )
                    gr.Markdown("*Choose an image from the dataset*")
                    
                    view_image_btn = gr.Button("👁️ View Annotated Image", variant="primary", size="lg")
                    
                    annotated_image_output = gr.Image(label="🎯 Annotated Image")
                    image_info_output = gr.Markdown(value="Select a dataset and image to view annotations", label="Image Information")
                
                # Sub-tab: Models
                with gr.TabItem("🤖 Models"):
                    gr.Markdown("### Manage your trained models")
                    
                    download_model_dropdown = gr.Dropdown(
                        label="Select Model to Download",
                        choices=[]  # Start empty, will be populated by load_initial_data
                    )
                    gr.Markdown("*Choose a model to download*")
                    
                    download_model_btn = gr.Button("📥 Download Model", variant="primary", size="lg")
                    
                    download_file = gr.File(label="Downloaded Model", interactive=False)
                    download_info = gr.Markdown(value="Select a model to download", label="Download Information")
        
        # Tab 5: Logs
        with gr.TabItem("📋 Logs"):
            gr.Markdown("""
            ### Deployment Logs
            View terminal output from running the deployment update script.
            """)
            
            with gr.Row():
                run_deployment_btn = gr.Button("🚀 Run Deployment Script", variant="primary", size="lg")
                clear_logs_btn = gr.Button("🗑️ Clear Logs", variant="secondary")
            
            log_output = gr.Textbox(
                label="📋 Deployment Output",
                lines=20,
                max_lines=50,
                interactive=False,
                placeholder="Click 'Run Deployment Script' to see output...",
                show_copy_button=True,
                elem_id="log_output"
            )
            
            deployment_status = gr.Markdown(value="Ready to run deployment", label="Status")

    
    # Deployment script execution functions
    def run_deployment_script():
        """Execute the deployment script and capture its output"""
        script_path = "/home/gautam/diy_hanoon/RVP-Jarvis/update_deployment.sh"
        
        # Check if script exists
        if not os.path.exists(script_path):
            error_msg = f"❌ Deployment script not found at: {script_path}"
            return error_msg, "❌ Script not found"
        
        try:
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Show the command being executed (like in terminal)
            command_info = f"$ bash {script_path}\n"
            yield command_info, "🔄 Running..."
            
            # Set environment variables to preserve colors and formatting
            env = os.environ.copy()
            env['TERM'] = 'xterm-256color'
            env['FORCE_COLOR'] = '1'
            env['CLICOLOR_FORCE'] = '1'
            env['COLUMNS'] = '120'  # Set terminal width
            env['LINES'] = '30'     # Set terminal height
            
            # Run the script and capture ALL output exactly as it would appear in terminal
            process = subprocess.Popen(
                ["/bin/bash", "-c", f"cd /home/gautam/diy_hanoon/RVP-Jarvis && bash {script_path}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=0,  # Unbuffered for real-time output
                env=env
            )
            
            output_lines = [command_info.rstrip()]
            
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    # Handle character by character to preserve real-time output
                    if char == '\n':
                        output_lines.append('')
                    else:
                        if not output_lines:
                            output_lines.append('')
                        output_lines[-1] += char
                    
                    # Yield current complete output every few characters for performance
                    if len(output_lines[-1]) % 10 == 0 or char == '\n':
                        current_output = "\n".join(output_lines)
                        yield current_output, "🔄 Running..."
            
            # Wait for process to complete and get final status
            return_code = process.wait()
            
            # Add final status to output
            if return_code == 0:
                output_lines.append(f"\n✅ Process completed with exit code: {return_code}")
                final_status = "✅ Deployment completed successfully!"
            else:
                output_lines.append(f"\n❌ Process failed with exit code: {return_code}")
                final_status = f"❌ Deployment failed with exit code {return_code}"
            
            final_output = "\n".join(output_lines)
            return final_output, final_status
            
        except Exception as e:
            error_msg = f"❌ Error running deployment script:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg, "❌ Error occurred"
    
    def clear_deployment_logs():
        """Clear the deployment logs"""
        return "", "Ready to run deployment"
    
    # Enhanced system theme detection and initialization
    def initialize_theme_from_system():
        """Initialize theme based on enhanced system preference detection"""
        import time
        from datetime import datetime
        
        # Allow more time for JS to execute and DOM to be ready
        time.sleep(0.3)
        
        # Multiple detection strategies with priority order
        detected_theme = "light"  # Safe default
        
        try:
            # Strategy 1: Time-based detection (works as immediate fallback)
            current_hour = datetime.now().hour
            time_based_theme = "dark" if (18 <= current_hour or current_hour < 6) else "light"
            
            # Strategy 2: User agent detection (some browsers include theme info)
            import os
            user_os = os.name
            
            # Strategy 3: Environment variables (Linux/Unix systems)
            desktop_session = os.environ.get('DESKTOP_SESSION', '').lower()
            xdg_current_desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
            
            # Check for dark mode indicators in environment
            dark_indicators = ['dark', 'gnome-dark', 'kde-dark', 'ubuntu-dark']
            if any(indicator in desktop_session for indicator in dark_indicators) or \
               any(indicator in xdg_current_desktop for indicator in dark_indicators):
                detected_theme = "dark"
                print(f"🐧 Detected DARK mode from desktop environment: {desktop_session or xdg_current_desktop}")
            
            # Strategy 4: Time-based as fallback
            else:
                detected_theme = time_based_theme
                print(f"⏰ Using time-based detection: {detected_theme} (hour: {current_hour})")
            
        except Exception as e:
            print(f"❌ Error in theme detection: {e}")
            detected_theme = "light"  # Safe fallback
        
        # Update button text based on detected theme
        if detected_theme == "light":
            button_text = "🌙 Switch to Dark Mode"
        else:
            button_text = "☀️ Switch to Light Mode"
        
        print(f"🎨 Auto-detected theme: {detected_theme}")
        
        return (
            detected_theme,  # theme_state
            get_app_styles(detected_theme),  # app_styles
            get_app_header(detected_theme),  # app_header
            button_text  # theme_toggle_btn
        )
    
    # Manual theme detection trigger (for testing)
    def detect_theme_manual():
        """Manual theme detection for debugging"""
        import time
        from datetime import datetime
        
        current_hour = datetime.now().hour
        
        # Simple logic: dark mode from 6 PM to 6 AM
        if 18 <= current_hour or current_hour < 6:
            detected_theme = "dark"
            button_text = "☀️ Switch to Light Mode"
        else:
            detected_theme = "light"
            button_text = "🌙 Switch to Dark Mode"
        
        print(f"🔍 Manual detection: {detected_theme} (time: {current_hour}:00)")
        
        return (
            detected_theme,
            get_app_styles(detected_theme),
            get_app_header(detected_theme),
            button_text
        )
    
    # Theme toggle functionality
    def toggle_theme(current_theme):
        """Toggle between dark and light themes"""
        new_theme = "light" if current_theme == "dark" else "dark"
        
        # Update button text and icon
        if new_theme == "light":
            button_text = "🌙 Switch to Dark Mode"
        else:
            button_text = "☀️ Switch to Light Mode"
        
        print(f"🎨 Theme switched to: {new_theme}")
        
        # Return updated values
        return (
            new_theme,  # theme_state
            get_app_styles(new_theme),  # app_styles
            get_app_header(new_theme),  # app_header
            button_text  # theme_toggle_btn
        )
    
    # Initialize theme on app load with delay
    def delayed_theme_init():
        """Delayed theme initialization to allow JS execution"""
        import time
        time.sleep(0.5)  # Longer delay for better JS detection
        return initialize_theme_from_system()
    
    # Multiple initialization attempts
    demo.load(
        delayed_theme_init,
        outputs=[theme_state, app_styles, app_header, theme_toggle_btn]
    )
    
    theme_toggle_btn.click(
        toggle_theme,
        inputs=[theme_state],
        outputs=[theme_state, app_styles, app_header, theme_toggle_btn]
    )

    
    # Annotate workflow events
    process_zip_btn.click(
        lambda zip_file, object_name: process_uploaded_zip(zip_file, object_name)[1],  # Only return status
        inputs=[zip_file_input, object_name_input],
        outputs=[current_status]
    )
    
    start_annotation_btn.click(
        lambda object_name: start_annotation(object_name)[1],  # Only return status
        inputs=[object_name_input],
        outputs=[current_status]
    )
    
    refresh_status_btn.click(
        lambda: get_current_status(),
        outputs=current_status
    )
    
    # Fine-tune workflow events
    train_btn.click(
        train_model,
        inputs=[dataset_dropdown, epochs_input, batch_size_input],
        outputs=training_output
    )
    
    # Run Inference workflow events
    def update_model_interface(source):
        print(f"🔄 Updating model interface for source: {source}")
        if source == "📁 Use existing model":
            models, model_data = get_available_models_for_testing()
            print(f"  - Models found: {models}")
            print(f"  - Model data count: {len(model_data)}")
            return (
                gr.update(visible=True, choices=models, value=models[0] if models else None),
                gr.update(visible=False),
                model_data
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                []
            )
    
    model_source_radio.change(
        update_model_interface,
        inputs=[model_source_radio],
        outputs=[model_dropdown, model_file_input, model_data_state]
    )
    
    # Load initial model data
    def load_initial_data():
        models, model_data = get_available_models_for_testing()
        datasets = get_available_annotated_datasets()
        view_datasets = get_available_annotated_datasets_for_viewing()
        print(f"🔄 Loading initial data:")
        print(f"  - Models: {models}")
        print(f"  - Model data count: {len(model_data)}")
        print(f"  - Datasets: {datasets}")
        print(f"  - View datasets: {view_datasets}")
        # Return gr.update objects for dropdowns
        model_update = gr.update(choices=models, value=models[0] if models else None)
        dataset_update = gr.update(choices=datasets, value=datasets[0] if datasets else None)
        download_update = gr.update(choices=models, value=models[0] if models else None)
        view_dataset_update = gr.update(choices=view_datasets, value=view_datasets[0] if view_datasets else None)
        return models, model_data, model_update, dataset_update, download_update, view_dataset_update
    
    demo.load(
        load_initial_data,
        outputs=[available_models_state, model_data_state, model_dropdown, dataset_dropdown, download_model_dropdown, view_dataset_dropdown]
    )
    
    model_dropdown.change(
        get_model_classes_info,
        inputs=[model_dropdown, model_data_state],
        outputs=model_classes_info
    )
    
    run_detection_btn.click(
        run_model_inference,
        inputs=[model_dropdown, model_data_state, test_image_input, confidence_slider, model_file_input],
        outputs=[detection_output_image, detection_info]
    )
    
    # Refresh models button
    refresh_models_btn.click(
        refresh_models,
        outputs=[available_models_state, model_data_state, model_dropdown, download_model_dropdown]
    )
    
    # JarvisDB workflow events
    # Global state for class names
    class_names_state = gr.State([])
    
    def update_image_choices(dataset_name):
        """Update image choices when dataset changes"""
        if not dataset_name:
            return gr.update(choices=[], value=None), []
        
        images, class_names = load_dataset_info(dataset_name)
        return gr.update(choices=images, value=images[0] if images else None), class_names
    
    view_dataset_dropdown.change(
        update_image_choices,
        inputs=[view_dataset_dropdown],
        outputs=[view_image_dropdown, class_names_state]
    )
    
    view_image_btn.click(
        view_annotated_image,
        inputs=[view_dataset_dropdown, view_image_dropdown, class_names_state],
        outputs=[annotated_image_output, image_info_output]
    )
    
    # Download workflow events
    download_model_btn.click(
        download_model,
        inputs=[download_model_dropdown, model_data_state],
        outputs=download_file
    )
    
    # Deployment logs workflow events
    run_deployment_btn.click(
        run_deployment_script,
        outputs=[log_output, deployment_status]
    )
    
    clear_logs_btn.click(
        clear_deployment_logs,
        outputs=[log_output, deployment_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False) 