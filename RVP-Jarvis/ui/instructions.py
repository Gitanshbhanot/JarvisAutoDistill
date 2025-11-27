import gradio as gr

def create_instructions_section(theme="dark"):
    """Create the How to use instructions section with improved UI"""
    
    # Theme-based colors
    if theme == "light":
        text_color = "#000000"
        bg_gradient_1 = "linear-gradient(135deg, #2563eb, #3b82f6)"
        bg_gradient_2 = "linear-gradient(135deg, #059669, #10b981)"
        bg_gradient_3 = "linear-gradient(135deg, #dc2626, #ef4444)"
        bg_gradient_4 = "linear-gradient(135deg, #7c2d12, #ea580c)"
        requirements_bg = "linear-gradient(135deg, #6b7280, #9ca3af)"
        border_color_1 = "#60a5fa"
        border_color_2 = "#34d399"
        border_color_3 = "#f87171"
        border_color_4 = "#fb923c"
        requirements_border = "#9ca3af"
        icon_color_1 = "#2563eb"
        icon_color_2 = "#059669"
        icon_color_3 = "#dc2626"
        icon_color_4 = "#7c2d12"
        requirements_icon = "#6b7280"
        step_bg_color = "#ffffff"
        checkmark_color = "#10b981"
    else:
        text_color = "#ffffff"
        bg_gradient_1 = "linear-gradient(135deg, #1e3a8a, #3b82f6)"
        bg_gradient_2 = "linear-gradient(135deg, #059669, #10b981)"
        bg_gradient_3 = "linear-gradient(135deg, #dc2626, #ef4444)"
        bg_gradient_4 = "linear-gradient(135deg, #7c2d12, #ea580c)"
        requirements_bg = "linear-gradient(135deg, #374151, #4b5563)"
        border_color_1 = "#60a5fa"
        border_color_2 = "#34d399"
        border_color_3 = "#f87171"
        border_color_4 = "#fb923c"
        requirements_border = "#6b7280"
        icon_color_1 = "#1e3a8a"
        icon_color_2 = "#059669"
        icon_color_3 = "#dc2626"
        icon_color_4 = "#7c2d12"
        requirements_icon = "#374151"
        step_bg_color = "#ffffff"
        checkmark_color = "#10b981"
    
    with gr.Accordion("How to use this platform", open=False):
        gr.HTML(f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: {text_color}; line-height: 1.5;">
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 20px 0;">
                
                <!-- Annotate Data Workflow -->
                <div style="background: {bg_gradient_1}; padding: 16px; border-radius: 12px; border: 1px solid {border_color_1}; box-shadow: 0 6px 20px rgba(59, 130, 246, 0.2); transition: all 0.3s ease; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: -50%; right: -50%; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%); pointer-events: none;"></div>
                    <div style="position: relative; z-index: 1;">
                        <div style="display: flex; align-items: center; margin-bottom: 14px;">
                            <div style="background: #ffffff; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M7 7H17V17H7V7Z" stroke="{icon_color_1}" stroke-width="2"/>
                                    <path d="M3 3L21 21" stroke="{icon_color_1}" stroke-width="1.5"/>
                                    <path d="M12 2V6" stroke="{icon_color_1}" stroke-width="1.5"/>
                                    <path d="M12 18V22" stroke="{icon_color_1}" stroke-width="1.5"/>
                                    <path d="M2 12H6" stroke="{icon_color_1}" stroke-width="1.5"/>
                                    <path d="M18 12H22" stroke="{icon_color_1}" stroke-width="1.5"/>
                                </svg>
                            </div>
                            <h3 style="color: {text_color}; margin: 0; font-size: 1.2rem; font-weight: 600;">Annotate Data</h3>
                        </div>
                        <div style="space-y: 10px;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_1}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">1</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Define your target object name</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_1}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">2</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Upload your image collection as ZIP</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_1}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">3</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Let AI automatically annotate images</span>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <span style="background: {step_bg_color}; color: {icon_color_1}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">4</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Move to training phase</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Fine-tune Model Workflow -->
                <div style="background: {bg_gradient_2}; padding: 16px; border-radius: 12px; border: 1px solid {border_color_2}; box-shadow: 0 6px 20px rgba(16, 185, 129, 0.2); transition: all 0.3s ease; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: -50%; right: -50%; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%); pointer-events: none;"></div>
                    <div style="position: relative; z-index: 1;">
                        <div style="display: flex; align-items: center; margin-bottom: 14px;">
                            <div style="background: #ffffff; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 2L13.09 8.26L19 9L13.09 9.74L12 16L10.91 9.74L5 9L10.91 8.26L12 2Z" stroke="{icon_color_2}" stroke-width="2"/>
                                    <path d="M19 15L20.09 18.26L23 19L20.09 19.74L19 23L17.91 19.74L15 19L17.91 18.26L19 15Z" stroke="{icon_color_2}" stroke-width="1.5"/>
                                    <path d="M5 6L6.09 9.26L9 10L6.09 10.74L5 14L3.91 10.74L1 10L3.91 9.26L5 6Z" stroke="{icon_color_2}" stroke-width="1.5"/>
                                </svg>
                            </div>
                            <h3 style="color: {text_color}; margin: 0; font-size: 1.2rem; font-weight: 600;">Fine-tune Model</h3>
                        </div>
                        <div style="space-y: 10px;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_2}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">1</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Select your annotated dataset</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_2}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">2</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Configure training parameters</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_2}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">3</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Train your custom YOLO model</span>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <span style="background: {step_bg_color}; color: {icon_color_2}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">4</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Download from JarvisDB</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Run Inference Workflow -->
                <div style="background: {bg_gradient_3}; padding: 16px; border-radius: 12px; border: 1px solid {border_color_3}; box-shadow: 0 6px 20px rgba(239, 68, 68, 0.2); transition: all 0.3s ease; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: -50%; right: -50%; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%); pointer-events: none;"></div>
                    <div style="position: relative; z-index: 1;">
                        <div style="display: flex; align-items: center; margin-bottom: 14px;">
                            <div style="background: #ffffff; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <circle cx="11" cy="11" r="8" stroke="{icon_color_3}" stroke-width="2"/>
                                    <path d="M21 21L16.65 16.65" stroke="{icon_color_3}" stroke-width="1.5"/>
                                    <circle cx="11" cy="11" r="3" stroke="{icon_color_3}" stroke-width="1.5"/>
                                </svg>
                            </div>
                            <h3 style="color: {text_color}; margin: 0; font-size: 1.2rem; font-weight: 600;">Run Inference</h3>
                        </div>
                        <div style="space-y: 10px;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_3}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">1</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Choose or upload your model</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_3}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">2</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Upload your test image</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_3}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">3</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Adjust confidence threshold</span>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <span style="background: {step_bg_color}; color: {icon_color_3}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">4</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Get instant detection results</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- JarvisDB Workflow -->
                <div style="background: {bg_gradient_4}; padding: 16px; border-radius: 12px; border: 1px solid {border_color_4}; box-shadow: 0 6px 20px rgba(234, 88, 12, 0.2); transition: all 0.3s ease; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: -50%; right: -50%; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%); pointer-events: none;"></div>
                    <div style="position: relative; z-index: 1;">
                        <div style="display: flex; align-items: center; margin-bottom: 14px;">
                            <div style="background: #ffffff; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="{icon_color_4}" stroke-width="2"/>
                                    <path d="M14 2V8H20" stroke="{icon_color_4}" stroke-width="1.5"/>
                                    <path d="M16 13H8" stroke="{icon_color_4}" stroke-width="1.5"/>
                                    <path d="M16 17H8" stroke="{icon_color_4}" stroke-width="1.5"/>
                                    <path d="M10 9H9H8" stroke="{icon_color_4}" stroke-width="1.5"/>
                                </svg>
                            </div>
                            <h3 style="color: {text_color}; margin: 0; font-size: 1.2rem; font-weight: 600;">JarvisDB</h3>
                        </div>
                        <div style="space-y: 10px;">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_4}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">1</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Browse all your datasets</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_4}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">2</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">View annotations and labels</span>
                            </div>
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="background: {step_bg_color}; color: {icon_color_4}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">3</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Manage trained models</span>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <span style="background: {step_bg_color}; color: {icon_color_4}; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; margin-right: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);">4</span>
                                <span style="color: {text_color}; font-weight: 500; font-size: 0.9rem;">Complete project overview</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Requirements Section -->
            <div style="background: {requirements_bg}; padding: 18px; border-radius: 12px; border: 1px solid {requirements_border}; margin-top: 20px; box-shadow: 0 6px 20px rgba(75, 85, 99, 0.2); position: relative; overflow: hidden;">
                <div style="position: absolute; top: -50%; right: -50%; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%); pointer-events: none;"></div>
                <div style="position: relative; z-index: 1;">
                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                        <div style="background: #ffffff; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; margin-right: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M9 12L11 14L15 10" stroke="{requirements_icon}" stroke-width="2"/>
                                <circle cx="12" cy="12" r="10" stroke="{requirements_icon}" stroke-width="1.5"/>
                            </svg>
                        </div>
                        <h3 style="color: {text_color}; margin: 0; font-size: 1.15rem; font-weight: 600;">System Requirements</h3>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px;">
                        <div style="display: flex; align-items: center; background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            <span style="color: {checkmark_color}; margin-right: 8px; font-size: 16px; font-weight: bold;">✓</span>
                            <span style="color: {text_color}; font-weight: 500; font-size: 0.85rem;">Images in JPG, PNG, JPEG formats</span>
                        </div>
                        <div style="display: flex; align-items: center; background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            <span style="color: {checkmark_color}; margin-right: 8px; font-size: 16px; font-weight: bold;">✓</span>
                            <span style="color: {text_color}; font-weight: 500; font-size: 0.85rem;">Gemini API credentials in .env file</span>
                        </div>
                        <div style="display: flex; align-items: center; background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            <span style="color: {checkmark_color}; margin-right: 8px; font-size: 16px; font-weight: bold;">✓</span>
                            <span style="color: {text_color}; font-weight: 500; font-size: 0.85rem;">Stable internet connection</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """) 