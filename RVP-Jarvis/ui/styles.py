import gradio as gr

def get_app_styles(theme="dark"):
    """Return the CSS styles for the Jarvis app based on theme"""
    
    if theme == "light":
        return get_light_theme_styles()
    else:
        return get_dark_theme_styles()

def get_dark_theme_styles():
    """Return dark theme CSS styles"""
    return """
    <style>
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        .gradio-interface {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        button, input, textarea, select {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        /* Dark theme - black backgrounds, white text */
        .gradio-container {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-interface {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-block {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-input, .gradio-output {
            background-color: #111111 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
        }
        .gradio-button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ffffff !important;
        }
        .gradio-button:hover {
            background-color: #cccccc !important;
            color: #000000 !important;
        }
        .gradio-dropdown, .gradio-textbox, .gradio-number, .gradio-slider {
            background-color: #111111 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
        }
        .gradio-tab-nav {
            background-color: #000000 !important;
            color: #ffffff !important;
            border-bottom: 1px solid #ffffff !important;
        }
        .gradio-tab-nav button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
        }
        .gradio-tab-nav button.selected {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-markdown {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-accordion {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
        }
        .gradio-tabs {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-row, .gradio-column {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-file {
            background-color: #111111 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
        }
        .gradio-image {
            background-color: #111111 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
        }
        .gradio-radio {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .gradio-state {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
    </style>
    """

def get_light_theme_styles():
    """Return light theme CSS styles"""
    return """
    <style>
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        .gradio-interface {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        button, input, textarea, select {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        /* Light theme - white backgrounds, dark text */
        .gradio-container {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-interface {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-block {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-input, .gradio-output {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6 !important;
        }
        .gradio-button {
            background-color: #007bff !important;
            color: #ffffff !important;
            border: 1px solid #007bff !important;
        }
        .gradio-button:hover {
            background-color: #0056b3 !important;
            color: #ffffff !important;
        }
        .gradio-dropdown, .gradio-textbox, .gradio-number, .gradio-slider {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6 !important;
        }
        .gradio-tab-nav {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-bottom: 1px solid #dee2e6 !important;
        }
        .gradio-tab-nav button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #dee2e6 !important;
        }
        .gradio-tab-nav button.selected {
            background-color: #007bff !important;
            color: #ffffff !important;
        }
        .gradio-markdown {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-accordion {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #dee2e6 !important;
        }
        .gradio-tabs {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-row, .gradio-column {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-file {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6 !important;
        }
        .gradio-image {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6 !important;
        }
        .gradio-radio {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .gradio-state {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
    </style>
    """

def get_app_header(theme="dark"):
    """Return the HTML header for the Jarvis app based on theme"""
    text_color = "#ffffff" if theme == "dark" else "#000000"
    
    return f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: {text_color} !important; font-size: 2.5rem; margin-bottom: 0.5rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;">Jarvis</h1>
        <p style="color: {text_color} !important; font-size: 1.2rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;">Annotate and fine tune models automatically</p>
    </div>
    """ 