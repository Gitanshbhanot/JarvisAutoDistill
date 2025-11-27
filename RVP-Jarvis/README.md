Jarvis
Automatically annotate images and train custom models through a web UI.Use a GPU server for faster fine-tuning
🛠️ Installation
1. Clone the Repository
cd RVP-Jarvis

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

For production, also install Gunicorn:
pip install gunicorn

4. Set Up Environment Variables
Create a .env file in the project root:
touch .env

Add the following environment variables:
GEMINI_API_KEY=your_gemini_api_key_here

🚀 Running Jarvis
Start the Application
For development (single-threaded):
python3 api/server.py

For production (recommended, multi-worker):
gunicorn -w 4 --timeout 600 --log-level debug api.server:app

The application will be available at:

Local: http://localhost:8000 (Gunicorn) or http://localhost:7860 (Flask)

📖 How to Use Jarvis
1. 🏷️ Annotate Data

Enter Object Name: Type the object to detect (e.g., "car", "person", "building").
Select Gemini Model: Choose from Gemini 2.5 Flash Lite (fastest), Gemini 2.5 Flash (balanced), or Gemini 2.5 Pro (highest accuracy).
Upload Zip File: Upload a ZIP file containing images (max 5GB).
Set Confidence Threshold: Adjust detection confidence (0-1, default 0.8).
Add Problem Statement (optional): Provide context for better annotations.
Upload Samples (optional): Add up to 3 sample images.
Start Annotation: AI annotates images using the selected Gemini model.

2. 🚀 Fine-tune Model

Select Dataset: Choose an annotated dataset.
Set Parameters: Configure training epochs and batch size.
Start Training: Fine-tune a new model.

3. 🔍 Run Inference

Choose Model: Use an existing model or upload a .pt file.
Upload Image: Upload the test image.
Set Confidence: Adjust detection confidence threshold (0.1-1.0).
Run Detection: View results with bounding boxes and labels.

4. 🗄️ JarvisDB

Browse Datasets: View and explore annotated datasets.
View Annotations: See bounding boxes and labels on images.
Manage Models: Download and manage trained models.
