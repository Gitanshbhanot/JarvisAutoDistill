# 🤖 Jarvis - AI-Powered Object Detection System

<div align="center">

**An end-to-end platform for automated image annotation, custom model training, and object detection inference**

[![React](https://img.shields.io/badge/React-18.3.1-61DAFB?logo=react)](https://reactjs.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11-00FFFF)](https://github.com/ultralytics/ultralytics)
[![Gemini](https://img.shields.io/badge/Gemini-2.5-4285F4?logo=google)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup-rvp-jarvis)
  - [Frontend Setup](#frontend-setup-jarvis-fe)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🌟 Overview

**Jarvis** is a comprehensive AI-powered object detection platform that streamlines the entire machine learning workflow from data annotation to model deployment. It combines state-of-the-art vision models (Gemini, Grounding DINO, OWL-ViT) with YOLO fine-tuning capabilities to create custom object detection models without manual annotation.

### Key Capabilities

- 🏷️ **Automated Annotation**: Use AI models (Gemini 2.5, Grounding DINO, OWL-ViT) to automatically annotate images
- 🚀 **Custom Training**: Fine-tune YOLO models on your annotated datasets
- 🔍 **Real-time Inference**: Test models with adjustable confidence thresholds
- 🗄️ **Dataset Management**: Organize, view, and manage annotated datasets
- 📊 **Model Registry**: Track, download, and deploy trained models
- 🎨 **Modern UI**: Beautiful React-based interface with real-time progress tracking

---

## ✨ Features

### 🏷️ Auto-Annotation Pipeline
- **Multi-Model Support**: Choose from Gemini 2.5 (Flash Lite, Flash, Pro), Grounding DINO, OWL-ViT, and more
- **Advanced Techniques**: SAHI (Sliced Aided Hyper Inference), Two-Stage Detection, SAM integration
- **Flexible Configuration**: Adjustable confidence thresholds, problem statements, and sample images
- **Batch Processing**: Upload ZIP files with hundreds of images for automated annotation

### 🚀 Model Training
- **YOLO Fine-tuning**: Train custom YOLO11 models on annotated datasets
- **Configurable Parameters**: Adjust epochs, batch size, and other hyperparameters
- **Live Progress Tracking**: Real-time training status and metrics
- **GPU Acceleration**: Optimized for GPU servers with multi-worker support

### 🔍 Inference & Testing
- **Model Selection**: Use existing models or upload custom `.pt` files
- **Interactive Testing**: Upload images and see detection results instantly
- **Confidence Tuning**: Adjust detection thresholds in real-time
- **Detailed Results**: View bounding boxes, class labels, and confidence scores

### 🗄️ JarvisDB
- **Dataset Browser**: Explore all annotated datasets with image previews
- **Annotation Viewer**: Visualize bounding boxes and class labels
- **Model Management**: Download, organize, and delete trained models
- **Search & Filter**: Find datasets by class names or object types

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Jarvis Frontend (React)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Upload  │  │  Train   │  │ Inference│  │ JarvisDB │   │
│  │   Data   │  │  Models  │  │  Testing │  │  Browser │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ REST API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend API (Flask/Gradio)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Annotation Engine                                    │  │
│  │  • Gemini 2.5 (Flash Lite, Flash, Pro)              │  │
│  │  • Grounding DINO (Tiny, Base)                       │  │
│  │  • OWL-ViT (Base, Large)                            │  │
│  │  • SAHI, Two-Stage, SAM Integration                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Training Pipeline                                    │  │
│  │  • YOLO11 Fine-tuning                                │  │
│  │  • Custom Dataset Preparation                        │  │
│  │  • Model Versioning & Storage                        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Inference Engine                                     │  │
│  │  • YOLO Model Loading                                │  │
│  │  • Real-time Detection                               │  │
│  │  • Result Visualization                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Storage Layer                       │
│  • Annotated Datasets (YOLO format)                        │
│  • Trained Models (.pt files)                              │
│  • Training Runs & Metrics                                 │
│  • Image Database                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **CUDA** (optional, for GPU acceleration)
- **Gemini API Key** (for auto-annotation)

### Backend Setup (RVP-Jarvis)

1. **Navigate to backend directory**
   ```bash
   cd RVP-Jarvis
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   touch .env
   
   # Add your Gemini API key
   echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env
   ```

5. **Run the backend**
   
   **Option A: Development (Flask)**
   ```bash
   python3 api/server.py
   ```
   Access at: `http://localhost:8000`
   
   **Option B: Production (Gunicorn)**
   ```bash
   gunicorn -w 4 --timeout 600 --log-level debug api.server:app
   ```
   Access at: `http://localhost:8000`
   
   **Option C: Gradio UI**
   ```bash
   python3 app.py
   ```
   Access at: `http://localhost:7860`

### Frontend Setup (Jarvis-FE)

1. **Navigate to frontend directory**
   ```bash
   cd Jarvis-FE
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment (optional)**
   ```bash
   # Create .env file if using custom API URL
   echo 'VITE_JARVIS_API=http://localhost:8000' > .env
   ```

4. **Run development server**
   ```bash
   npm start
   # or
   npm run dev
   ```
   Access at: `http://localhost:3000` or `http://localhost:5173`

5. **Build for production**
   ```bash
   npm run build
   ```

---

## 📖 Usage Guide

### 1️⃣ Annotate Data

1. **Enter Object Name**: Specify what you want to detect (e.g., "car", "person", "building")
2. **Select AI Model**: Choose from:
   - **Gemini 2.5 Flash Lite** (Fastest, lowest cost)
   - **Gemini 2.5 Flash** (Balanced performance)
   - **Gemini 2.5 Pro** (Highest accuracy)
   - **Gemini 3 Pro** (Latest model)
   - **SAHI variants** (For small object detection)
   - **Two-Stage Detection** (Enhanced accuracy)
3. **Upload Images**: Upload a ZIP file containing your images (max 5GB)
4. **Set Confidence**: Adjust detection confidence threshold (0-1, default 0.8)
5. **Add Context** (Optional):
   - Problem statement for better annotations
   - Up to 3 sample images
6. **Start Annotation**: AI automatically annotates all images

### 2️⃣ Fine-tune Model

1. **Select Dataset**: Choose from previously annotated datasets
2. **Configure Training**:
   - Number of epochs (10-500, default 100)
   - Batch size (1-64, default 16)
3. **Start Training**: Fine-tune a custom YOLO11 model
4. **Monitor Progress**: View real-time training logs and metrics

### 3️⃣ Run Inference

1. **Choose Model Source**:
   - Use existing trained model from registry
   - Upload custom `.pt` model file
2. **Upload Test Image**: Select an image for detection
3. **Adjust Confidence**: Set detection threshold (0.1-1.0)
4. **Run Detection**: View results with bounding boxes and labels

### 4️⃣ JarvisDB

**Datasets Tab**
- Browse all annotated datasets
- View images with annotations
- Filter by class names
- Download datasets as ZIP files

**Models Tab**
- List all trained models
- View model metadata (classes, training date)
- Download model files
- Delete unused models

---

## 📁 Project Structure

```
Jarvis/
├── RVP-Jarvis/                 # Backend (Python/Flask/Gradio)
│   ├── annotate/               # Auto-annotation engine
│   │   ├── detectors/          # AI model implementations
│   │   │   ├── gemini.py       # Gemini 2.5 integration
│   │   │   ├── dino.py         # Grounding DINO
│   │   │   ├── owl.py          # OWL-ViT
│   │   │   ├── sahi.py         # SAHI implementation
│   │   │   └── ...
│   │   ├── main.py             # Annotation orchestrator
│   │   └── enhance.py          # Image enhancement
│   ├── core/                   # Core functionality
│   │   ├── data_processing.py  # Dataset management
│   │   ├── database.py         # Data persistence
│   │   ├── inference.py        # Model inference
│   │   ├── training.py         # YOLO training
│   │   └── state.py            # Application state
│   ├── api/                    # REST API
│   │   └── server.py           # Flask server
│   ├── ui/                     # Gradio UI components
│   │   ├── instructions.py
│   │   └── styles.py
│   ├── models/                 # Trained models storage
│   ├── data/                   # Datasets storage
│   ├── app.py                  # Gradio application
│   ├── requirements.txt        # Python dependencies
│   └── README.md
│
├── Jarvis-FE/                  # Frontend (React/Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Jarvis/         # Main application components
│   │   │   │   ├── Landing.jsx # Home page
│   │   │   │   ├── Data/       # Dataset management
│   │   │   │   │   ├── DataHome.jsx
│   │   │   │   │   └── DataDetail.jsx
│   │   │   │   ├── Model/      # Model management
│   │   │   │   │   ├── ModelHome.jsx
│   │   │   │   │   └── InferenceTest.jsx
│   │   │   │   └── Constants.js # Model configurations
│   │   │   ├── Login/          # Authentication
│   │   │   ├── NavBar.jsx      # Navigation
│   │   │   └── ErrorBoundary/  # Error handling
│   │   ├── api/                # API client
│   │   ├── App.js              # Main app component
│   │   ├── Route.js            # Routing configuration
│   │   └── index.js            # Entry point
│   ├── public/                 # Static assets
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite configuration
│   └── README.md
│
├── IOU_Test/                   # Testing utilities
└── README.md                   # This file
```

---

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask 3.0+ (REST API), Gradio 4.0+ (Web UI)
- **ML/AI**:
  - **YOLO**: Ultralytics 8.2+ (Object detection)
  - **Gemini**: Google GenAI 0.3+ (Vision LLM)
  - **Transformers**: HuggingFace 4.45+ (Grounding DINO, OWL-ViT)
  - **PyTorch**: 2.2+ (Deep learning framework)
- **Computer Vision**: OpenCV 4.8+, Pillow 11.2+
- **Data Processing**: NumPy 1.23+, Pandas 2.3+
- **Server**: Gunicorn 23.0+ (Production WSGI)

### Frontend
- **Framework**: React 18.3
- **Build Tool**: Vite 5.4
- **Routing**: React Router 6.14
- **UI Components**: Material-UI 5.16, Emotion
- **Styling**: TailwindCSS 3.4
- **Animations**: Framer Motion 12.12, Lottie React
- **Image Annotation**: @starwit/react-image-annotate
- **HTTP Client**: Axios 1.1
- **Analytics**: Mixpanel

### DevOps
- **Version Control**: Git
- **Package Management**: npm (frontend), pip (backend)
- **Environment**: dotenv (configuration)
- **Linting**: ESLint (frontend)

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Datasets

**Upload Dataset**
```http
POST /api/datasets/upload
Content-Type: multipart/form-data

{
  "file": <zip_file>,
  "object_name": "car",
  "model": "gemini:gemini-2.5-flash",
  "confidence": 0.8,
  "problem_statement": "Detect cars in parking lot",
  "samples": [<image1>, <image2>, <image3>]
}
```

**Get Datasets**
```http
GET /api/datasets
Response: [
  {
    "id": "dataset_123",
    "name": "car_detection",
    "object_name": "car",
    "image_count": 150,
    "status": "completed",
    "created_at": "2025-12-02T10:30:00Z"
  }
]
```

**Get Dataset Details**
```http
GET /api/datasets/{dataset_id}
Response: {
  "id": "dataset_123",
  "images": [...],
  "classes": ["car"],
  "annotations": [...]
}
```

#### Training

**Start Training**
```http
POST /api/train
Content-Type: application/json

{
  "dataset_id": "dataset_123",
  "epochs": 100,
  "batch_size": 16
}
```

**Get Training Status**
```http
GET /api/train/status/{training_id}
Response: {
  "status": "training",
  "progress": 45,
  "current_epoch": 45,
  "total_epochs": 100
}
```

#### Models

**Get Models**
```http
GET /api/models
Response: [
  {
    "id": "model_456",
    "name": "car_detector_v1",
    "classes": ["car"],
    "accuracy": 0.92,
    "created_at": "2025-12-02T12:00:00Z"
  }
]
```

**Run Inference**
```http
POST /api/inference
Content-Type: multipart/form-data

{
  "model_id": "model_456",
  "image": <image_file>,
  "confidence": 0.25
}

Response: {
  "detections": [
    {
      "class": "car",
      "confidence": 0.89,
      "bbox": [100, 150, 300, 400]
    }
  ],
  "image": <annotated_image_base64>
}
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow existing code style and conventions
- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'google.genai'`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem**: `GEMINI_API_KEY not found`
```bash
# Solution: Set environment variable
echo "GEMINI_API_KEY=your_key_here" >> .env
```

**Problem**: CUDA out of memory during training
```bash
# Solution: Reduce batch size
# In training parameters, set batch_size to 8 or 4
```

### Frontend Issues

**Problem**: API connection refused
```bash
# Solution: Ensure backend is running
cd RVP-Jarvis
python3 api/server.py
```

**Problem**: CORS errors
```bash
# Solution: Backend has CORS enabled by default
# Check VITE_JARVIS_API in .env matches backend URL
```

**Problem**: Annotation progress not updating
```bash
# Solution: Keep browser tab open
# Progress is polled every 2 seconds
# Check backend logs for API errors
```

### Common Issues

**Problem**: Slow annotation speed
- **Solution**: Use Gemini 2.5 Flash Lite for faster processing
- Consider using GPU server for better performance

**Problem**: Low detection accuracy
- **Solution**: 
  - Increase confidence threshold
  - Use Gemini 2.5 Pro for better accuracy
  - Provide problem statement and sample images
  - Try SAHI for small objects

**Problem**: Training fails
- **Solution**:
  - Ensure dataset has sufficient images (>50 recommended)
  - Check annotations are valid
  - Verify GPU/CPU resources available

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLO implementation
- **Google** for Gemini API
- **HuggingFace** for Transformers library
- **IDEA Research** for Grounding DINO
- **Meta** for Segment Anything Model (SAM)

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

---

<div align="center">

**Built with ❤️ by the Jarvis Team**

⭐ Star us on GitHub if you find this project useful!

</div>
