## Jarvis Frontend

React UI for dataset management, auto-annotation, YOLO fine-tuning, and inference.

### Features

- **Datasets**: Upload ZIPs, provide problem statement, confidence threshold, and up to 3 example images. View datasets (annotated/raw), search, filter by classes, preview images, and download ZIPs.
- **Auto-annotation**: Starts after upload; progress is polled via API until completed/failed.
- **Training**: Fine-tune YOLO on an annotated dataset. UI shows live status and notifies on completion/failure.
- **Models**: List, test inference, download, and delete models. Heavy runs-directory cleanup is non-blocking.

### Prerequisites

- Node.js 18+
- A running backend API (Flask) at `http://localhost:8000` by default; configurable via `VITE_JARVIS_API`.

### Quick Start

1) Install dependencies
```bash
npm install
```

2) Configure environment (optional if using the default URL)
Create `.env` in this folder if you want a custom API URL:
```bash
echo 'VITE_JARVIS_API=http://localhost:8000' > .env
```

3) Run the dev server
```bash
npm start
```

4) Access dev server at
```bash
http://localhost:3000
```

5) Build for production
```bash
npm run build
```

### Running the backend (from repo root)

```bash
python3 -m venv RVP-Jarvis/venv
source RVP-Jarvis/venv/bin/activate
pip install -r RVP-Jarvis/requirements.txt
python RVP-Jarvis/api/server.py
```

The API listens on `http://0.0.0.0:8000` with CORS enabled. Adjust `VITE_JARVIS_API` if needed.

### Environment variables

- **VITE_JARVIS_API**: Base URL of the Jarvis API. Default: `http://localhost:8000`.

### Available scripts

- `npm run dev`: Start Vite dev server
- `npm run build`: Production build
- `npm run preview`: Preview the production build
- `npm run lint`: Lint the codebase

### Troubleshooting

- **APIs pending/hanging**: Ensure the backend is running. Model delete and cleanup operations run non-blocking; the API should return quickly while cleanup proceeds in the background.
- **CORS errors**: Verify the API URL and that Flask CORS is enabled (it is by default).
- **Annotation not progressing**: Keep the tab open; progress is polled every ~2s. Check backend logs for Gemini API limits/timeouts.

### Tech stack

- React 18, Vite 5, React Router, Material UI
