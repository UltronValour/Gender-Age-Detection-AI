# Gender & Age Detection AI

# 🔍 Overview
This is a real-time computer vision system that detects faces and predicts gender and age using OpenCV DNN, MediaPipe face alignment, and Caffe models, alongside a reactive Streamlit analytics dashboard.

# 🚀 Live Demo
[Gender & Age Detection AI Dashboard](https://ultronvalour-gender-age-detection-ai-app-ngbm2m.streamlit.app/)

# 🧠 Features
- Real-time face detection
- Age & gender prediction
- MediaPipe face alignment
- Temporal smoothing (anti-flicker)
- FPS & latency telemetry
- Confidence scoring
- CSV logging of predictions
- Batch image processing
- Interactive Streamlit dashboard
- Gender & age distribution charts
- Downloadable results

# 🏗️ Tech Stack
- Python
- OpenCV (DNN)
- MediaPipe
- Streamlit
- Caffe Models (Adience)
- Pandas / NumPy

AI Models Used:
- OpenCV DNN SSD face detector (TensorFlow)
- Caffe CNN models for age and gender prediction (Adience dataset)
- MediaPipe Face Mesh for facial alignment
- Temporal smoothing using centroid tracking and majority voting

# ⚙️ How It Works
Image/Webcam → Face Detection → Alignment → Age/Gender Inference → Smoothing → Telemetry → Dashboard → CSV Logging

# 📊 Performance
- Skip-frame optimization
- Real-time FPS tracking
- Average latency measurement
- CPU-friendly inference

# 🧪 Configuration
- Confidence threshold
- Alignment toggle
- Smoothing toggle
- Frame size
- Minimum face size

# 📂 Project Structure
```text
gender-age-detection/
│── app.py
│── requirements.txt
│── src/
│── logs/
│── outputs/
```

# ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

# ☁️ Deployment
This app is deployed natively on Streamlit Community Cloud and large neural-network models are automatically downloaded at runtime to bypass GitHub storage limits.

# ⚠️ Limitations
- Adience model predicts age ranges, not exact age
- Accuracy depends heavily on ambient lighting and camera image quality

# 📌 Future Improvements
- Emotion detection integration
- Model architecture upgrades (MobileNet/EfficientNet)
- REST API implementation

# 👨‍💻 Author
Valour Moraes
