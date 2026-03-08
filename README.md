# HealthVision AI

**AI-Powered Health Detection System**

A comprehensive AI-based health screening platform that uses deep learning models to detect diseases from visual analysis of eyes, face/skin, and nails — developed as a final-year engineering project.

> **Department of Computer Science & Engineering (Artificial Intelligence)**
> **Govt. Engineering College, Munger**

---

## Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Endpoints](#api-endpoints)
- [Detection Modules](#detection-modules)
- [Model Training](#model-training)
- [Future Scope](#future-scope)
- [Team](#team)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Abstract

HealthVision AI is an intelligent health screening web application that leverages deep learning and computer vision techniques to assist in the early detection of diseases through visual analysis. The system provides three detection modules — **Eye/Jaundice Detection**, **Face/Skin Disease Detection**, and **Nail Disease Detection** — each powered by transfer learning models trained on medical image datasets.

The platform offers a user-friendly web interface that supports both image upload and live camera capture, processes images using advanced preprocessing techniques (bilateral filtering, CLAHE contrast normalization, Haar cascade detection), and delivers instant predictions with confidence scores, severity levels, and detailed medical information including causes, symptoms, and doctor recommendations.

---

## Features

- **Three Detection Modules** — Eye (Jaundice), Face (Skin Diseases), and Nail (Nail Diseases)
- **Deep Learning Models** — Transfer learning with EfficientNetB0 and MobileNetV2 architectures
- **Multiple Input Methods** — File upload (PNG, JPG, JPEG, JFIF, BMP, WebP) and live camera capture
- **Advanced Image Processing** — Haar cascade eye detection, bilateral filtering, CLAHE normalization
- **Comprehensive Results** — Disease prediction with confidence scores, top-3 predictions, severity levels
- **Detailed Health Reports** — Disease descriptions, causes, symptoms, and doctor recommendations
- **Printable Reports** — Generate and print health assessment reports
- **Demo Mode** — Fully functional UI demonstration when models are not loaded
- **Responsive Design** — Works across desktop and mobile devices
- **Production Ready** — Configured for deployment on Render with Gunicorn

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| **Backend** | Python, Flask, Flask-CORS |
| **Deep Learning** | TensorFlow 2.16.2, Keras |
| **Computer Vision** | OpenCV (Haar Cascades), Pillow |
| **Data Processing** | NumPy, Matplotlib, Seaborn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Camera Integration** | WebRTC, Canvas API |
| **Client-Server Communication** | Fetch API (AJAX) |
| **WSGI Server** | Gunicorn |
| **Deployment** | Render |
| **Model Format** | HDF5 (.h5) |
| **Version Control** | Git, GitHub |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (Browser)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Image Upload │  │ Camera Input │  │  UI/Results │ │
│  └──────┬──────┘  └──────┬───────┘  └─────▲──────┘ │
│         │                │                 │         │
│         └────────┬───────┘                 │         │
│                  │ (Fetch API / AJAX)       │         │
└──────────────────┼─────────────────────────┼─────────┘
                   │                         │
                   ▼                         │
┌──────────────────────────────────────────────────────┐
│                Flask Backend (app.py)                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │           Image Preprocessing                 │    │
│  │  • Bilateral Filtering (Denoising)            │    │
│  │  • CLAHE Contrast Normalization               │    │
│  │  • Haar Cascade Eye/Face Detection            │    │
│  │  • HSV/LAB Colorspace Analysis                │    │
│  │  • Resize to 224×224                          │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                 │
│  ┌──────────────────▼───────────────────────────┐    │
│  │           TensorFlow Models                   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │    │
│  │  │ Jaundice │ │   Face   │ │    Nail      │  │    │
│  │  │  Model   │ │  Model   │ │   Model      │  │    │
│  │  │  (.h5)   │ │  (.h5)   │ │   (.h5)      │  │    │
│  │  └──────────┘ └──────────┘ └──────────────┘  │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                 │
│  ┌──────────────────▼───────────────────────────┐    │
│  │        Disease Information Database           │    │
│  │  • Severity Levels   • Descriptions           │    │
│  │  • Causes            • Symptoms               │    │
│  │  • Recommendations                            │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
HealthVision-AI/
│
├── app.py                         # Main Flask application
├── requirements.txt               # Python dependencies
├── render.yaml                    # Render deployment configuration
├── .python-version                # Python version (3.11.9)
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation (this file)
│
├── templates/                     # HTML templates
│   ├── index.html                 # Landing page
│   ├── detect.html                # Detection interface
│   ├── documentation.html         # Documentation page
│   └── documentation_detailed.html # Detailed medical documentation
│
├── Face Dection/                  # Face model training
│   └── Face_dataset.ipynb         # Jupyter notebook for face model
│
├── Jaundice Dection/              # Jaundice model training
│   └── Jaundice.ipynb             # Jupyter notebook for jaundice model
│
├── Nail Dection/                  # Nail model training
│   └── nail_disease_code.ipynb    # Jupyter notebook for nail model
│
├── Team/                          # Team member profile images
│   ├── 01.png ... 05.png
│
├── face_model.h5                  # Pre-trained face disease model
├── jaundice_model.h5              # Pre-trained jaundice detection model
└── nail_model.h5                  # Pre-trained nail disease model
```

---

## Installation & Setup

### Prerequisites

- Python 3.11.9 or compatible version
- pip (Python package manager)
- Git

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/gauravssah/HealthVision-AI.git
   cd HealthVision-AI
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   # or
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

5. **Open in browser**

   Navigate to `http://localhost:5000`

### Dependencies

```
tensorflow==2.16.2
flask
flask-cors
Pillow
numpy
opencv-python-headless
gunicorn
```

---

## Usage Guide

### 1. Eye/Jaundice Detection

1. Navigate to the **Detection** page
2. Select the **Eye/Jaundice** tab
3. Upload an eye image or capture via camera
4. Click **Analyze** to get the prediction
5. View results including jaundice probability, eye crop, and health report

### 2. Face/Skin Disease Detection

1. Select the **Face/Skin** tab
2. Upload a face image or use camera capture
3. The system analyzes for: Acne, Eczema, Herpes, Panu (Tinea Versicolor), and Rosacea
4. View top-3 predictions with confidence scores

### 3. Nail Disease Detection

1. Select the **Nail** tab
2. Upload a nail image or use camera capture
3. The system analyzes for: Healthy, Onychomycosis, Psoriasis, and additional conditions
4. View detailed disease information and recommendations

### Supported Image Formats

PNG, JPG, JPEG, JFIF, BMP, WebP

---

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Landing page |
| `/detect` | GET | Detection interface |
| `/documentation` | GET | Documentation page |
| `/documentation/detailed` | GET | Detailed medical documentation |
| `/team/<filename>` | GET | Team member profile images |
| `/predict/eye` | POST | Eye/Jaundice detection — accepts image file, returns prediction with confidence and disease info |
| `/predict/face` | POST | Face/Skin disease detection — accepts image file, returns top-3 predictions with disease info |
| `/predict/nail` | POST | Nail disease detection — accepts image file, returns prediction with disease info |

### Example API Response (Eye Detection)

```json
{
  "prediction": "Normal",
  "confidence": 95.2,
  "eye_crop": "<base64_encoded_image>",
  "disease_info": {
    "severity": "healthy",
    "description": "No signs of jaundice detected.",
    "causes": "N/A",
    "symptoms": "N/A",
    "recommendation": "Continue regular health check-ups."
  }
}
```

---

## Detection Modules

### Module 1: Eye/Jaundice Detection

| Aspect | Details |
|--------|---------|
| **Model File** | `jaundice_model.h5` |
| **Architecture** | Transfer learning (CNN) |
| **Output** | Binary — Jaundice / Normal |
| **Preprocessing** | Bilateral filtering, CLAHE, Haar cascade eye detection, HSV/LAB colorspace analysis |
| **Input Size** | 224 × 224 pixels |

### Module 2: Face/Skin Disease Detection

| Aspect | Details |
|--------|---------|
| **Model File** | `face_model.h5` |
| **Architecture** | EfficientNetB0 (Transfer Learning) |
| **Classes** | Acne, Eczema, Herpes, Panu, Rosacea |
| **Preprocessing** | Bilateral filtering, CLAHE, EfficientNet preprocessing |
| **Input Size** | 224 × 224 pixels |

### Module 3: Nail Disease Detection

| Aspect | Details |
|--------|---------|
| **Model File** | `nail_model.h5` |
| **Architecture** | MobileNetV2 (Transfer Learning) |
| **Classes** | Healthy, Onychomycosis, Psoriasis (+ additional conditions) |
| **Preprocessing** | Bilateral filtering, CLAHE, custom normalization |
| **Input Size** | 224 × 224 pixels |

---

## Model Training

The training notebooks are available in the repository for reference:

| Module | Notebook | Location |
|--------|----------|----------|
| Face/Skin Detection | `Face_dataset.ipynb` | `Face Dection/` |
| Jaundice Detection | `Jaundice.ipynb` | `Jaundice Dection/` |
| Nail Disease Detection | `nail_disease_code.ipynb` | `Nail Dection/` |

### Training Approach

- **Transfer Learning** — Pre-trained models (EfficientNetB0, MobileNetV2) fine-tuned on medical image datasets
- **Data Augmentation** — Applied during training to improve model generalization
- **Optimizer** — Adam optimizer with learning rate scheduling
- **Loss Function** — Categorical cross-entropy for multi-class, binary cross-entropy for binary classification

---

## Future Scope

- **Additional Disease Modules** — Expand to other body parts and disease categories
- **Model Improvement** — Train on larger, more diverse medical datasets for higher accuracy
- **Mobile Application** — Develop native Android/iOS applications
- **Multi-Language Support** — Add support for regional languages
- **Patient History Tracking** — Allow users to maintain a health screening history
- **Doctor Integration** — Connect with healthcare professionals for follow-up consultations
- **Cloud-Based Model Serving** — Use TensorFlow Serving for scalable inference
- **Real-Time Video Analysis** — Continuous detection from live video feed

---

## Team

### Project Supervisor

| Name | Designation | Department |
|------|-------------|------------|
| **Dr. Saurabh Suman** | Assistant Professor | CSE, Govt. Engineering College, Munger |

### Student Developers

| Name | Roll No. | Branch |
|------|----------|--------|
| **Gaurav Kumar** | 23151144901 | CSE (Artificial Intelligence) |
| **Nitesh Kumar** | 22151144040 | CSE (Artificial Intelligence) |
| **Rupesh Kumar** | 22151144010 | CSE (Artificial Intelligence) |
| **Indrajeet Kumar** | 2315114490 | CSE (Artificial Intelligence) |

---

## Disclaimer

> ⚠️ **Medical Disclaimer:** HealthVision AI is an academic research project and screening tool only. It is **NOT** a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider for any medical concerns. The predictions provided by this system are indicative and should be verified by clinical examination.

---

## License

This project is developed as an academic project at **Govt. Engineering College, Munger** under the **Department of Computer Science & Engineering (Artificial Intelligence)**.

---

<p align="center">
  <b>HealthVision AI</b> — Empowering Health Through Artificial Intelligence
</p>
