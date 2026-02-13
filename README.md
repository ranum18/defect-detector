# 🔍 PCB Defect Detection Suite

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ranum18-defect-detector.streamlit.app)

An end-to-end Computer Vision project featuring an automated training pipeline, a local inspection gallery, and a live Web UI. The system uses a **ResNet18** architecture to identify hand-drawn defects on 450x450px board images.

---

## 🌐 Live Web Interface

The project includes a **Streamlit** web application that allows users to upload their own images to test the model in real-time.

- **Upload:** Supports BMP, JPG, and PNG formats.
- **Processing:** Automatically resizes and normalizes any image size to $224 \times 224$ for the AI.
- **Side-by-Side Analysis:** View the original image and the AI's confidence score simultaneously.

---

## ⚙️ Automated Pipeline (CI/CD)

This project features a "Self-Healing" pipeline via GitHub Actions. Every push triggers:

1.  **Environment Setup:** Python 3.9 environment with `pip` caching for high-speed builds.
2.  **Automated Training:** The model is retrained from scratch (model weights are ignored by Git to keep the repo light).
3.  **Rigorous Testing:** `pytest` evaluates the new model against a hidden test set.
    - **Accuracy Requirement:** The build will fail if the model scores below **66%** on new images.

---

## 🛠️ Installation & Usage

### 1. Setup

```bash
# Clone the repository
git clone [https://github.com/ranum18/defect-detector.git](https://github.com/ranum18/defect-detector.git)
cd defect-detector

# Install dependencies
pip install -r requirements.txt
```
