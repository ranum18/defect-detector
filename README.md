🔍 AI-Powered Defect Detector
A complete computer vision pipeline that uses Transfer Learning (ResNet18) to detect hand-drawn defects in BMP images. This project includes an automated training pipeline and an interactive visualizer.

🚀 Key Features
Automated Training: Uses PyTorch to fine-tune a ResNet18 model on your custom BMP dataset.

Interactive Gallery: A desktop UI built with OpenCV to cycle through images using the spacebar.

CI/CD Integration: GitHub Actions automatically trains the model and runs tests on every push.

Hidden Image Validation: Evaluates model performance on a "secret" test set to ensure accuracy.

🛠️ Setup & Installation
Clone the repo:

Bash
git clone https://github.com/YOUR_USERNAME/defect-detector.git
cd defect-detector
Create a Virtual Environment:

Bash
python -m venv .venv

# Windows:

.venv\Scripts\activate

# Mac/Linux:

source .venv/bin/activate
Install Dependencies:

Bash
pip install -r requirements.txt
💻 How to Use

1. Training the AI
   To train the model on your images (3 clean, 6 defective), run:

Bash
python train_and_test.py
This will generate defect_model.pth. Note: This file is ignored by Git to keep the repository light.

2. Interactive Gallery
   To see the AI in action and cycle through the hidden test images:

Bash
python app.py
SPACEBAR: Next Image

ESC: Exit Program

## Author

Anum Rehman
