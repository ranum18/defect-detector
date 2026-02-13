import cv2
import torch
import os
from torchvision import models, transforms

# 1. Load the Model Architecture
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("defect_model.pth"))
model.eval()

# 2. Setup Image Transformation (Must match training!)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import os

# ... (rest of your imports and model loading code)

def run_ml_gallery(folder_path):
    is_github = os.getenv('GITHUB_ACTIONS') == 'true'
    images = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
    
    if not images:
        print(f"No images found in {folder_path}")
        return

    # Create the named window once, outside the loop
    if not is_github:
        cv2.namedWindow("ML Defect Detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ML Defect Detector", 600, 600)

    idx = 0
    while True:
        img_name = images[idx]
        img_path = os.path.join(folder_path, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            idx = (idx + 1) % len(images)
            continue

        # AI Prediction Logic
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
            label = "DEFECT" if pred.item() == 1 else "CLEAN"

        # UI Overlay
        color = (0, 0, 255) if label == "DEFECT" else (0, 255, 0)
        cv2.rectangle(img, (0,0), (450, 80), (255, 255, 255), -1) # White header bar
        cv2.putText(img, f"File: {img_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(img, f"RESULT: {label}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        if not is_github:
            # Re-use the same window name
            cv2.imshow("ML Defect Detector", img)
            
            # waitKey(0) pauses execution until a key is pressed
            key = cv2.waitKey(0) & 0xFF 
            
            if key == 27:    # ESC key
                print("Exiting...")
                break
            elif key == 32:  # Spacebar
                idx = (idx + 1) % len(images)
        else:
            print(f"Image: {img_name} | AI Says: {label}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ml_gallery("images/test") # Pointing to your 6 hidden images