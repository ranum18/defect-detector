import torch
import os
from app import model, transform
import cv2

def run_evaluation_logic():
    """
    Heads into the 'images/test' folder, runs the AI on everything,
    and returns the percentage of correct answers.
    """
    test_dir = "images/test"
    images = [f for f in os.listdir(test_dir) if f.endswith('.bmp')]
    
    correct = 0
    total = len(images)
    
    if total == 0:
        return 0

    model.eval()
    for img_name in images:
        # 1. Determine the 'True' label from the filename
        # (Assumes your test files are named 'clean_X.bmp' or 'defect_X.bmp')
        true_label = 1 if "defect" in img_name.lower() else 0
        
        # 2. Get the AI's prediction
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
            predicted_label = pred.item()

        # 3. Check if the AI was right
        if predicted_label == true_label:
            correct += 1
            
    return correct / total

def test_model_accuracy():
    # Now this function exists and can be called!
    accuracy = run_evaluation_logic()
    print(f"\n🎯 Model Accuracy on Hidden Images: {accuracy * 100}%")
    
    # Require at least 66% accuracy (4 out of 6 images) to pass the build
    assert accuracy >= 0.66, f"AI failed the test! Accuracy was only {accuracy*100}%"