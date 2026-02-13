import cv2
import torch
from torchvision import transforms
from model import DefectModel

def detect_defect(image_path):
    # 1. Load Image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 2. Pre-process for PyTorch
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img)

    # 3. Run Inference
    detector = DefectModel()
    results = detector.predict(img_tensor)

    # 4. Filter results (COCO class 'bottle' is index 44, often used for testing)
    # For a real defect detector, you'd use your own custom class IDs
    found_objects = []
    for label, score in zip(results['labels'], results['scores']):
        if score > 0.5:
            found_objects.append(detector.categories[label])
            
    return found_objects

if __name__ == "__main__":
    print(f"Detected: {detect_defect('sample.jpg')}")