import pytest
import os
import cv2
from app import detect_defect

# Path to your new images
SAMPLE_DIR = os.path.join('images', 'sample')

def get_sample_images():
    """Helper to find images 1.bmp through 6.bmp"""
    image_paths = []
    if os.path.exists(SAMPLE_DIR):
        for i in range(1, 7):
            path = os.path.join(SAMPLE_DIR, f"{i}.bmp")
            if os.path.exists(path):
                image_paths.append(path)
    return image_paths

@pytest.mark.parametrize("image_path", get_sample_images())
def test_defect_detection_on_bmp(image_path):
    # 1. Verify image loads correctly (Paint BMPs can sometimes be tricky)
    img = cv2.imread(image_path)
    assert img is not None, f"Could not read image at {image_path}"
    assert img.shape[:2] == (450, 450), f"Image {image_path} is not 450x450"

    # 2. Run your detection logic
    results = detect_defect(image_path)

    # 3. Assertions
    assert isinstance(results, list), "Result should be a list of detected objects"
    print(f"Tested {image_path}: Found {results}")