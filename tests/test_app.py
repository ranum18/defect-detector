import numpy as np
from app import detect_defect


def test_dark_image_is_defect():
    dark_image = np.zeros((100, 100, 3))
    assert detect_defect(dark_image) == "DEFECT"


def test_bright_image_is_ok():
    bright_image = np.ones((100, 100, 3)) * 255
    assert detect_defect(bright_image) == "OK"
