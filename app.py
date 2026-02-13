import numpy as np
from model import SimpleDefectModel


def detect_defect(image: np.ndarray) -> str:
    """
    Detect defect based on average brightness.
    """
    model = SimpleDefectModel()
    prediction = model.predict(image)
    return prediction
