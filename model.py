import numpy as np


class SimpleDefectModel:
    """
    A simple fake defect model based on brightness.
    """

    def predict(self, image: np.ndarray) -> str:
        brightness = np.mean(image)

        if brightness < 50:
            return "DEFECT"
        else:
            return "OK"
