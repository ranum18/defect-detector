import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class DefectModel:
    def __init__(self):
        # Load pre-trained model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        self.categories = weights.meta["categories"]

    def predict(self, img_tensor):
        with torch.no_grad():
            prediction = self.model([img_tensor])
        return prediction[0]