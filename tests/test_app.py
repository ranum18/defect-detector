import os
import torch
from app import model # Import the model from your app

def test_model_loaded():
    # Check if the training actually produced a model file
    assert os.path.exists("defect_model.pth")

def test_prediction_output():
    # Create a fake input and ensure it outputs 2 classes
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 2) # Should output [BatchSize, NumClasses]
def test_training_produces_model():
    # This ensures that when the CI runs, it actually creates the model
    import os
    assert os.path.exists("defect_model.pth"), "Training failed to produce defect_model.pth"
def test_model_accuracy():
    # Run the testing logic on your 6 hidden images
    # If the model gets fewer than 4 correct, fail the build
    accuracy = run_evaluation_logic() 
    assert accuracy >= 0.80, f"Model accuracy too low: {accuracy}"