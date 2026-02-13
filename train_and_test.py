import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import cv2
import os

class DefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

def get_model():
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def train_system():
    # --- 1. DATA PREPARATION ---
    train_dir = 'images/train'
    
    if not os.path.exists(train_dir):
        print(f"❌ FOLDER NOT FOUND: {train_dir}")
        return

    image_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.lower().endswith('.bmp')]
    print(f"🔎 Found {len(image_files)} BMP files in {train_dir}")
    
    if len(image_files) == 0:
        print("❌ ABORTING: No images found. Check your folder path and file extensions.")
        return

    labels = [0 if 'clean' in f.lower() else 1 for f in image_files]
    print(f"🏷️ Labels generated: {labels}")

    # Split data
    train_x, val_x, train_y, val_y = train_test_split(image_files, labels, test_size=0.22, random_state=42)
    print(f"📊 Split: {len(train_x)} training, {len(val_x)} validation")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(DefectDataset(train_x, train_y, transform), batch_size=2, shuffle=True)

    # --- 2. TRAINING ---
    print("🧠 Loading Model...")
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("🚀 Starting Training Loop...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"✅ Epoch [{epoch+1}/10] finished. Average Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "defect_model.pth")
    print("💾 Model saved successfully as defect_model.pth")

def test_hidden_images():
    # --- 3. EVALUATION ON HIDDEN IMAGES ---
    print("\n--- Testing on 6 Hidden Images ---")
    model = get_model()
    model.load_state_dict(torch.load("defect_model.pth"))
    model.eval()

    test_dir = 'images/test'
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.bmp')]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for f_path in test_files:
        img = cv2.imread(f_path)
        input_tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            result = "DEFECT" if predicted.item() == 1 else "CLEAN"
            print(f"Image: {os.path.basename(f_path)} | Prediction: {result}")

if __name__ == "__main__":
    # Sanity Check
    train_dir = 'images/train'
    if not os.path.exists(train_dir):
        print(f"❌ Error: {train_dir} not found!")
    else:
        files = [f for f in os.listdir(train_dir) if f.endswith('.bmp')]
        print(f"✅ Found {len(files)} images in training folder.")
        
    # Now run the actual training
    train_system()
    test_hidden_images()