import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_model

# Config
DATASET_PATH = r"d:\Projects\MiniProject2\data\raw\Dataset"
METADATA_PATH = r"d:\Projects\MiniProject2\data\processed\clinical_metadata.csv"
MODEL_SAVE_PATH = r"d:\Projects\MiniProject2\data\models\final_model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

class MultimodalDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        label_dir = row['label_dir']
        filename = row['filename']
        img_path = os.path.join(DATASET_PATH, label_dir, filename)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0 # (C, H, W)
        
        # 2. Clinical Data: [protein, glucose, age, symptom_severity]
        clinical_data = torch.tensor([
            row['csf_protein'], 
            row['csf_glucose'], 
            row['age'], 
            row['symptom_severity']
        ], dtype=torch.float32)
        
        # 3. Target Label: 0 (Healthy) or 1 (Tumor/Encephalitis)
        label = 0 if label_dir == "healthy" else 1
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img, clinical_data, label_tensor

def train_model():
    # Load metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file {METADATA_PATH} not found. Run generate_clinical_data.py first.")
        return
        
    df = pd.read_csv(METADATA_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = MultimodalDataset(train_df)
    val_dataset = MultimodalDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ml_model.MultimodalModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, clinical, labels in train_loader:
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, clinical, labels in val_loader:
                imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
                outputs = model(imgs, clinical)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
    # Save the model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
