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

from torchvision import transforms

# Config
DATASET_PATH = r"d:\Projects\MiniProject2\data\raw\Dataset"
METADATA_PATH = r"d:\Projects\MiniProject2\data\processed\clinical_metadata.csv"
MODEL_SAVE_PATH = r"d:\Projects\MiniProject2\data\models\final_model.pth"
BATCH_SIZE = 16
EPOCHS = 40 # Increased for better convergence with early stopping
LEARNING_RATE = 0.0001 # Lowered for fine-tuning ResNet

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        label_dir = row['label_dir']
        filename = row['filename']
        img_path = os.path.join(DATASET_PATH, label_dir, filename)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for torchvision
        img = cv2.resize(img, (224, 224))
        
        if self.transform:
            import PIL.Image as Image
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # 2. Clinical Data: [protein, glucose, age, symptom_severity]
        clinical_data = torch.tensor([
            row['csf_protein'], 
            row['csf_glucose'], 
            row['age'], 
            row['symptom_severity']
        ], dtype=torch.float32)
        
        # 3. Target Label
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
    
    # Enhanced Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultimodalDataset(train_df, transform=train_transform)
    val_dataset = MultimodalDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ml_model.MultimodalModel(num_classes=2).to(device)
    
    # Calculate class weights to handle imbalance
    healthy_count = len(train_df[train_df['label_dir'] == 'healthy'])
    enc_count = len(train_df) - healthy_count
    total_train = len(train_df)
    
    if healthy_count > 0 and enc_count > 0:
        weights = [total_train / healthy_count, total_train / enc_count]
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight decay for regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 7
    
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
        val_loss = 0.0
        with torch.no_grad():
            for imgs, clinical, labels in val_loader:
                imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
                outputs = model(imgs, clinical)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Early Stopping & Model Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model improved! Saved current best model to {MODEL_SAVE_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    train_model()
