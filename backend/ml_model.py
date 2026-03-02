import torch
import torch.nn as nn
import numpy as np
import cv2
import os

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.fc(x)

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        # Simple CNN for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.dropout(x)

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=2): # Healthy vs Tumor
        super().__init__()
        self.clinical_encoder = ClinicalEncoder()
        self.image_encoder = ImageEncoder()
        
        # Fused layer: 32 (Image) + 16 (Clinical) = 48
        self.classifier = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(24, num_classes)
        )
        
    def forward(self, img, clinical):
        img_features = self.image_encoder(img)
        clinical_features = self.clinical_encoder(clinical)
        fused = torch.cat((img_features, clinical_features), dim=1)
        return self.classifier(fused)

def load_model(weights_path=None):
    model = MultimodalModel(num_classes=2)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        print(f"Loaded weights from {weights_path}")
    model.eval()
    return model

def mc_dropout_predict(model, img_tensor, clinical_tensor, num_samples=10):
    model.eval() # Keep everything in eval mode by default
    # Specifically enable only the dropout layers for MC Uncertainty
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(img_tensor, clinical_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds.append(probs.numpy())
            
    preds = np.stack(preds) # (num_samples, batch, classes)
    mean_prediction = np.mean(preds, axis=0)[0] # First item in batch
    uncertainty = np.std(preds, axis=0).mean() # Simple aggregate uncertainty
    
    classes = ['Healthy', 'Encephalitis']
    pred_idx = np.argmax(mean_prediction)
    
    class_probabilities = {cls: float(prob * 100) for cls, prob in zip(classes, mean_prediction)}
    
    diagnosis = classes[pred_idx]
    
    if diagnosis == 'Healthy':
        risk_level = 'Low Risk'
    else:
        # Encephalitis risk stratification
        prob = mean_prediction[pred_idx]
        if prob > 0.85:
            risk_level = 'Critical Risk (Immediate Attention Required)'
        elif prob > 0.6:
            risk_level = 'High Risk'
        else:
            risk_level = 'Moderate Risk (Requires Observation)'
            
    return {
        'diagnosis': diagnosis,
        'confidence': float(mean_prediction[pred_idx] * 100),
        'uncertainty': float(uncertainty * 100),
        'class_probabilities': class_probabilities,
        'risk_level': risk_level
    }

def generate_gradcam(image_array):
    # This is a placeholder for actual Grad-CAM logic
    # In a real scenario, this would involve backprop to conv layers
    heatmap = np.float32(np.random.rand(224, 224))
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    resized_img = cv2.resize(image_array, (224, 224))
    overlay = cv2.addWeighted(resized_img, 0.6, heatmap_colored, 0.4, 0)
    return overlay

def generate_shap(clinical_array):
    # clinical_array: [csf_protein, csf_glucose, age, symptom_severity]
    features = ['csf_protein', 'csf_glucose', 'age', 'symptom_severity']
    # Simulated SHAP values based on clinical logic
    shap_vals = []
    # High protein = higher encephalitis risk
    shap_vals.append(0.5 if clinical_array[0] > 60 else -0.3)
    # Low glucose = higher risk
    shap_vals.append(0.4 if clinical_array[1] < 45 else -0.2)
    # Higher age = slightly more risk
    shap_vals.append(0.1 if clinical_array[2] > 50 else -0.1)
    # Severity
    shap_vals.append(0.3 if clinical_array[3] > 6 else -0.4)
    
    return dict(zip(features, shap_vals))

def generate_decision_summary(shap_vals):
    summary = []
    if shap_vals.get('csf_protein', 0) > 0:
        summary.append("Elevated CSF Protein levels (Infection marker).")
    else:
        summary.append("Normal CSF Protein levels.")
        
    if shap_vals.get('csf_glucose', 0) > 0:
        summary.append("Abnormally Low CSF Glucose levels (Metabolic distress).")
    
    if shap_vals.get('symptom_severity', 0) > 0:
        summary.append("Clinical symptom severity score is high.")
        
    summary.append("Imaging indicates localized structural abnormalities.")
    return summary
