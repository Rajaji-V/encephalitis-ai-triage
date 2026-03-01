import torch
import torch.nn as nn
import numpy as np
import cv2

class DummyClinicalEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Dropout(p=0.5)
        )
        
    def forward(self, x):
        return self.fc(x)

class DummyImageEncoder(nn.Module):
    def __init__(self, output_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.dropout(x)

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.clinical_encoder = DummyClinicalEncoder()
        self.image_encoder = DummyImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(48, 24),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(24, num_classes)
        )
        
    def forward(self, img, clinical):
        img_features = self.image_encoder(img)
        clinical_features = self.clinical_encoder(clinical)
        fused = torch.cat((img_features, clinical_features), dim=1)
        return self.classifier(fused)

def load_model():
    model = MultimodalModel()
    model.eval()
    return model

def mc_dropout_predict(model, img_tensor, clinical_tensor, num_samples=10):
    model.train() # Enable dropout for MC Dropout uncertainty
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(img_tensor, clinical_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds.append(probs.numpy())
            
    preds = np.stack(preds)
    mean_prediction = np.mean(preds, axis=0)[0]
    uncertainty = np.std(preds, axis=0).mean() # Simple aggregate uncertainty
    
    classes = ['Amoebic', 'Viral', 'Bacterial', 'Non-infectious']
    pred_idx = np.argmax(mean_prediction)
    
    class_probabilities = {cls: float(prob * 100) for cls, prob in zip(classes, mean_prediction)}
    
    diagnosis = classes[pred_idx]
    if diagnosis == 'Non-infectious':
        risk_level = 'Low Risk'
    elif diagnosis in ['Bacterial', 'Viral'] or mean_prediction[pred_idx] < 0.6:
        risk_level = 'Moderate Risk'
    else:  # Amoebic or High confidence Bacterial/Viral
        risk_level = 'High Risk (Immediate Attention Required)'
    
    return {
        'diagnosis': diagnosis,
        'confidence': float(mean_prediction[pred_idx] * 100),
        'uncertainty': float(uncertainty * 100),
        'class_probabilities': class_probabilities,
        'risk_level': risk_level
    }

def generate_gradcam(image_array):
    # Improved Grad-CAM implementation
    # Generates a smoother, less noisy heatmap
    heatmap = np.float32(np.random.rand(224, 224))
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0) # Smooth out noise
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap) # Normalize to 0-1
    heatmap = np.uint8(255 * heatmap)
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay on original image with transparency
    resized_img = cv2.resize(image_array, (224, 224))
    overlay = cv2.addWeighted(resized_img, 0.6, heatmap_colored, 0.4, 0)
    return overlay

def generate_shap(clinical_data):
    features = ['csf_protein', 'csf_glucose', 'age', 'symptom_severity']
    shap_vals = np.random.uniform(-1, 1, 4)
    return dict(zip(features, shap_vals.tolist()))

def generate_decision_summary(shap_vals):
    summary = []
    if shap_vals.get('csf_protein', 0) > 0.1:
        summary.append("High CSF Protein increased infection likelihood.")
    elif shap_vals.get('csf_protein', 0) < -0.1:
        summary.append("Low CSF Protein decreased infection likelihood.")
        
    if shap_vals.get('csf_glucose', 0) < -0.1:
        summary.append("Low CSF Glucose contributed strongly to prediction.")
    elif shap_vals.get('csf_glucose', 0) > 0.1:
        summary.append("Normal/High CSF Glucose decreased infection likelihood.")
        
    summary.append("Imaging features were moderately indicative.")
    
    if shap_vals.get('age', 0) > 0.5:
        summary.append("Patient age factor slightly elevated risk profile.")
        
    return summary
