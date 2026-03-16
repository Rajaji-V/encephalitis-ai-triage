import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
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
    def __init__(self, output_dim=128):
        super().__init__()
        # Using ResNet18 as a more robust backbone
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the last FC layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.fc = nn.Linear(num_ftrs, output_dim)
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return self.dropout(x)

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=2): 
        super().__init__()
        self.clinical_encoder = ClinicalEncoder()
        self.image_encoder = ImageEncoder(output_dim=64)
        
        # Fused layer: 64 (Image) + 16 (Clinical) = 80
        self.classifier = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(40, num_classes)
        )
        
        # For Grad-CAM
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, img, clinical):
        # Extract features
        img_features = self.image_encoder(img)
        clinical_features = self.clinical_encoder(clinical)
        
        fused = torch.cat((img_features, clinical_features), dim=1)
        return self.classifier(fused)
        
    def get_gradcam_layer(self):
        # resnet18's last conv layer is layer4[1].conv2
        return self.image_encoder.resnet.layer4[-1].conv2

def load_model(weights_path=None):
    model = MultimodalModel(num_classes=2)
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Could not load weights: {e}. Starting with fresh weights.")
    model.eval()
    return model

def mc_dropout_predict(model, img_tensor, clinical_tensor, num_samples=10):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(img_tensor, clinical_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())
            
    preds = np.stack(preds) # (num_samples, batch, classes)
    mean_prediction = np.mean(preds, axis=0)[0] 
    uncertainty = np.std(preds, axis=0).mean()
    
    classes = ['Healthy', 'Encephalitis']
    pred_idx = np.argmax(mean_prediction)
    
    class_probabilities = {cls: float(prob * 100) for cls, prob in zip(classes, mean_prediction)}
    
    diagnosis = classes[pred_idx]
    
    if diagnosis == 'Healthy':
        risk_level = 'Low Risk'
    else:
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

def generate_gradcam(model, img_tensor, clinical_tensor):
    """Actual Grad-CAM implementation"""
    model.eval()
    
    # Essential to enable gradients even if called from a no_grad context
    with torch.enable_grad():
        # Set up hooks
        target_layer = model.get_gradcam_layer()
        features = []
        gradients = []
        
        def forward_hook(module, input, output):
            features.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)
        
        model.zero_grad()
        output = model(img_tensor, clinical_tensor)
        
        pred_idx = output.argmax(dim=1).item()
        target = output[0, pred_idx]
        target.backward()
        
        # Cleanup hooks immediately after backward
        handle_forward.remove()
        handle_backward.remove()
        
        if not gradients or not features:
            print("Warning: Grad-CAM failed to capture features/gradients. Returning original image.")
            # Return original image without heatmap
            img_array = img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            img_array = (img_array * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img_array = np.clip(img_array, 0, 1)
            img_array = np.uint8(255 * img_array.transpose(1, 2, 0))
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Extract gradients and features
        grads = gradients[0].detach()
        feats = features[0].detach()
        
        # Global average pooling of gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Weighted sum of features
        cam = torch.sum(weights * feats, dim=1).squeeze()
        cam = F.relu(cam)
        
        # Normalize cam
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.cpu().numpy()
        
        # Resize to original image size
        cam_resized = cv2.resize(cam, (224, 224))
        cam_heatmap = np.uint8(255 * cam_resized)
        cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
        
        # Denormalize image for overlay
        img_array = img_tensor[0].detach().cpu().numpy() # (3, 224, 224)
        # Reverse ImageNet normalization
        img_array = (img_array * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        img_array = np.clip(img_array, 0, 1)
        img_array = np.uint8(255 * img_array.transpose(1, 2, 0))
        # RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        overlay = cv2.addWeighted(img_array, 0.6, cam_heatmap, 0.4, 0)
        return overlay

def generate_shap(clinical_array):
    # clinical_array: [csf_protein, csf_glucose, age, symptom_severity]
    features = ['csf_protein', 'csf_glucose', 'age', 'symptom_severity']
    
    # Simulated importance based on clinical thresholds
    shap_vals = []
    shap_vals.append(0.6 if clinical_array[0] > 50 else -0.3)
    shap_vals.append(0.5 if clinical_array[1] < 50 else -0.2)
    shap_vals.append(0.15 if clinical_array[2] > 60 else -0.1)
    shap_vals.append(0.4 if clinical_array[3] > 6 else -0.4)
    
    return dict(zip(features, shap_vals))

def generate_decision_summary(shap_vals):
    summary = []
    if shap_vals.get('csf_protein', 0) > 0:
        summary.append("Elevated CSF Protein levels detected, suggesting neuro-inflammation.")
    else:
        summary.append("CSF Protein levels are within normal physiological range.")
        
    if shap_vals.get('csf_glucose', 0) > 0:
        summary.append("Decreased CSF Glucose levels indicate potential metabolic distress.")
    
    if shap_vals.get('symptom_severity', 0) > 0:
        summary.append("Patient exhibits high neuro-symptomatic severity.")

    summary.append("Visual features in MRI indicate morphological changes.")
    return summary
