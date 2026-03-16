from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from schemas import ClinicalData
import ml_model
import numpy as np
import cv2
import base64
import torch
import uvicorn

import os

app = FastAPI(title="Encephalitis Diagnosis API")

# Allow specific origins for dev (FastAPI doesn't allow "*" with allow_credentials=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = r"d:\Projects\MiniProject2\data\models\final_model.pth"
model = ml_model.load_model(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Diagnosis API running"}

@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    csf_protein: float = Form(...),
    csf_glucose: float = Form(...),
    symptom_severity: int = Form(...)
):
    try:
        contents = await image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # Preprocess image consistent with training
        resized = cv2.resize(img_array, (224, 224))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0)
        
        clinical_array = np.array([csf_protein, csf_glucose, age, symptom_severity], dtype=np.float32)
        clinical_tensor = torch.tensor(clinical_array).unsqueeze(0)
        
        # Prediction with MC Dropout
        result = ml_model.mc_dropout_predict(model, img_tensor, clinical_tensor)
        
        # Also return explanations
        gradcam_img = ml_model.generate_gradcam(model, img_tensor, clinical_tensor)
        _, buffer = cv2.imencode('.png', gradcam_img)
        gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
        
        shap_vals = ml_model.generate_shap(clinical_array)
        decision_summary = ml_model.generate_decision_summary(shap_vals)
        
        return {
            "prediction": result,
            "explanation": {
                "heatmap": gradcam_base64,
                "shap_values": shap_vals,
                "decision_summary": decision_summary
            }
        }
    except Exception as e:
        import traceback
        print(f"Error in predict_endpoint: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.post("/explain")
async def explain_endpoint(
    image: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    csf_protein: float = Form(...),
    csf_glucose: float = Form(...),
    symptom_severity: int = Form(...)
):
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Consistent preprocessing
    resized = cv2.resize(img_array, (224, 224))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0)
    
    clinical_array = np.array([csf_protein, csf_glucose, age, symptom_severity], dtype=np.float32)
    clinical_tensor = torch.tensor(clinical_array).unsqueeze(0)

    gradcam_img = ml_model.generate_gradcam(model, img_tensor, clinical_tensor)
    _, buffer = cv2.imencode('.png', gradcam_img)
    gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
    
    shap_vals = ml_model.generate_shap(clinical_array)
    decision_summary = ml_model.generate_decision_summary(shap_vals)
    
    return {
        "heatmap": gradcam_base64,
        "shap_values": shap_vals,
        "decision_summary": decision_summary
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
