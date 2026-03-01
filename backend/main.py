from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from schemas import ClinicalData
import ml_model
import numpy as np
import cv2
import base64
import torch
import uvicorn

app = FastAPI(title="Encephalitis Diagnosis API")

# Allow all origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ml_model.load_model()

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
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Preprocess image
    resized = cv2.resize(img_array, (224, 224))
    img_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    gender_val = 0 if gender.lower().startswith('m') else 1
    clinical_array = np.array([csf_protein, csf_glucose, age, symptom_severity], dtype=np.float32)
    clinical_tensor = torch.tensor(clinical_array).unsqueeze(0)
    
    # Prediction with MC Dropout
    result = ml_model.mc_dropout_predict(model, img_tensor, clinical_tensor)
    
    # Also return explanations
    gradcam_img = ml_model.generate_gradcam(img_array)
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
    
    gradcam_img = ml_model.generate_gradcam(img_array)
    _, buffer = cv2.imencode('.png', gradcam_img)
    gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
    
    clinical_array = np.array([csf_protein, csf_glucose, age, symptom_severity], dtype=np.float32)
    shap_vals = ml_model.generate_shap(clinical_array)
    decision_summary = ml_model.generate_decision_summary(shap_vals)
    
    return {
        "heatmap": gradcam_base64,
        "shap_values": shap_vals,
        "decision_summary": decision_summary
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
