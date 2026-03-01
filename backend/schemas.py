from pydantic import BaseModel

class ClinicalData(BaseModel):
    age: int
    gender: str
    csf_protein: float
    csf_glucose: float
    symptom_severity: int
    
class ExplanationResponse(BaseModel):
    shap_values: dict
