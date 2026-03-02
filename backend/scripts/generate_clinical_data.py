import os
import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

DATASET_PATH = r"d:\Projects\MiniProject2\data\raw\Dataset"
OUTPUT_PATH = r"d:\Projects\MiniProject2\data\processed\clinical_metadata.csv"

def generate_clinical_data():
    data = []
    classes = ["healthy", "tumor"]
    
    for label in classes:
        folder_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} not found.")
            continue
            
        filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in filenames:
            # Clinical feature generation logic
            if label == "healthy":
                age = random.randint(18, 70)
                csf_protein = round(random.uniform(15, 45), 2)  # Normal range: 15-45 mg/dL
                csf_glucose = round(random.uniform(50, 80), 2)   # Normal range: 50-80 mg/dL
                symptom_severity = random.randint(0, 2)         # Low severity
                diagnosis = "Healthy"
            else:
                age = random.randint(5, 80)
                # Encephalitis/Tumor profile
                csf_protein = round(random.uniform(60, 200), 2) # Elevated protein in encephalitis
                csf_glucose = round(random.uniform(20, 45), 2)  # Low glucose in bacterial/fungal
                symptom_severity = random.randint(6, 10)        # High severity
                diagnosis = "Encephalitis"
            
            data.append({
                "filename": filename,
                "label_dir": label,
                "age": age,
                "csf_protein": csf_protein,
                "csf_glucose": csf_glucose,
                "symptom_severity": symptom_severity,
                "diagnosis": diagnosis
            })
            
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully generated clinical data for {len(df)} images at {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_clinical_data()
