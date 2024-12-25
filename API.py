from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle préalablement sauvegardé
model = joblib.load('lgb_model.pkl')

# Initialiser FastAPI
app = FastAPI()

# Définir la structure des données d'entrée
class InputData(BaseModel):
    EXT_SOURCE_2: float
    DAYS_BIRTH: int
    EXT_SOURCE_3: float
    bureau_DAYS_CREDIT_max: float
    bureau_DAYS_CREDIT_min: float
    bureau_DAYS_CREDIT_UPDATE_mean: float
    bureau_DAYS_CREDIT_mean: float
    bureau_CREDIT_ACTIVE_Closed_mean: float
    CODE_GENDER_M: int
    bureau_CREDIT_ACTIVE_Active_mean: float

# Définir l'endpoint de prédiction
@app.post("/predict/")
def predict(input_data: InputData):
    # Convertir les données d'entrée en numpy array
    data = np.array([[
        input_data.EXT_SOURCE_2, input_data.DAYS_BIRTH, input_data.EXT_SOURCE_3, 
        input_data.bureau_DAYS_CREDIT_max, input_data.bureau_DAYS_CREDIT_min, 
        input_data.bureau_DAYS_CREDIT_UPDATE_mean, input_data.bureau_DAYS_CREDIT_mean, 
        input_data.bureau_CREDIT_ACTIVE_Closed_mean, input_data.CODE_GENDER_M, 
        input_data.bureau_CREDIT_ACTIVE_Active_mean
    ]])

    # Faire la prédiction
    prediction = model.predict(data)
    proba = model.predict_proba(data)[:, 1]  # Probabilité de la classe 1

    # Retourner la prédiction et la probabilité
    return {"prediction": int(prediction[0]), "probability": float(proba[0])}

# Lancer l'API avec uvicorn main:app --reload
# Exemple d'utilisation avec json:
# {
#   "EXT_SOURCE_2": 0.5,
#   "DAYS_BIRTH": -12000,
#   "EXT_SOURCE_3": 0.7,
#   "bureau_DAYS_CREDIT_max": -1000,
#   "bureau_DAYS_CREDIT_min": -2000,
#   "bureau_DAYS_CREDIT_UPDATE_mean": -1500,
#   "bureau_DAYS_CREDIT_mean": -1750,
#   "bureau_CREDIT_ACTIVE_Closed_mean": 0.2,
#   "CODE_GENDER_M": 1,
#   "bureau_CREDIT_ACTIVE_Active_mean": 0.3
# }



