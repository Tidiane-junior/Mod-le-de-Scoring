# %% Chargement des librairies
import pytest
from fastapi.testclient import TestClient
from API import app  

# %%
# Création du client de test
client = TestClient(app)

#%% Test de l'API
def test_predict():
    # Données d'entrée pour l'API 
    input_data = {
        "EXT_SOURCE_2": 0.5,
        "DAYS_BIRTH": -12000,
        "EXT_SOURCE_3": 0.4,
        "bureau_DAYS_CREDIT_max": -500,
        "bureau_DAYS_CREDIT_min": -1000,
        "bureau_DAYS_CREDIT_UPDATE_mean": -750,
        "bureau_DAYS_CREDIT_mean": -800,
        "bureau_CREDIT_ACTIVE_Closed_mean": 0.3,
        "CODE_GENDER_M": 1,
        "bureau_CREDIT_ACTIVE_Active_mean": 0.7
    }
    
    # Envoie une requête POST à l'endpoint /predict
    response = client.post("/predict", json=input_data)
    
    # Vérifions si le statut HTTP est 200
    assert response.status_code == 200, f"Erreur : {response.text}"
    
    # Vérifions si le résultat contient une clé "prediction"
    result = response.json()
    assert "prediction" in result, "La réponse ne contient pas de clé 'prediction'."
    
    # Vérifie que la prédiction est correcte (type attendu : int ou float)
    assert isinstance(result["prediction"], (int, float)), "La prédiction doit être un entier ou un flottant."

# %%
