# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Charger ton modèle
model = joblib.load("modele_regression_moyenne.pkl")

# Définir le schéma d'entrée
class NotesInput(BaseModel):
    interro1: float
    interro2: float
    devoir1: float

# Définir l'application
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["capacitor://localhost", "http://localhost", ...]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Fonction pour convertir moyenne en niveau
def moyenne_to_niveau(m):
    if m < 7:
        return "grande_difficulte"
    elif m < 10:
        return "à_risque"
    elif m < 13:
        return "moyen"
    else:
        return "bon"

@app.post("/predict/")
def predict(notes: NotesInput):
    data = [[notes.interro1, notes.interro2, notes.devoir1]]
    moyenne = model.predict(data)[0]
    niveau = moyenne_to_niveau(moyenne)
    return {
        "moyenne_predite": round(moyenne, 2),
        "niveau": niveau
    }
