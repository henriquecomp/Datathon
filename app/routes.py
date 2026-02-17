import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException
from app.schemas.aluno_request import AlunoRequest
from app.schemas.risco_response import RiscoResponse

# Cria o roteador
router = APIRouter()

# Limiar para resposta
limiar_fixo = 0.40

# Carregamento do Modelo
# Carregamos aqui para estar disponível para as rotas
try:
    # Verifique se o nome do arquivo na pasta models é 'model.pkl' ou 'modelo_otimizado.pkl'
    model = joblib.load("app/model/modelo.pkl")
    print("Modelo carregado no routes.py!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None

# Rotas

@router.get("/")
def home():
    return {"status": "ok"}

@router.post("/predict", response_model=RiscoResponse)
def predict_risk(aluno: AlunoRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
    
    # Converter input para DataFrame
    data = {
        "IAA": [aluno.IAA],
        "IEG": [aluno.IEG],
        "IPS": [aluno.IPS],
        "IDA": [aluno.IDA],
        "IPV": [aluno.IPV]
    }
    df_input = pd.DataFrame(data)
    
    # Engenharia de Features (idêntica ao treinamento)
    df_input['IEG_x_IDA'] = df_input['IEG'] * df_input['IDA']
    df_input['IEG_x_IAA'] = df_input['IEG'] * df_input['IAA']
    df_input['IPS_x_IDA'] = df_input['IPS'] * df_input['IDA']
    
    # Predição
    try:
        proba = model.predict_proba(df_input)[0][1]
        risco = 1 if proba >= limiar_fixo else 0
        
        return {
            "risco_defasagem": int(risco),
            "probabilidade_risco": float(round(proba, 4)),
            "mensagem": "ALERTA: Risco detectado!" if risco == 1 else "Risco baixo"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")