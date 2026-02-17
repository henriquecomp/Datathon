import joblib
import pandas as pd
import numpy as np
import os
import pytest
from sklearn.pipeline import Pipeline

MODEL_PATH = "app/model/modelo.pkl"

def test_model_file_exists():
    """Verifica se o arquivo .pkl foi gerado no caminho esperado"""
    assert os.path.exists(MODEL_PATH), f"O arquivo do modelo não foi encontrado em: {MODEL_PATH}"

def test_model_loading():
    """Verifica se conseguimos carregar o arquivo com joblib"""
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        pytest.fail(f"Falha ao carregar o modelo .pkl: {e}")
    
    # Verifica se é um Pipeline (como definimos no train.py) ou um modelo Scikit-Learn
    assert isinstance(model, Pipeline) or hasattr(model, "predict"), "O objeto carregado não parece ser um modelo válido."

def test_model_prediction_mechanics():
    """
    Verifica se o modelo consegue receber um DataFrame 'sujo' (com Nulos)
    e devolver uma previsão sem quebrar
    """
    model = joblib.load(MODEL_PATH)

    # Cria dados de entrada simulando o que a API envia
    # IMPORTANTE: Deve conter as 8 colunas que o modelo espera (5 originais + 3 interações)
    input_data = pd.DataFrame({
        'IAA': [5.5, np.nan],  # Testando com valor Nulo para ver se o Imputer funciona
        'IEG': [6.0, 4.0],
        'IPS': [7.0, 5.0],
        'IDA': [8.0, 6.0],
        'IPV': [9.0, 7.0],
        # Features de Interação (que a API/Feature Engineering calculam)
        'IEG_x_IDA': [48.0, 24.0],
        'IEG_x_IAA': [33.0, np.nan], # Interação com nulo
        'IPS_x_IDA': [56.0, 30.0]
    })

    # Tenta fazer a predição
    try:
        # Teste de predict (classe 0 ou 1)
        preds = model.predict(input_data)
        
        # Teste de predict_proba (probabilidades)
        probs = model.predict_proba(input_data)
        
    except Exception as e:
        pytest.fail(f"O modelo falhou ao realizar a predição: {e}")

    # Verificações de Saída
    assert len(preds) == 2
    assert preds[0] in [0, 1]
    
    assert probs.shape == (2, 2) # 2 amostras, 2 classes
    assert 0.0 <= probs[0][1] <= 1.0 # Probabilidade válida