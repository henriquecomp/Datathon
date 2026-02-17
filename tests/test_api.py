import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import importlib
import sys

from app.main import app
from app import routes 

client = TestClient(app)

# Fixture para Mockar o Modelo
@pytest.fixture
def mock_model():
    model_mock = MagicMock()
    # Simula retorno: 20% classe 0 (sem risco), 80% classe 1 (risco)
    model_mock.predict_proba.return_value = [[0.2, 0.8]] 
    return model_mock

def test_home():
    # Arrange & Act & Assert
    response = client.get("/")
    assert response.status_code == 200    
    assert response.json().get("status") == "ok" 

def test_predict_risk_high(mock_model):
    # Arrange & Act & Assert
    with patch.object(routes, 'model', mock_model):
        payload = {
            "IAA": 5.5, "IEG": 2.0, "IPS": 6.0, "IDA": 4.5, "IPV": 7.0
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["risco_defasagem"] == 1
        assert "ALERTA" in data["mensagem"]

def test_predict_risk_low():
    # Arrange & Act & Assert
    low_risk_model = MagicMock()
    low_risk_model.predict_proba.return_value = [[0.9, 0.1]]
    
    with patch.object(routes, 'model', low_risk_model):
        payload = { "IAA": 10, "IEG": 10, "IPS": 10, "IDA": 10, "IPV": 10 }
        response = client.post("/predict", json=payload)
        
        assert response.json()["risco_defasagem"] == 0
        assert "baixo" in response.json()["mensagem"]

def test_predict_model_not_loaded():
    # Arrange & Act & Assert
    with patch.object(routes, 'model', None):
        payload = { "IAA": 0, "IEG": 0, "IPS": 0, "IDA": 0, "IPV": 0 }
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Modelo não carregado" in response.json()["detail"]

def test_prediction_internal_error():
    # Arrange & Act & Assert
    mock_model_error = MagicMock()
    mock_model_error.predict_proba.side_effect = Exception("Erro interno matemático")

    with patch.object(routes, 'model', mock_model_error):
        payload = {
            "IAA": 5.5, "IEG": 6.0, "IPS": 7.0, "IDA": 8.0, "IPV": 9.0
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 500
        assert "Erro na predição" in response.json()['detail']

def test_model_loading_exception():
    # Arrange & Act & Assert
    # Simula falha no load (arquivo sumiu)    
    with patch('joblib.load', side_effect=FileNotFoundError("Arquivo sumiu")):
        importlib.reload(routes) # Recarrega o módulo routes
        assert routes.model is None

    # LIMPEZA SEGURA (TEARDOWN)
    # Recarregamos o módulo forçando o joblib.load a funcionar (MOCKADO),
    # para não depender se o arquivo real existe ou não no disco.
    with patch('joblib.load', return_value="Modelo Recuperado"):
        importlib.reload(routes)
    
    assert routes.model is not None