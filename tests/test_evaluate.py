import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.evaluate import evaluate_model

@pytest.fixture
def mock_model():
    model = MagicMock()
    # Simula predict_proba retornando:
    # Caso 0: 0.2 (Risco Baixo)
    # Caso 1: 0.8 (Risco Alto)
    # Caso 2: 0.41 (Risco Alto se limiar for 0.40)
    model.predict_proba.return_value = np.array([
        [0.8, 0.2], 
        [0.2, 0.8], 
        [0.54, 0.41]
    ])
    # O feature_importances_ para modelos de árvore
    model.feature_importances_ = np.array([0.1, 0.2, 0.7])
    return model

# 1. Mockamos o MLflow para não sujar o banco de dados durante os testes
@patch('src.evaluate.mlflow') 
# 2. Mockamos o gerador de gráficos do Sklearn para evitar o erro de classificador falso
@patch('src.evaluate.ConfusionMatrixDisplay.from_estimator') 
def test_evaluate_model_metrics(mock_cmd, mock_mlflow, mock_model):
    # Arrange
    X_test = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})
    y_test = pd.Series([0, 1, 1]) 
    
    # Act
    results = evaluate_model(mock_model, X_test, y_test, threshold=0.40)
    
    # Assert
    assert results['f1'] == 1.0
    assert results['recall'] == 1.0
    
    # Verifica matriz de confusão (Tudo acerto)
    cm = results['confusion_matrix']
    assert cm[0][0] == 1 # VN
    assert cm[1][1] == 2 # VP
    
    # Garante que as funções mockadas foram chamadas corretamente
    mock_cmd.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()


@patch('src.evaluate.mlflow')
@patch('src.evaluate.ConfusionMatrixDisplay.from_estimator')
def test_evaluate_model_threshold_influence(mock_cmd, mock_mlflow, mock_model):
    # Arrange
    X_test = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})
    y_test = pd.Series([0, 1, 1])
    
    # Act
    # Se aumentarmos o limiar para 0.90, o modelo deve errar tudo que é positivo
    results = evaluate_model(mock_model, X_test, y_test, threshold=0.90)
    
    # Assert
    # Recall deve ser 0, pois não detectou nenhum dos 2 positivos
    assert results['recall'] == 0.0