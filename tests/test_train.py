import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.train import run_training

@patch('src.train.load_data')
@patch('src.train.clean_data')
@patch('src.train.create_features')
@patch('src.train.train_test_split')
@patch('src.train.RandomizedSearchCV') # Mockamos a busca de hiperparâmetros
@patch('src.train.evaluate_model')
@patch('src.train.joblib.dump')
@patch('os.makedirs') # Para não criar pastas de verdade
def test_run_training_pipeline(mock_makedirs, mock_dump, mock_eval, mock_search, 
                               mock_split, mock_features, mock_clean, mock_load):
    
    # Configurando os retornos dos Mocks para o fluxo seguir
    mock_load.return_value = pd.DataFrame({'raw': [1]})
    mock_clean.return_value = pd.DataFrame({'clean': [1]})
    mock_features.return_value = (pd.DataFrame({'X': [1]}), pd.Series([1])) # X, y
    
    # Mock do train_test_split retornando 4 valores
    mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    # Mock do objeto RandomizedSearchCV e seu .fit
    mock_search_instance = mock_search.return_value
    mock_search_instance.best_estimator_ = "Modelo Treinado"
    
    # Executa a função principal
    run_training()
    
    # Asserts: Verifica se cada etapa foi chamada
    mock_load.assert_called_once()
    mock_clean.assert_called_once()
    mock_features.assert_called_once()
    mock_split.assert_called_once()
    
    # Verifica se o fit foi chamado
    mock_search_instance.fit.assert_called_once()
    
    # Verifica se avaliou o modelo
    mock_eval.assert_called_once()
    
    # Verifica se salvou o modelo
    mock_dump.assert_called_once()
    args, _ = mock_dump.call_args
    assert args[0] == "Modelo Treinado" # O objeto salvo deve ser o best_estimator
    assert "app/model" in args[1] or "models" in args[1] # Verifica o caminho

@patch('src.train.load_data')
def test_run_training_file_error(mock_load):
    # Arrange
    mock_load.side_effect = FileNotFoundError("Arquivo sumiu")
    
    # Act
    run_training()
    
    # Assert
    mock_load.assert_called_once()