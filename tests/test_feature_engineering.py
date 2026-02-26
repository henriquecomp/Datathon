import pandas as pd
import pytest
from src.feature_engineering import create_features


def test_create_features_success():
    # Arrange
    df = pd.DataFrame(
        {
            "IAA": [5.0, 6.0],
            "IEG": [2.0, 10.0],
            "IPS": [5.0, 5.0],
            "IDA": [4.0, 8.0],
            "IPV": [3.0, 3.0],
            "Defasagem": [-1, 1],
            "Ano_Base": [2022, 2023],
            "INDE": [1, 1],
            "IAN": [1, 1],
        }
    )

    # Act
    X, y = create_features(df)

    # Assert
    assert "Defasagem" not in X.columns
    assert "Ano_Base" not in X.columns
    assert "INDE" not in X.columns
    assert "IAN" not in X.columns
    assert "IEG_x_IDA" in X.columns
    assert X["IEG_x_IDA"].iloc[0] == 2.0 * 4.0  # 8.0

    assert y is not None
    assert y.iloc[0] == 1  # Defasagem < 0
    assert y.iloc[1] == 0


def test_create_features_missing_target():
    # Arrange
    df = pd.DataFrame({"IAA": [1], "IEG": [2], "IPS": [3], "IDA": [4]})

    # Act
    X, y = create_features(df)

    # Assert
    # Como alterámos a função para a API, y deve ser retornado como None em vez de gerar um erro
    assert y is None
    assert "IAA" in X.columns
