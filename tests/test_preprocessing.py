import pandas as pd
import numpy as np
from src.preprocessing import clean_data


def test_clean_data_conversion():
    # Dados brutos com vírgulas e pontos
    # Arrange
    df_raw = pd.DataFrame(
        {
            "IAA": ["5,5", "7.0", "8,1"],
            "IEG": [10, "0,0", "NaN"],
            "Defasagem": ["-1", "0", "2"],
        }
    )

    # Act
    df_clean = clean_data(df_raw)

    # Assert
    assert df_clean["IAA"].dtype == float or df_clean["IAA"].dtype == np.float64
    assert df_clean["IEG"].iloc[1] == 0.0
    assert pd.isna(df_clean["IEG"].iloc[2])
    assert df_clean["Defasagem"].iloc[0] == -1.0


def test_clean_data_missing_columns():
    # DataFrame vazio ou sem colunas esperadas
    # Arrange
    df_raw = pd.DataFrame({"Outra": [1, 2]})
    # Act
    df_clean = clean_data(df_raw)

    # Assert
    assert "Outra" in df_clean.columns
