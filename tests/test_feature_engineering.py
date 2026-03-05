import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import create_features, extrair_fase


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
            "Fase": ["8", "ALFA"],
            "Pedra": ["AGATA", "TOPAZIO"],
            "Genero": ["MENINO", "MENINA"],
            "Instituicao_de_ensino": ["ESCOLA PUBLICA", "PRIVADA - PROGRAMA DE APADRINHAMENTO"],
        }
    )

    # Act
    X, y = create_features(df)

    # Assert — leakage e identificadores removidos
    assert "Defasagem" not in X.columns
    assert "Ano_Base" not in X.columns
    assert "INDE" not in X.columns
    assert "IAN" not in X.columns
    assert "Pedra" not in X.columns   # Leakage: derivada do INDE
    assert "Fase" not in X.columns    # Substituída por Fase_Num

    # Assert — Genero padronizado (MENINO→MASCULINO, MENINA→FEMININO)
    assert X["Genero"].iloc[0] == "MASCULINO"
    assert X["Genero"].iloc[1] == "FEMININO"

    # Assert — Instituição agrupada em 3 categorias
    assert X["Instituicao_de_ensino"].iloc[0] == "PUBLICA"
    assert X["Instituicao_de_ensino"].iloc[1] == "PRIVADA"

    # Assert — features criadas
    assert "IEG_x_IDA" in X.columns
    assert X["IEG_x_IDA"].iloc[0] == 2.0 * 4.0
    assert "Fase_Num" in X.columns
    assert X["Fase_Num"].iloc[0] == 8
    assert X["Fase_Num"].iloc[1] == 0  # ALFA vira 0

    assert y is not None
    assert y.iloc[0] == 1
    assert y.iloc[1] == 0


def test_create_features_missing_target():
    # Arrange
    df = pd.DataFrame({"IAA": [1], "IEG": [2], "IPS": [3], "IDA": [4]})
    # Act
    X, y = create_features(df)
    # Assert
    assert y is None
    assert "IAA" in X.columns


def test_extrair_fase_edge_cases():
    # Testa diretamente os casos estranhos da função
    assert extrair_fase("FASE 2") == 2
    assert extrair_fase("Alpha") == 0
    assert pd.isna(extrair_fase("Texto Sem Numero"))
