import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def bloquear_mlflow_globalmente():
    """
    Fixture de segurança: Garante que o MLflow nunca executa de verdade
    durante os testes unitários, evitando a poluição da base de dados.
    """
    with patch("src.train.mlflow"), patch("src.evaluate.mlflow"), patch(
        "app.routes.mlflow"
    ):
        yield
