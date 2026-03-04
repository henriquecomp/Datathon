"""
Configuração via variáveis de ambiente.
Valores padrão são usados quando a variável não está definida.
"""
import os


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# String de conexão do MLflow (ex: sqlite:///mlflow.db ou http://host:5000)
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Limiar de probabilidade para classificar como risco (0.0 a 1.0)
LIMIAR_FIXO: float = _get_float("LIMIAR_FIXO", 0.40)
