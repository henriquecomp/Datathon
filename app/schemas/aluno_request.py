import re
import unicodedata

from pydantic import BaseModel, Field, field_validator


def _normalizar(val: str) -> str:
    """Normaliza texto da mesma forma que clean_data no treino."""
    s = str(val).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s.upper()


GENEROS_VALIDOS = {"FEMININO", "MASCULINO"}

INSTITUICOES_VALIDAS = {"PUBLICA", "PRIVADA", "REDE DECISAO"}


class AlunoRequest(BaseModel):
    IAA: float = Field(..., ge=0, le=10, description="Indicador de Auto Avaliação (0 a 10)")
    IEG: float = Field(..., ge=0, le=10, description="Indicador de Engajamento (0 a 10)")
    IPS: float = Field(..., ge=0, le=10, description="Indicador Psicossocial (0 a 10)")
    IDA: float = Field(..., ge=0, le=10, description="Indicador de Aprendizagem (0 a 10)")
    IPV: float = Field(..., ge=0, le=10, description="Indicador do Ponto de Virada (0 a 10)")
    Idade: int = Field(..., ge=5, le=30, description="Idade do aluno (5 a 30)")

    Fase: str = Field(..., description="Fase escolar (ex: ALFA, 1, 8, 2A, FASE 3)")
    Instituicao_de_ensino: str = Field(..., description="Tipo de instituição: PUBLICA, PRIVADA ou REDE DECISAO")
    Genero: str = Field(..., description="Gênero: FEMININO ou MASCULINO")

    @field_validator("Fase")
    @classmethod
    def validar_fase(cls, v: str) -> str:
        norm = _normalizar(v)
        if "ALFA" in norm or "ALPHA" in norm:
            return v
        if re.search(r"\d", norm):
            return v
        raise ValueError(
            f"Fase inválida '{v}'. "
            "Use ALFA/ALPHA ou um valor contendo número (ex: 1, 8, 2A, FASE 3)."
        )

    @field_validator("Instituicao_de_ensino")
    @classmethod
    def validar_instituicao(cls, v: str) -> str:
        norm = _normalizar(v)
        if norm not in INSTITUICOES_VALIDAS:
            raise ValueError(
                f"Instituição inválida '{v}'. "
                f"Valores aceitos: {sorted(INSTITUICOES_VALIDAS)}"
            )
        return v

    @field_validator("Genero")
    @classmethod
    def validar_genero(cls, v: str) -> str:
        norm = _normalizar(v)
        if norm not in GENEROS_VALIDOS:
            raise ValueError(
                f"Gênero inválido '{v}'. Valores aceitos: {sorted(GENEROS_VALIDOS)}"
            )
        return v
