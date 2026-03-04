from pydantic import BaseModel

class AlunoRequest(BaseModel):
    # Numéricos
    IAA: float
    IEG: float
    IPS: float
    IDA: float
    IPV: float
    Idade: int
    
    # Categóricas
    Fase: str
    Pedra: str
    Instituicao_de_ensino: str
    Genero: str
    