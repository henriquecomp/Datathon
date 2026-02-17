from pydantic import BaseModel

class RiscoResponse(BaseModel):
    risco_defasagem: int
    probabilidade_risco: float
    mensagem: str