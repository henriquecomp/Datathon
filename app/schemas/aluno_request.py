from pydantic import BaseModel

class AlunoRequest(BaseModel):
    IAA: float
    IEG: float
    IPS: float
    IDA: float
    IPV: float