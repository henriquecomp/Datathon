from fastapi import FastAPI
from app.routes import router

# Inicializa a API
app = FastAPI(
    title="Previsão de Risco",
    description="API para identificar alunos com risco de defasagem escolar",
    version="1.0"
)

# Inclui as rotas definidas no routes.py
app.include_router(router)