from fastapi import FastAPI, Request
from app.routes import router
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time
import os

# Cria a pasta de logs se não existir
os.makedirs("logs", exist_ok=True)

# Configuração profissional do Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api_events.log"), # Salva no arquivo para o Promtail ler
        logging.StreamHandler()                     # Mostra no terminal para você ver
    ]
)
logger = logging.getLogger("API_PassosMagicos")

# Inicializa a API
app = FastAPI(
    title="Previsão de Risco",
    description="API para identificar alunos com risco de defasagem escolar",
    version="1.0"
)

# Inclui as rotas definidas no routes.py
app.include_router(router)

# Inicia o monitoramento padrão (CPU, memória, latência, requisições) e cria a rota /metrics
Instrumentator().instrument(app).expose(app)

# Middleware loga TODAS as requisições que chegam na API
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"IP: {request.client.host} | Method: {request.method} | Path: {request.url.path} | "
        f"Status: {response.status_code} | Time: {process_time:.2f}ms"
    )
    return response