import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import logging
import subprocess
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.aluno_request import AlunoRequest
from app.schemas.risco_response import RiscoResponse
from prometheus_client import Counter, Histogram, Gauge

# Recupera o logger
logger = logging.getLogger(__name__)

# Cria o roteador
router = APIRouter()

# Limiar para resposta
limiar_fixo = 0.40

# MÉTRICAS CUSTOMIZADAS PARA O GRAFANA
# Conta quantas predições de cada tipo foram feitas
PREDICOES_TOTAL = Counter('modelo_predicoes_total', 'Total de predições', ['risco_detectado'])
# Guarda a distribuição das probabilidades (bom para ver se o modelo está confiante)
PROBABILIDADE_HISTOGRAMA = Histogram('modelo_probabilidade_risco', 'Distribuição das probabilidades geradas')
# Guarda o valor médio das features (Acompanhamento visual de DRIFT)
FEATURE_IAA = Gauge('feature_input_iaa', 'Valor da feature IAA recebida')
FEATURE_IEG = Gauge('feature_input_ieg', 'Valor da feature IEG recebida')

# Carregamento do Modelo
# Aponta para o banco de dados local do MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Define qual modelo queremos buscar
model_name = "Modelo_Risco_Defasagem"
alias = "production" # Pega sempre a última versão treinada

try:
    model_uri = f"models:/{model_name}@{alias}"
    logger.info(f"Aplicação iniciando... Modelo a ser carregado: {model_uri}...")   
    
    # Carrega o modelo diretamente do mlflow
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Aplicação iniciada! Modelo carregado: {model_uri}...")       
except Exception as e:
    logger.error(f"Erro ao carregar o modelo do MLflow: {e}")
    model = None

def executar_treinamento_em_background():
    logger.info("Iniciando o processo de re-treinamento (executando src/train.py)...")
    try:
        # Chama o script Python como se estivesse no terminal
        # capture_output=True permite-nos ler os prints do train.py e guardar no nosso log
        resultado = subprocess.run(
            ["python", "src/train.py"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        logger.info("Treinamento concluído com sucesso!")
        logger.info("Para a utilização do modelo, adicione o alias @production no MLFlow e acione o endpoint /reload!")
        
        # Loga as mensagens que o train.py imprimiu na tela
        for linha in resultado.stdout.split('\n'):
            if linha.strip():
                logger.info(f"[train.py] {linha}")
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao treinar o modelo. Código de falha: {e.returncode}")
        logger.error(f"Erro do train.py -> {e.stderr}")
    except Exception as e:
        logger.error(f"Falha crítica ao tentar iniciar o script de treino: {str(e)}")    

# Rotas

@router.get("/")
def home():
    return {"status": "ok"}

@router.post("/predict", response_model=RiscoResponse)
def predict_risk(aluno: AlunoRequest):
    # Previsão de risco de defasagem
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
        
    # GRAFANA: Atualiza as métricas das features recebidas (Drift)    
    FEATURE_IAA.set(aluno.IAA)
    FEATURE_IEG.set(aluno.IEG)    
    
    # Converter input para DataFrame
    data = {
        "IAA": [aluno.IAA],
        "IEG": [aluno.IEG],
        "IPS": [aluno.IPS],
        "IDA": [aluno.IDA],
        "IPV": [aluno.IPV]
    }
    df_input = pd.DataFrame(data)
    
    # Engenharia de Features (idêntica ao treinamento)
    df_input['IEG_x_IDA'] = df_input['IEG'] * df_input['IDA']
    df_input['IEG_x_IAA'] = df_input['IEG'] * df_input['IAA']
    df_input['IPS_x_IDA'] = df_input['IPS'] * df_input['IDA']
    
    # Predição
    try:
        proba = model.predict_proba(df_input)[0][1]
        risco = 1 if proba >= limiar_fixo else 0

        # -----------------------------------------------------------------
        # GRAFANA: Grava a predição final e a probabilidade
        # -----------------------------------------------------------------
        PREDICOES_TOTAL.labels(risco_detectado=str(risco)).inc()
        PROBABILIDADE_HISTOGRAMA.observe(proba)      

        logger.info(
            f"PREDIÇÃO | Aluno IAA: {aluno.IAA} | IEG: {aluno.IEG} | "
            f"Risco: {risco} | Probabilidade: {proba:.4f} | Mensagem: {"ALERTA: Risco detectado!" if risco == 1 else "Risco baixo" }"
        )
        
        return {
            "risco_defasagem": int(risco),
            "probabilidade_risco": float(round(proba, 4)),
            "mensagem": "ALERTA: Risco detectado!" if risco == 1 else "Risco baixo"
        }
    except Exception as e:
        logger.error(f"Falha na predição para o aluno {aluno.IAA}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
    
@router.post("/reload")
def reload_model():
    """
    Rota administrativa para recarregar o modelo em memória sem precisar reiniciar o servidor Uvicorn.
    """
    global model
    try:  
        logger.info(f"Recarga de Modelo solicitada. Modelo a ser carregado: {model_uri}...")   
        model = mlflow.sklearn.load_model(model_uri)
        return {"status": "sucesso", "mensagem": "Modelo atualizado com a última versão de produção!"}
    except Exception as e:
        logger.error(f"Erro ao recarregar o modelo {model_uri}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao recarregar o modelo: {str(e)}")    
    
@router.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """
    Endpoint administrativo para forçar o re-treinamento do modelo.
    Executa o processo de forma assíncrona (Background Task).
    """
    logger.info("Requisição recebida no endpoint /retrain.")
    
    # Envia a função pesada para rodar em segundo plano
    background_tasks.add_task(executar_treinamento_em_background)
    
    return {
        "status": "sucesso",
        "mensagem": "Treinamento iniciado em segundo plano. Acompanhe os logs no Grafana/Loki para ver o progresso e o resultado."
    }    