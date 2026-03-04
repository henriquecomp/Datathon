# Previsão de Risco de Defasagem Escolar

API de machine learning para identificar alunos com risco de defasagem escolar, desenvolvida para o contexto da Passos Mágicos. O modelo utiliza Random Forest com otimização de hiperparâmetros via MLflow e métricas focadas em **recall** (sensibilidade).

---

## Estrutura do Projeto

```
├── app/                    # API FastAPI
│   ├── config.py           # Configuração (variáveis de ambiente)
│   ├── main.py             # Aplicação principal
│   ├── routes.py           # Endpoints e lógica de predição
│   ├── schemas/            # Pydantic (AlunoRequest, RiscoResponse)
│   └── model/              # Modelo .pkl (gerado no treino)
├── src/                    # Pipeline de ML
│   ├── utils.py            # Carregamento e unificação de dados
│   ├── preprocessing.py    # Limpeza e conversão de tipos
│   ├── feature_engineering.py  # Features e target
│   ├── train.py            # Treinamento com MLflow
│   └── evaluate.py         # Métricas e matriz de confusão
├── tests/                  # Testes pytest
├── notebooks/              # EDA e exploração
├── files/                  # CSVs de entrada (PEDE2022, 2023, 2024)
├── grafana/                # Dashboards e datasources
├── docker-compose.yml      # Stack completa (API, MLflow, Prometheus, Grafana, Loki)
└── .env.example            # Exemplo de variáveis de ambiente
```

---

## Requisitos

- Python 3.11+
- Dados em `files/` (PEDE2022.csv, PEDE2023.csv, PEDE2024.csv)

---

## Instalação

```bash
# Clone o repositório e entre na pasta
cd Datathon

# Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# ou: .venv\Scripts\activate   # Windows

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente (opcional)
cp .env.example .env
# Edite .env conforme necessário
```

---

## Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `MLFLOW_TRACKING_URI` | String de conexão do MLflow | `sqlite:///mlflow.db` |
| `LIMIAR_FIXO` | Limiar de probabilidade para risco (0.0–1.0) | `0.40` |

---

## Uso Local

### 1. Treinar o modelo

```bash
python -m src.train
```

O script carrega os CSVs em `files/`, aplica pré-processamento, engenharia de features e treina um Random Forest otimizado por recall. O modelo é salvo em `app/model/modelo.pkl` e registrado no MLflow. **Configure o alias `production` no MLflow UI** para que a API carregue o modelo.

### 2. Subir a API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Documentação interativa

- Swagger UI: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc  

---

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predição de risco para um aluno |
| `POST` | `/reload` | Recarrega o modelo em memória |
| `POST` | `/retrain` | Dispara retreinamento em background |
| `GET` | `/metrics` | Métricas Prometheus |

### Exemplo de predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IAA": 5.5,
    "IEG": 2.0,
    "IPS": 6.0,
    "IDA": 4.5,
    "IPV": 7.0,
    "Idade": 15,
    "Fase": "8",
    "Pedra": "AGATA",
    "Instituicao_de_ensino": "Escola Publica",
    "Genero": "F"
  }'
```

**Resposta:**
```json
{
  "risco_defasagem": 1,
  "probabilidade_risco": 0.7234,
  "mensagem": "ALERTA: Risco detectado!"
}
```

---

## Docker

```bash
# Subir toda a stack
docker compose up -d

# Serviços
# - API:        http://localhost:8000
# - MLflow UI:  http://localhost:5050
# - Grafana:    http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

---

## Monitoramento

- **Prometheus**: coleta métricas da API e do node-exporter  
- **Grafana**: dashboards de predições, probabilidades e infraestrutura  
- **Loki + Promtail**: centralização de logs da API  

---

## Testes

```bash
# Todos os testes (exceto test_model, que depende do .pkl)
python -m pytest tests/ -v --ignore=tests/test_model.py

# Com cobertura
python -m pytest tests/ -v --cov=app --cov=src --ignore=tests/test_model.py

# Teste do modelo (após treinar)
python -m pytest tests/test_model.py -v
```

---

## Pipeline de Dados

1. **`load_data`**: carrega e unifica CSVs (2022–2024), padroniza colunas  
2. **`clean_data`**: converte tipos, trata idades/notas, normaliza texto  
3. **`create_features`**: cria target (Defasagem < 0 → risco), interações (IEG×IDA, etc.), Fase_Num  
4. **Treino**: Random Forest + OneHotEncoder + SimpleImputer, otimizado por recall  

