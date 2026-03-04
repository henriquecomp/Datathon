# Previsão de Risco de Defasagem Escolar - Passos Mágicos

---

## 1. Visão Geral do Projeto

### Objetivo

A Associação Passos Mágicos atua há mais de 32 anos na transformação da vida de crianças e jovens de baixa renda por meio da educação. Este projeto resolve um problema concreto da associação: **identificar precocemente alunos com risco de defasagem escolar**, permitindo que educadores e psicólogos intervenham antes que o aluno se distancie do nível esperado para sua série.

A variável-alvo é a **Defasagem** (valores negativos indicam que o aluno está abaixo do esperado). O modelo classifica cada estudante como "em risco" ou "sem risco" com base em indicadores educacionais, comportamentais e psicossociais coletados entre 2022 e 2024.

### Solução Proposta

Pipeline completa de Machine Learning com boas práticas de MLOps:

1. Pré-processamento e engenharia de features dos dados educacionais (PEDE 2022–2024).
2. Treinamento e otimização de um modelo Random Forest com busca de hiperparâmetros.
3. Registro e versionamento do modelo no MLflow.
4. API REST (FastAPI) para predição em tempo real.
5. Empacotamento com Docker e deploy local via Docker Compose.
6. Monitoramento contínuo com Prometheus, Grafana e Loki.

### Métrica de Avaliação e Confiabilidade

A métrica principal é o **Recall (Sensibilidade)**. Ela foi escolhida porque, no contexto educacional, **é mais grave deixar de identificar um aluno em risco (falso negativo) do que gerar um alerta desnecessário (falso positivo)**. Um aluno em risco não detectado perde a chance de receber apoio a tempo.

Para maximizar o recall sem sacrificar totalmente a precisão:
- O modelo é otimizado via `RandomizedSearchCV` com `scoring='recall'`.
- Utilizamos um **limiar de decisão de 0.40** (em vez do padrão 0.50), tornando o modelo mais sensível a casos de risco.
- O `class_weight='balanced'` compensa o desbalanceamento natural das classes.
- Colunas com **data leakage** confirmado (INDE, IAN, Pedra) são removidas para garantir que o modelo generalize corretamente. A coluna `Pedra` é derivada diretamente das faixas do INDE, portanto carrega a mesma informação vazada.
- A coluna `Fase` (categórica) é substituída por `Fase_Num` (numérica), evitando duplicação de informação no modelo.

Com essas estratégias, o modelo atinge recall acima de 90% no conjunto de teste, o que o torna confiável para uso em produção como ferramenta de apoio à decisão.

### Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| Linguagem | Python 3.11 |
| Frameworks de ML | scikit-learn, pandas, numpy |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Serialização | joblib (pickle) |
| Testes | pytest (97% de cobertura) |
| Empacotamento | Docker + Docker Compose |
| Deploy | Local (Docker Compose) |
| Monitoramento | Prometheus + Grafana + Loki/Promtail |

---

## 2. Estrutura do Projeto

```
├── app/                        # API FastAPI
│   ├── config.py               # Configuração centralizada (variáveis de ambiente)
│   ├── main.py                 # Aplicação principal (startup, middleware, logging)
│   ├── routes.py               # Endpoints (/predict, /reload, /retrain, /metrics)
│   ├── schemas/                # Schemas Pydantic
│   │   ├── aluno_request.py    # Payload de entrada (AlunoRequest) com validação
│   │   └── risco_response.py   # Payload de saída (RiscoResponse)
│   └── model/                  # Modelo .pkl (gerado após treinamento)
├── src/                        # Pipeline de ML
│   ├── utils.py                # Carregamento e unificação dos CSVs (2022–2024)
│   ├── preprocessing.py        # Limpeza, conversão de tipos, normalização de texto
│   ├── feature_engineering.py  # Criação do target, interações, Fase_Num, remoção de leakage
│   ├── train.py                # Treinamento com RandomizedSearchCV + MLflow
│   └── evaluate.py             # Métricas, importância de features, matriz de confusão
├── tests/                      # Testes unitários (35 testes, 97% cobertura)
│   ├── conftest.py             # Fixtures globais (bloqueio do MLflow em testes)
│   ├── test_api.py             # Testes dos endpoints da API
│   ├── test_utils.py           # Testes de carregamento de dados
│   ├── test_preprocessing.py   # Testes de limpeza
│   ├── test_feature_engineering.py  # Testes de engenharia de features
│   ├── test_train.py           # Testes do pipeline de treinamento
│   ├── test_evaluate.py        # Testes de avaliação
│   └── test_model.py           # Testes de integração do modelo salvo
├── notebooks/                  # EDA e exploração de dados
├── files/                      # CSVs de entrada (PEDE2022, PEDE2023, PEDE2024)
├── grafana/                    # Dashboards e datasources provisionados
│   └── provisioning/
│       ├── datasources/        # Prometheus + Loki
│       └── dashboards/         # Painéis JSON (modelo + infraestrutura)
├── Dockerfile                  # Imagem Docker da API
├── docker-compose.yml          # Stack completa (API, MLflow, Prometheus, Grafana, Loki)
├── prometheus.yml              # Configuração do Prometheus
├── promtail-config.yml         # Configuração do Promtail (coleta de logs)
├── requirements.txt            # Dependências Python
├── .env.example                # Exemplo de variáveis de ambiente
└── README.md                   # Esta documentação
```

---

## 3. Instruções de Deploy

### Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (para deploy containerizado)
- Dados em `files/` (PEDE2022.csv, PEDE2023.csv, PEDE2024.csv)

### Instalação de dependências

```bash
git clone <url-do-repositorio>
cd Datathon

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# ou: .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# Variáveis de ambiente (opcional)
cp .env.example .env
```

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `MLFLOW_TRACKING_URI` | String de conexão do MLflow | `sqlite:///mlflow.db` |
| `LIMIAR_FIXO` | Limiar de probabilidade para classificar risco (0.0–1.0) | `0.40` |

### Treinar o modelo

```bash
python -m src.train
```

O script executa todo o pipeline (carga, limpeza, features, treino, avaliação), salva o modelo em `app/model/modelo.pkl` e registra no MLflow. Após o treino, configure o alias `production` no MLflow UI para que a API carregue o modelo.

### Subir a API (local sem Docker)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Subir com Docker Compose (stack completa)

```bash
docker compose up -d --build
```

| Serviço | URL |
|---------|-----|
| API (FastAPI) | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5050 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

### Testes

```bash
# Testes unitários (32 testes)
python -m pytest tests/ -v --ignore=tests/test_model.py

# Com cobertura (97%)
python -m pytest tests/ -v --cov=app --cov=src --cov-report=term-missing --ignore=tests/test_model.py

# Teste de integração do modelo (requer modelo treinado)
python -m pytest tests/test_model.py -v
```

---

## 4. Exemplos de Chamadas à API

### Endpoints disponíveis

| Metodo | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predição de risco para um aluno |
| `POST` | `/reload` | Recarrega o modelo em memória (sem reiniciar o servidor) |
| `POST` | `/retrain` | Dispara retreinamento em background |
| `GET` | `/metrics` | Métricas Prometheus |

### POST /predict -- Predição de risco

**Input esperado:**

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
    "Instituicao_de_ensino": "Escola Publica",
    "Genero": "Masculino"
  }'
```

**Output gerado (aluno em risco):**

```json
{
  "risco_defasagem": 1,
  "probabilidade_risco": 0.7234,
  "mensagem": "ALERTA: Risco detectado!"
}
```

**Output gerado (aluno sem risco):**

```json
{
  "risco_defasagem": 0,
  "probabilidade_risco": 0.1520,
  "mensagem": "Risco baixo"
}
```

### POST /reload -- Recarregar modelo

```bash
curl -X POST http://localhost:8000/reload
```

```json
{
  "status": "sucesso",
  "mensagem": "Modelo atualizado com a última versão de produção!"
}
```

### POST /retrain -- Retreinamento

```bash
curl -X POST http://localhost:8000/retrain
```

```json
{
  "status": "sucesso",
  "mensagem": "Treinamento iniciado em segundo plano. Acompanhe os logs no Grafana/Loki para ver o progresso e o resultado."
}
```

---

## 5. Etapas do Pipeline de Machine Learning

### 5.1 Carregamento e Unificação dos Dados (`src/utils.py`)

- Carrega os CSVs de 2022, 2023 e 2024 (separadores `;` ou `,`).
- Padroniza os nomes das colunas (ex: `"IAA 2022"` vira `"IAA"`, `"Defas"` vira `"Defasagem"`).
- Remove espaços invisíveis dos headers.
- Unifica em um único DataFrame com coluna `Ano_Base`.

### 5.2 Pré-processamento dos Dados (`src/preprocessing.py`)

- Converte `Defasagem` para numérico e remove linhas com valores nulos.
- Trata idades corrompidas no formato Excel (ex: `"1/17/00"` -> `17`).
- Normaliza colunas numéricas: remove pontos de milhar, troca vírgula por ponto decimal.
- Aplica clipping em notas (0–10) e filtra idades fora da faixa 5–30.
- Normaliza colunas categóricas: remove acentos, converte para UPPER, substitui strings `"NAN"` por nulos reais.

### 5.3 Engenharia de Features (`src/feature_engineering.py`)

- **Target**: `alvo_risco = 1` se `Defasagem < 0`, caso contrário `0`.
- **Interações numéricas**: `IEG × IDA` (esforço vs resultado), `IEG × IAA` (esforço vs autoimagem), `IPS × IDA` (psicológico vs resultado).
- **Conversão de Fase**: extrai número da string (ex: `"FASE 8"` -> `8`, `"ALFA"` -> `0`).
- **Remoção de leakage**: elimina `INDE`, `IAN`, `Pedra` (derivada das faixas do INDE), `Ano_Base` e colunas identificadoras (RA, Nome, etc.).
- **Remoção de duplicação**: a coluna categórica `Fase` é removida após a criação de `Fase_Num`, evitando que a mesma informação entre no modelo duas vezes (via OneHotEncoder e como variável numérica).

### 5.4 Treinamento e Validação (`src/train.py`)

- Split estratificado: 80% treino / 20% teste.
- Pipeline sklearn com:
  - `SimpleImputer(strategy='median')` para numéricas.
  - `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` para categóricas.
  - `RandomForestClassifier` com `class_weight='balanced'`.
- Otimização via `RandomizedSearchCV` (20 iterações, 3-fold CV, `scoring='recall'`).
- Hiperparâmetros explorados: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`.

### 5.5 Avaliação (`src/evaluate.py`)

- Métricas calculadas com limiar customizado (0.40): acuracia, precisao, recall, F1-score.
- Matriz de confusão salva como artefato no MLflow.
- Importância de features extraída do classificador para investigar possível leakage.
- Relatório completo via `classification_report`.

### 5.6 Pós-processamento (Inferência na API)

- Categorias normalizadas (NFKD, ASCII, UPPER) para manter consistência com o treino.
- Features de interação recriadas no momento da predição (`IEG×IDA`, `IEG×IAA`, `IPS×IDA`, `Fase_Num`).
- `Pedra` não é solicitada na API (leakage). `Fase` é convertida em `Fase_Num` e não entra como categórica.
- Limiar de decisão configurável via variável de ambiente `LIMIAR_FIXO`.

---

## 6. Monitoramento Contínuo

O monitoramento é implementado com 4 componentes orquestrados via Docker Compose:

| Componente | Função |
|-----------|--------|
| **Prometheus** | Coleta métricas da API (`/metrics`) e do host (node-exporter) a cada 5s |
| **Grafana** | Dashboards visuais para predições, probabilidades e infraestrutura |
| **Loki** | Armazena logs centralizados |
| **Promtail** | Coleta logs da API (`logs/api_events.log`) e envia para o Loki |

### Métricas customizadas expostas pela API

| Metrica | Tipo | Descrição |
|---------|------|-----------|
| `modelo_predicoes_total` | Counter | Total de predições por tipo de risco |
| `modelo_probabilidade_risco` | Histogram | Distribuição das probabilidades geradas |
| `feature_input_iaa` | Gauge | Ultimo valor de IAA recebido (drift) |
| `feature_input_ieg` | Gauge | Ultimo valor de IEG recebido (drift) |

Os dashboards Grafana provisionados automaticamente permitem acompanhar visualmente a distribuição das predições e detectar possíveis desvios (drift) nos valores das features de entrada ao longo do tempo.
