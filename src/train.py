import joblib
import pandas as pd
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from utils import load_data
from preprocessing import clean_data
from feature_engineering import create_features
from evaluate import evaluate_model

def run_training():
    print("Iniciando Pipeline de Treinamento com MLFLOW...")
    
    # Cria uma pasta 'mlruns' localmente para salvar os dados
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PassosMagicos_Risco_Defasagem")

    # Habilita o log automático (salva params, métricas e o modelo .pkl)
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False)

    # Definição de Caminhos
    paths = {
        '2022': 'files/PEDE2022.csv',
        '2023': 'files/PEDE2023.csv',
        '2024': 'files/PEDE2024.csv'
    }
    
    # Pipeline de Dados
    print("   [1/6] Carregando dados (Utils)...")
    try:
        df_raw = load_data(paths)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return

    print("   [2/6] Limpando dados (Preprocessing)...")
    df_clean = clean_data(df_raw)
    
    print("   [3/6] Engenharia de Features...")
    X, y = create_features(df_clean)
    
    print(f"         Features finais: {list(X.columns)}")

    # Split de Dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Criação do Pipeline
    # Imputer: Preenche nulos com a mediana
    # Model: O seu Random Forest
    model_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    # Otimização (Random Search)
    print("   [4/6] Buscando melhores hiperparâmetros...")
    
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }    
    

    with mlflow.start_run() as run:
        print(f"   [MLflow] Run iniciada. ID: {run.info.run_id}")
        random_search = RandomizedSearchCV(
            estimator=model_pipeline, 
            param_distributions=param_dist, 
            n_iter=10, 
            cv=3, 
            verbose=1, 
            random_state=42, 
            n_jobs=1, 
            scoring='recall' # Otimizando para Recall
        )
        
        print("   [5/6] Treinando o modelo...")

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Avaliação
        print("   [6/6] Avaliando...")        
        evaluate_model(best_model, X_test, y_test, threshold=0.40)

        assinatura = infer_signature(X_test, best_model.predict(X_test))
        
        # Grava o modelo no MLflow com a assinatura e registra oficialmente
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="modelo",
            signature=assinatura,
            registered_model_name="Modelo_Risco_Defasagem"
        )
        
        # Salvar
        output_dir = 'app/model'
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(best_model, 'app/model/modelo.pkl')
        print("\nModelo salvo em app/model/modelo.pkl")

if __name__ == "__main__":
    run_training()