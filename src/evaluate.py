import pandas as pd
from sklearn.metrics import classification_report, recall_score, confusion_matrix, accuracy_score, f1_score, precision_score

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Calcula métricas com base em um limiar de decisão customizado.
    
    Args:
        model: O modelo treinado.
        X_test: Features de teste.
        y_test: Target real de teste.
        threshold (float): Limiar de corte (ex: 0.45). Se prob > threshold, é risco.
    """

    print("\n" + "="*100)
    print(f"MELHORES PARÂMETROS:")
    print(f"{model.get_params()}")
    print("\n" + "="*100)
    

    # Predição de Probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Aplicação do Threshold
    y_pred = (y_proba >= threshold).astype(int)


    # Importância das Features (CORREÇÃO APLICADA AQUI)
    print("\n" + "="*100)
    print("IMPORTANCIA DAS FEATURES E INVESTIGAÇÃO DE POSSÍVEL VAZAMENTO DE DADOS:")
    print("="*100)

    # Verifica se é um Pipeline e extrai o classificador
    classifier = model
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
    
    if hasattr(classifier, 'feature_importances_'):
        importances = pd.Series(data=classifier.feature_importances_, index=X_test.columns)
        print(importances.sort_values(ascending=False))
    else:
        print("Feature importance indisponível para este estimador.")
    
    # Cálculo das Métricas
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*100)
    print(f"RESULTADOS DA AVALIAÇÃO DO MODELO (Limiar: {threshold})")
    print("="*100)
    print(f"ACURÁCIA:              {accuracy:.2%}")
    print(f"PRECISÃO:              {precision:.2%}")
    print(f"* RECALL:              {recall:.2%}")
    print(f"F1-SCORE:              {f1:.2%}")
    print("* RECALL É A MÉTRICA MAIS IMPORANTE POIS ELA REPRESENTA A SENSIBILIADE DO NOSSO MODELO")
    print("-" * 100)
    
    print("\nMatriz de Confusão:")
    print(f"Verdadeiros Negativos: {cm[0][0]} | Falsos Positivos: {cm[0][1]}")
    print(f"Falsos Negativos:      {cm[1][0]} | Verdadeiros Positivos: {cm[1][1]}")
    
    print("\nRelatório Completo:")
    print(classification_report(y_test, y_pred))
    
    return {
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }