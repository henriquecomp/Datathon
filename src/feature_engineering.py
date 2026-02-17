import pandas as pd
import numpy as np

def create_features(df):
    """
    Cria o target, remove colunas de leakage e imputa nulos.
    """
    df = df.copy()
    
    # Criação do Target (Risco)
    if 'Defasagem' not in df.columns:
        raise ValueError("Coluna 'Defasagem' necessária para criar o target.")
        
    # Defasagem Negativa = Risco (Classe 1)
    df['alvo_risco'] = np.where(df['Defasagem'] < 0, 1, 0)
    
    # Seleção de Features e Remoção de Vazamento
    # Removemos:
    # - Defasagem (Resposta)
    # - Ano_Base (Metadado)
    # - INDE e IAN (Vazamento de dados / Data Leakage confirmados)
        # IAN     0.834992
        # INDE    0.086597
    cols_to_drop = ['Defasagem', 'Ano_Base', 'INDE', 'IAN']

    # Criamos interações para capturar nuances    
    if all(c in df.columns for c in ['IEG', 'IDA', 'IAA', 'IPS']):
        df['IEG_x_IDA'] = df['IEG'] * df['IDA']  # Esforço vs Resultado
        df['IEG_x_IAA'] = df['IEG'] * df['IAA']  # Esforço vs Autoimagem
        df['IPS_x_IDA'] = df['IPS'] * df['IDA']  # Psicológico vs Resultado    
    
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Garante que o target não fique no X
    if 'alvo_risco' in X.columns:
        X = X.drop(columns=['alvo_risco'])
        
    y = df['alvo_risco']
    
    return X, y