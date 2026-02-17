import pandas as pd
import numpy as np

def clean_data(df):
    """
    Realiza a limpeza inicial e conversão de tipos dos dados.
    """
    # Conversão de Strings Numéricas (ex: "5,7" -> 5.7)
    numeric_cols = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE']
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove ponto e troca vírgula
            df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Garante que Defasagem seja numérica
    if 'Defasagem' in df.columns:
        df['Defasagem'] = pd.to_numeric(df['Defasagem'], errors='coerce').fillna(0)
        
    return df