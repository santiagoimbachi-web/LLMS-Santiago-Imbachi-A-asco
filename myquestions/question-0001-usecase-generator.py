# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:41:00 2026

@author: Santiago
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random

def generar_caso_de_uso_analizar_prioridad_retencion():
    n_rows = random.randint(15, 25)
    data = {
        'antiguedad': np.random.randint(1, 60, size=n_rows),
        'clv': np.random.uniform(1000, 5000, size=n_rows),
        'soporte': [random.choice(['Si', 'No']) for _ in range(n_rows)],
        'churn': [random.choice([0, 1, np.nan]) for _ in range(n_rows)]
    }
    df_input = pd.DataFrame(data)
    target = 'churn'
    clv = 'clv'
    thr = 0.6
    
    df_gt = df_input.dropna(subset=[target]).copy()
    X = pd.get_dummies(df_gt.drop(columns=[target]), drop_first=True)
    y = df_gt[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df_gt['probabilidad_churn'] = model.predict_proba(X)[:, 1]
    p80 = df_gt[clv].quantile(0.8)
    df_gt['prioridad_alta'] = (df_gt['probabilidad_churn'] >= thr) & (df_gt[clv] > p80)
    df_output = df_gt[df_gt['prioridad_alta']].sort_values(by='probabilidad_churn', ascending=False)
    
    return {'df': df_input, 'target_col': target, 'clv_col': clv, 'threshold': thr}, df_output

# --- Bloque de Prueba ---
inputs, expected = generar_caso_de_uso_analizar_prioridad_retencion()
print("--- TEST PREGUNTA 0001 ---")
print("Input (Primeras 3 filas):\n", inputs['df'].head(3))
print("\nOutput esperado (Prioridad Alta):\n", expected)