# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:46:58 2026

@author: Santiago
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random

def generar_caso_de_uso_entrenar_modelo_energia():
    n = random.randint(15, 20)
    x = np.random.uniform(18, 35, n)
    y = 5 + 1.5*(x**2) + np.random.normal(0, 2, n)
    df_input = pd.DataFrame({'temp': x, 'consumo': y})
    
    df_gt = df_input.copy()
    df_gt['x_cuadrada'] = df_gt['temp']**2
    X = df_gt[['temp', 'x_cuadrada']]
    model = LinearRegression().fit(X, df_gt['consumo'])
    
    output_dict = {
        'coeficientes': model.coef_,
        'intercepto': model.intercept_,
        'r2': model.score(X, df_gt['consumo'])
    }
    
    return {'df': df_input, 'x_col': 'temp', 'y_col': 'consumo'}, output_dict

# --- Bloque de Prueba ---
inputs, resultados = generar_caso_de_uso_entrenar_modelo_energia()
print("--- TEST PREGUNTA 0004 ---")
print("Métricas del modelo:")
for k, v in resultados.items():
    print(f"{k}: {v}")