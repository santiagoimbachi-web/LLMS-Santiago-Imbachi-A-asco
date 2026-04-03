# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:43:59 2026

@author: Santiago
"""

import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_filtrar_anomalias_sensores():
    n = random.randint(12, 18)
    temp = np.random.normal(50, 5, n).tolist()
    temp[0] = 200.0  
    temp[1] = -50.0  
    df_input = pd.DataFrame({'temp': temp})
    
    mu = df_input['temp'].mean()
    sigma = df_input['temp'].std()
    df_output = df_input.copy()
    mask_outlier = (df_output['temp'] < mu - 2*sigma) | (df_output['temp'] > mu + 2*sigma)
    df_output.loc[mask_outlier, 'temp'] = mu
    diccionario_stats = {'media': round(mu, 2), 'desviacion': round(sigma, 2)}
    
    return {'df': df_input, 'col_sensor': 'temp'}, (df_output, diccionario_stats)

# --- Bloque de Prueba ---
inputs, (df_out, stats) = generar_caso_de_uso_filtrar_anomalias_sensores()
print("--- TEST PREGUNTA 0002 ---")
print("Estadísticas calculadas:", stats)
print("\nDataFrame Procesado (primeras 5 filas):\n", df_out.head())
    