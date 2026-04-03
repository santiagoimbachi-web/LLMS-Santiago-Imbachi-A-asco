# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:45:39 2026

@author: Santiago
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random

def generar_caso_de_uso_segmentar_productos():
    n = random.randint(20, 30)
    df_input = pd.DataFrame({
        'precio': np.random.uniform(10, 500, n),
        'cantidad_vendida': np.random.randint(1, 1000, n)
    })
    df_input.iloc[5, 0] = np.nan 
    cols = ['precio', 'cantidad_vendida']
    
    df_gt = df_input.copy()
    for c in cols:
        df_gt[c] = df_gt[c].fillna(df_gt[c].median())
    km = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df_gt['cluster'] = km.fit_predict(df_gt[cols])
    promedios_precio = df_gt.groupby('cluster')['precio'].mean()
    
    return {'df': df_input, 'columnas_features': cols}, (df_gt, promedios_precio)

# --- Bloque de Prueba ---
inputs, (df_clustered, promedios) = generar_caso_de_uso_segmentar_productos()
print("--- TEST PREGUNTA 0003 ---")
print("Promedio de precio por cluster:\n", promedios)
print("\nDataFrame con clusters (primeras 5 filas):\n", df_clustered.head())