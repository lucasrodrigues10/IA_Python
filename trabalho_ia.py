# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:46:49 2018

@author: Lucas de Moura Rodrigues
@ra: 14.00556-5

"""
import time

import numpy as np
import pandas as pd

tempo_inicial = time.time()

# le o csv
print('Le e trata os dados: ', round(time.time() - tempo_inicial, 3))
precos_casa = pd.read_csv("precos_casa_california.csv")

# trata o dataframe
precos_casa = precos_casa.replace(' ', np.nan)
precos_casa = precos_casa.dropna()
colunas = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
for i in colunas:
    precos_casa[i] = pd.to_numeric(precos_casa[i])
    precos_casa = precos_casa.loc[precos_casa[i] < precos_casa[i].quantile(0.95)]
precos_casa = precos_casa.reset_index(drop=True)

# separa o dataframe
dados = precos_casa.loc[:, precos_casa.columns != 'median_house_value']
target = precos_casa.loc[:, precos_casa.columns == 'median_house_value']

# separa o dataframe: dados_treino = 0 - 60%  dados_validacao = 60 - 100%
tamanho_precos_casa = len(precos_casa)
index_60 = int(tamanho_precos_casa * 0.6)  # primeiros 60% do dataframe
dados_treino = dados[:index_60]
dados_validacao = dados[index_60:]
target_treino = target[:index_60]
target_validacao = target[index_60:]

# numeriza o ocean
dic_ocean = {0: 'NEAR BAY', 1: '<1H OCEAN', 2: 'INLAND', 3: 'NEAR OCEAN', 4: 'ISLAND'}
for chave, valor in dic_ocean.items():
    dados_treino.replace(valor, chave, inplace=True)
    dados_validacao.replace(valor, chave, inplace=True)

# machine learning
print('Predicao dos Dados: ', round(time.time() - tempo_inicial, 3))
clf = linear_model.SGDClassifier(loss="hinge", penalty="l2")
clf.fit(dados_treino, target_treino)
predicao = clf.predict(dados_treino)
resultado = target_treino.reset_index(drop=True).copy()
predicao = [float(i) for i in predicao]

# compara os dados
print('Comparando os Dados: ', round(time.time() - tempo_inicial, 3))
resultado['predicao'] = pd.Series(predicao)
resultado = resultado.rename(index=str, columns={'median_house_value': 'Valor Correto', 'predicao': 'Valor_Calculado'})
resultado['Diferenca_Absoluta'] = resultado['Valor Correto'] - resultado['Valor_Calculado']
resultado['Diferenca_Porcentagem'] = (resultado['Valor Correto'] / resultado['Valor_Calculado'] - 1) * 100

# salva o resultado em um excel
excel = pd.ExcelWriter(F"Resultado.xlsx")
precos_casa.to_excel(excel, sheet_name='Precos_Casa_Tratado')
resultado.to_excel(excel, sheet_name='Comparacao')
excel.save()

print('Tempo do Programa: ', round(time.time() - tempo_inicial, 3))
