import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from geneticalgorithm import geneticalgorithm as ga
import random as rd

dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:,0])
temperatura = np.asarray(dados.iloc[:,1])
preenchimento = np.asarray(dados.iloc[:,2])
espessura = np.asarray(dados.iloc[:,3])
orientacao = np.asarray(dados.iloc[:,4])
resistencia = np.asarray(dados.iloc[:,5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

def conversorBinarioReal(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[i] * (2 ** (- i - 1))
    return v

def aptidao(x):
    learning_rate_init=conversorBinarioReal(x[0:29])
    regr = MLPRegressor(random_state=1, max_iter=10000, learning_rate_init=learning_rate_init).fit(x_treino, y_treino)
    # pred = regr.predict(x_teste)
    score = regr.score(x_teste, y_teste)
    return -score

pop_i = np.array([[0,1]]*50)

model = ga(function=aptidao, dimension=50, variable_type='int', variable_boundaries=pop_i)
model.run()
# print(score)
# x_linha = np.linspace(0, 60, 1000)
# plt.plot(y_teste, pred, 'bo')
# plt.plot(x_linha, x_linha, 'r')
