import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from geneticalgorithm import geneticalgorithm as ga
import random as rd

dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.9, test_size=0.1, random_state=1)

print(x_treino, x_teste, y_treino, y_teste)

def conversorBinarioReal(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[i] * (2 ** (- i - 1))
    if (v > 0):
        return v
    else:
        return 0.000000001


def aptidao():
    regr = MLPRegressor(random_state=1, max_iter=10000,
                        solver='sgd', activation='logistic').fit(x_treino, y_treino)
    score = regr.score(x_teste, y_teste)
    pred = regr.predict(x_teste)
    mse = mean_squared_error(y_teste, pred)
    print(score, mse)
    return pred

pred = aptidao()
x_linha = np.linspace(0, 60, 1000)
plt.plot(y_teste, pred, 'bo')
plt.plot(x_linha, x_linha, 'r')
plt.show()
