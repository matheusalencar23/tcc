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
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, train_size=0.9, test_size=0.1, random_state=1)


def conversorBinarioReal(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[i] * (2 ** (- i - 1))
    if v > 0:
        return v
    else:
        return v + 0.0000000001


def conversorBinarioInteiro(binario):
    v = 0
    for i in range(len(binario)):
        v += binario[len(binario) - i - 1] * 2**(i)
    if v > 0:
        return int(v)
    else:
        return int(1)


def aptidao(x):
    learning_rate_init = conversorBinarioReal(x[:25])
    beta_1 = conversorBinarioReal(x[25:50])
    beta_2 = conversorBinarioReal(x[50:75])
    epsilon = conversorBinarioReal(x[75:100])
    hidden_layer_sizes = (
        conversorBinarioInteiro(x[100:106]),
        conversorBinarioInteiro(x[106:112]),
        conversorBinarioInteiro(x[112:]))
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init,
                        max_iter=5000, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes).fit(x_treino, y_treino)
    pred = regr.predict(x_teste)
    score = regr.score(x_teste, y_teste)
    mse = mean_squared_error(y_teste, pred)
    print('learning_rate_init ', learning_rate_init)
    print('beta_1 ', beta_1)
    print('beta_2 ', beta_2)
    print('epsilon ', epsilon)
    print('hidden_layer_sizes ', hidden_layer_sizes)
    print('score ', score)
    print('mse ', mse)
    return pred



indStr = '0000000010000000000000000101011000000001000000000010000000000000000000010000000000000000000000000001001011001011001010'
ind = []
for i in indStr:
    ind.append(int(i))
pred = aptidao(ind)
plt.plot(y_teste, pred, 'ro')
x = np.linspace(0, 60, 100)
plt.plot(x, x, 'b')
plt.show()

