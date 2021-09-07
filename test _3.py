import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
bias = np.ones((len(velocidade)))
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao, bias]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, train_size=0.9, random_state=1)


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


def regr(ind, x, y):
    learning_rate_init = conversorBinarioReal(ind[:25])
    beta_1 = conversorBinarioReal(ind[25:50])
    beta_2 = conversorBinarioReal(ind[50:75])
    epsilon = conversorBinarioReal(ind[75:100])
    hidden_layer_sizes = (
        conversorBinarioInteiro(ind[100:106]),
        conversorBinarioInteiro(ind[106:112]),
        conversorBinarioInteiro(ind[112:]))
    print(learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes)
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init, shuffle=True,
                        max_iter=1000, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes).fit(x, y)
    return regr


ind = np.asarray([0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,0,1
,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0
,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,0,1,0,1,1,1,0,1,0,0,1,1,0
,1,1,1,1,0,0,0])
regr_treino = regr(ind, x_treino, y_treino)
regr_teste = regr(ind, x_teste, y_teste)
pred_treino = regr_treino.predict(x_treino)
pred_teste = regr_teste.predict(x_teste)
print('R^2 treino: {}'.format(regr_teste.score(x_treino, y_treino)))
print('MSE treino: {}'.format(mean_squared_error(pred_treino, y_treino)))
print('EVS treino: {}'.format(explained_variance_score(pred_treino, y_treino)))
print('R^2 teste: {}'.format(regr_teste.score(x_teste, y_teste)))
print('MSE teste: {}'.format(mean_squared_error(pred_teste, y_teste)))
print('EVS teste: {}'.format(explained_variance_score(pred_teste, y_teste)))

plt.plot(regr_treino.loss_curve_, 'b')
plt.plot(regr_teste.loss_curve_, 'r')
plt.show()
