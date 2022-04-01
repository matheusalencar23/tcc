import pandas as pd
import numpy as np
import helpers
from sklearn.neural_network import MLPRegressor

dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia

arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

ind = np.asarray(arr)


def regr():
    learning_rate_init = helpers.conversorBinarioReal(ind[:25])
    beta_1 = helpers.conversorBinarioReal(ind[25:50])
    beta_2 = helpers.conversorBinarioReal(ind[50:75])
    epsilon = helpers.conversorBinarioReal(ind[75:100])
    hidden_layer_sizes = (
        helpers.conversorBinarioInteiro(ind[100:106]),
        helpers.conversorBinarioInteiro(ind[106:112]),
        helpers.conversorBinarioInteiro(ind[112:]))
    print(learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes)
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init, shuffle=True,
                        max_iter=1000, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes, early_stopping=False).fit(x, y)
    return regr


regressao = regr()
while True:
    nova_velocidade = float(input("What is the printing speed(mm/s)?\n"))
    nova_temperatura = float(input("What is the extrusion temperature(°C)?\n"))
    novo_preenchimento = float(input("What is the fill density(%)?\n"))
    nova_espessura = float(
        input("What is the thickness of the extruded filament(mm)?\n"))
    nova_orientacao = float(input("What is the extrusion orientation?(°)?\n"))
    nova_entrada = [nova_velocidade, nova_temperatura,
                    novo_preenchimento, nova_espessura, nova_orientacao]
    nova_entrada = np.reshape(nova_entrada, (1, -1))
    print(nova_entrada)
    predicao = regressao.predict(nova_entrada)
    print("The tensile strength is: {} MPa".format(predicao))
