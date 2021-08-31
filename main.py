from numpy.core.shape_base import block
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pygad
import sys
sys.tracebacklimit = 0

dados = pd.read_csv('./data.csv')
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, train_size=0.9, random_state=1)


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


MAX_ITER_RN = 100


def aptidao(x, i):
    learning_rate_init = conversorBinarioReal(x[:25])
    beta_1 = conversorBinarioReal(x[25:50])
    beta_2 = conversorBinarioReal(x[50:75])
    epsilon = conversorBinarioReal(x[75:100])
    hidden_layer_sizes = (
        conversorBinarioInteiro(x[100:106]),
        conversorBinarioInteiro(x[106:112]),
        conversorBinarioInteiro(x[112:]))
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init, shuffle=True,
                        max_iter=MAX_ITER_RN, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes, n_iter_no_change=10, tol=0.00001,
                        early_stopping=True, validation_fraction=0.1).fit(x_treino, y_treino)
    score = regr.score(x_teste, y_teste)
    if score and score > 0:
        return score
    return 0


def predicao(x):
    learning_rate_init = conversorBinarioReal(x[:25])
    beta_1 = conversorBinarioReal(x[25:50])
    beta_2 = conversorBinarioReal(x[50:75])
    epsilon = conversorBinarioReal(x[75:100])
    hidden_layer_sizes = (
        conversorBinarioInteiro(x[100:106]),
        conversorBinarioInteiro(x[106:112]),
        conversorBinarioInteiro(x[112:]))
    regr = MLPRegressor(random_state=1, learning_rate_init=learning_rate_init, shuffle=True,
                        max_iter=MAX_ITER_RN, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                        solver='adam', activation='relu', learning_rate='constant',
                        hidden_layer_sizes=hidden_layer_sizes, n_iter_no_change=10, tol=0.00001,
                        early_stopping=True, validation_fraction=0.1).fit(x_treino, y_treino)
    pred = regr.predict(x_teste)
    score = regr.score(x_teste, y_teste)
    mse = mean_squared_error(y_teste, pred)
    return pred, learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes, score, mse


NUM_GERACOES = 500
TAM_POP = 100
NUM_GENES = 118


def on_start(model):
    print('------------------------------------------------ Algoritmo Genético Iniciado -------------------------------------------------')
    print('Tamanho da população {}'.format(model.pop_size))


def on_generation(model):
    print("Geração {}/{}".format(model.generations_completed, NUM_GERACOES))
    solution, solution_fitness, solution_idx = model.best_solution()
    print("Aptidão do melhor indivíduo: {}".format(solution_fitness))
    print(model.last_generation_fitness)


def on_stop(model, aptidoesFinais):
    print('------------------------------------------------- Algoritmo Genético Finalizado ------------------------------------------------')


for i in range(1):
    print("------------------------------------------------- Iteração #{} ------------------------------------------------".format(i + 1))
    model = pygad.GA(num_generations=NUM_GERACOES, num_parents_mating=TAM_POP,
                     fitness_func=aptidao, sol_per_pop=TAM_POP, keep_parents=int(TAM_POP/4),
                     num_genes=NUM_GENES, gene_type=int,
                     init_range_low=0, init_range_high=2,
                     parent_selection_type="tournament", K_tournament=3, crossover_type="two_points",
                     crossover_probability=0.9, mutation_type="random", suppress_warnings=True,
                     mutation_probability=0.05, on_start=on_start, on_stop=on_stop,
                     on_generation=on_generation, stop_criteria="saturate_10")
    model.run()
    solution, solution_fitness, solution_idx = model.best_solution()
    pred, learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes, score, mse = predicao(
        solution)

    print("Melhor indivíduo: {}".format(solution))
    print("Aptidão do melhor indivíduo: {}".format(solution_fitness))
    print('learning_rate_init: ', learning_rate_init)
    print('beta_1: ', beta_1)
    print('beta_2: ', beta_2)
    print('epsilon: ', epsilon)
    print('hidden_layer_sizes: ', hidden_layer_sizes)
    print('r^2: ', score)
    print('mean_squared_error: ', mse)
    print("\n")

    plt.subplot(1, 2, 1)
    plt.plot(model.best_solutions_fitness)
    plt.title('Aptidão x Geração')
    plt.xlabel('Geração')
    plt.ylabel('Aptidão')
    plt.subplot(1, 2, 2)
    plt.plot(y_teste, pred, 'ro')
    plt.plot(np.linspace(0, 60, 100), np.linspace(0, 60, 100), 'b')
    plt.title('Real x Predição')
    plt.xlabel('Real')
    plt.ylabel('Predição')
    plt.savefig('./images/{}.png'.format(i + 1), format="png")
    plt.show(block=False)
    plt.close()

    arquivo = open("./testes.txt", "a")
    arquivo.write(
        '----------------------------------------------------------------------------------------------------------------------\n')
    arquivo.write('Teste {}\n'.format(i + 1))
    arquivo.write("Melhor indivíduo: {}\n".format(solution))
    arquivo.write("Aptidão do melhor indivíduo: {}\n".format(solution_fitness))
    arquivo.write('learning_rate_init: {}\n'.format(learning_rate_init))
    arquivo.write('beta_1: {}\n'.format(beta_1))
    arquivo.write('beta_2: {}\n'.format(beta_2))
    arquivo.write('epsilon: {}\n'.format(epsilon))
    arquivo.write('hidden_layer_sizes: {}\n'.format(hidden_layer_sizes))
    arquivo.write('r^2: {}\n'.format(score))
    arquivo.write('mean_squared_error: {}\n'.format(mse))
    arquivo.write("\n")
    arquivo.close()
