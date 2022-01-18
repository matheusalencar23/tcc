import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pygad
import os
from datetime import datetime
import csv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(ROOT_DIR, 'data.csv')

MAX_ITER_RN = 100
NUM_GERACOES = 1000
NUM_GENES = 118
TAM_TREINO = 0.9
TAMS_POPS = [10, 100, 1000, 5000, 10000]

dados = pd.read_csv(FILE_DIR)
velocidade = np.asarray(dados.iloc[:, 0])
temperatura = np.asarray(dados.iloc[:, 1])
preenchimento = np.asarray(dados.iloc[:, 2])
espessura = np.asarray(dados.iloc[:, 3])
orientacao = np.asarray(dados.iloc[:, 4])
resistencia = np.asarray(dados.iloc[:, 5])

x = np.c_[velocidade, temperatura, preenchimento, espessura, orientacao]
y = resistencia
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, train_size=TAM_TREINO, random_state=1)


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
                        hidden_layer_sizes=hidden_layer_sizes, n_iter_no_change=25, tol=0.00001,
                        early_stopping=True, validation_fraction=0.1).fit(x_treino, y_treino)
    score = regr.score(x_teste, y_teste)
    return score


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
                        hidden_layer_sizes=hidden_layer_sizes, n_iter_no_change=25, tol=0.00001,
                        early_stopping=True, validation_fraction=0.1).fit(x_treino, y_treino)
    pred = regr.predict(x_teste)
    score = regr.score(x_teste, y_teste)
    mse = mean_squared_error(y_teste, pred)
    return pred, learning_rate_init, beta_1, beta_2, epsilon, hidden_layer_sizes, score, mse


def on_start(model):
    print('----------------------- Algoritmo Genético Iniciado {} -------------------------'.format(datetime.today()))
    arquivo = open(os.path.join(ROOT_DIR, "./tempos.txt"), "a")
    arquivo.write(
        '----------------------- Algoritmo Genético Iniciado {} -------------------------\n'.format(datetime.today()))
    arquivo.write('Tamanho da população {}\n'.format(model.pop_size))
    arquivo.close()
    print('Tamanho da população {}'.format(model.pop_size))


def on_generation(model):
    print("Geração {}/{}".format(model.generations_completed, NUM_GERACOES))
    solution, solution_fitness, solution_idx = model.best_solution()
    print("Aptidão do melhor indivíduo: {}".format(solution_fitness))
    print(model.last_generation_fitness)


def on_stop(model, aptidoesFinais):
    print('--------------------- Algoritmo Genético Finalizado {} -----------------------'.format(datetime.today()))
    arquivo = open(os.path.join(ROOT_DIR, "./tempos.txt"), "a")
    arquivo.write(
        '----------------------- Algoritmo Genético Finalizado {} -------------------------\n'.format(datetime.today()))
    arquivo.close()


with open(os.path.join(ROOT_DIR, 'tabela_dados.csv'), 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Tamanho da população', 'Teste', 'Melhor indivíduo', 'Aptidão do melhor indivíduo',
                    'learning_rate_init', 'beta_1', 'beta_2', 'epsilon', 'hidden_layer_sizes', 'r^2', 'mean_squared_error'])
    for tam_pop in TAMS_POPS:
        for i in range(1):
            print("------------------------------------------------- Iteração #{} ------------------------------------------------".format(i + 1))
            model = pygad.GA(num_generations=NUM_GERACOES, num_parents_mating=tam_pop,
                                fitness_func=aptidao, sol_per_pop=tam_pop, keep_parents=int(
                                    tam_pop/4),
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
            plt.subplots_adjust(wspace=0.25)
            plt.plot(model.best_solutions_fitness)
            plt.title('Aptidão x Geração ({})'.format(tam_pop))
            plt.xlabel('Geração')
            plt.ylabel('Aptidão')
            plt.subplot(1, 2, 2)
            plt.plot(y_teste, pred, 'ro')
            plt.plot(np.linspace(0, 60, 100), np.linspace(0, 60, 100), 'b')
            plt.title('Real x Predição ({})'.format(tam_pop))
            plt.xlabel('Real')
            plt.ylabel('Predição')
            if not os.path.exists(os.path.join(ROOT_DIR, 'images')):
                os.makedirs(os.path.join(ROOT_DIR, 'images'))
            plt.savefig(os.path.join(ROOT_DIR, 'images', '{}_{}.png'.format(tam_pop, i + 1)), format="png")
            # plt.show(block=False)
            # plt.close()

            arquivo = open(os.path.join(ROOT_DIR, "testes.txt"), "a")
            arquivo.write(
                '----------------------------------------------------------------------------------------------------------------------\n')
            arquivo.write('Tamanho da população {}\n'.format(tam_pop))
            arquivo.write('Teste {}\n'.format(i + 1))
            arquivo.write("Melhor indivíduo: {}\n".format(solution))
            arquivo.write(
                "Aptidão do melhor indivíduo: {}\n".format(solution_fitness))
            arquivo.write(
                'learning_rate_init: {}\n'.format(learning_rate_init))
            arquivo.write('beta_1: {}\n'.format(beta_1))
            arquivo.write('beta_2: {}\n'.format(beta_2))
            arquivo.write('epsilon: {}\n'.format(epsilon))
            arquivo.write(
                'hidden_layer_sizes: {}\n'.format(hidden_layer_sizes))
            arquivo.write('r^2: {}\n'.format(score))
            arquivo.write('mean_squared_error: {}\n'.format(mse))
            arquivo.write("\n")
            arquivo.close()

            writer.writerow([tam_pop, i + 1, solution, solution_fitness, learning_rate_init,
                            beta_1, beta_2, epsilon, hidden_layer_sizes, score, mse])
