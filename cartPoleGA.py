"""
Serie de funcoes que implementa um algoritmo genetico para solucionar um problema de controle do jogo CartPole 
"""
import gym
import random
import numpy as np
import pandas as pd
from functools import reduce
from operator import add
from statistics import median, mean
from collections import Counter
from keras.models import Sequential # cria uma pilha de camadas de redes neurais
from keras.layers import Dense, Activation

env = gym.make("CartPole-v2")
N_FRAMES = 100

def rede_neural(input_shape=4, output_shape=2):
    """Cria uma rede neural
        Args:
            input_shape (int): Entrada da rede, no caso do problema do CartPole, default de 4
            output_shape (int): Saída da rede, última cada.
        """
    model = Sequential()
    model.add(Dense(units=4, input_dim=input_shape))
    model.add(Activation("relu"))
    model.add(Dense(units=output_shape))
    model.add(Activation("softmax"))

    return model

def population(size):
    """Define uma população inicial de redes neurais.
        Args:
            size (int): tamanho da população

        Retorna:
            pop (list): População de tuplas. Cada tupla é composta por (fitness, rede neural)
        """
    pop = []
    for i in range(size):
        model = rede_neural()
        fit = fitness(model)
        
        pop.append((fit,model))

    return pop

def fitness(rede_neural, it = 10):
    """Calcula o fitness para um modelo dado como entrada. 
        Args:
            model (Sequential): rede neural
            it (int): Número de vezes que o jogo recomeçará;

        Retorna:
            score (float): Média dado it iterações dos scores obtidos pela rede. 
        """

    prev_obs = []
    scores = []
    for _ in range(it):
        env.reset()
        score = 0
        for _ in range(N_FRAMES):
            if len(prev_obs)==0:
                # primeiro movimento é difinido de forma aleatória
                action = random.randrange(0,2)
            else:
                action = np.argmax(rede_neural.predict(prev_obs.reshape(1, 4)))

            new_observation, reward, done, info = env.step(action)
            
            score += reward
            prev_obs = new_observation
            if done: break
        scores.append(score)
    summed = sum(scores)

    return summed /float(len(scores))

def crossover(pop, mother, father):
    """ Realiza o crossover entre dois modelos. 
        Args:
            pop (list): lista de redes neurais
            mother (int): índice do componente mae que realizará crossover
            father (int): índice do componente pai que realizará crossover

        Retorna:
            tupla (np.array, np.array): Tupla com pesos de crossover entre pai e mãe
        """

    population = [a[1] for a in pop] # Seleciona somente as redes neurais da lista (fit, rede_neural)
    
    w_mother = population[mother].get_weights()
    w_father = population[father].get_weights()

    for _ in range(3):
        point = random.sample([0,1,2,3], 1)[0]

        if point == 0 or point == 2: # ponto de crossover ocorrerá nas matrizes de pesos, isto é, em uma matriz de pesos 4x4
            inside_point = random.sample([0,1,2,3], 1)[0] # cada ponto representa uma linha da matriz de pesos.

            w_new_mother = w_mother[point][inside_point]
            w_new_father = w_father[point][inside_point]

            w_mother[point][inside_point] = w_new_father
            w_father[point][inside_point] = w_new_mother

        else: # troca somente os bias entre mother e father
            w_new_mother = w_mother[point]
            w_new_father = w_father[point]
            w_mother[point] = w_new_father
            w_father[point] = w_new_mother
    
    return np.asarray([w_mother, w_father])

def mutate(weights):
    """ Realiza a mutação nos pesos. 
        Args:
            weights (np..array): matriz com pesos da rede neural

        Retorna:
            weights (np.array): matriz com pesos da rede neural modificados pela mutação
        """
    mutate_layer = random.sample([0,1,2,3], 1)[0]
    mutate_point = random.sample([0,1,2,3], 1)[0]
    change = random.uniform(-0.50,0.50)

    try:
        weights[mutate_layer][mutate_point] += change
    except:
        weights[mutate_layer] += change
       
    return weights


def evolve(population, retain_lenght=0.4, mutate_rate=0.3, max_pop=10):
    """ Evolui a população de redes neurais. 
        Args:
           population (list): lista com tuplas (fitness, rede_neural) para cada modelo da população
           retain_lenght (float): Taxa de seleção. Default 0.4, significa que vamos manter os 40% melhores a cada geração
           mutate_rate (float): Taxa de mutação.
           max_pop (int): Número de integrantes da populaçao.

        Retorna:
            parents(list): Retorna uma lista com os novos membros da população
    """
    new_weights = []
    graded = population

    # Aplica o sort baseado nos scores
    graded = sorted(population, key=lambda x: x[0], reverse=True)

    retained = int(len(graded)*retain_lenght)
    parents = graded[:retained]
    #print(parents)

    parents_length = len(parents)
    desired_length = max_pop - parents_length
    children = [] 
    
    # Adiciona children, gerados a partir de dois indivíduos com melhores fitness selecionados.
    while len(children) < desired_length:
        # Seleciona mother e father aleatoriamente
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)
    
        if male != female:
            new_weights1 = crossover(parents, female, male)
            male = rede_neural()
            female = rede_neural()

            if mutate_rate > random.random():
                new_weights1[0] = mutate(new_weights1[0])
                new_weights1[1] = mutate(new_weights1[1])
                
                
            female.set_weights(new_weights1[0])
            male.set_weights(new_weights1[1])
                
            if len(children) < desired_length:
                fit_male = fitness(male)
                fit_female = fitness(female)
                if fit_male > fit_female:
                    children.append((fit_male,male))
                else:
                    children.append((fit_female, female))
                    
    parents.extend(children)

    return parents

def show_game(rede_neural, n_jogos = 1):
    env = gym.make("CartPole-v2")
    scores = []
    choices = []
    for each_game in range(n_jogos):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(1000):
            env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(rede_neural.predict(prev_obs.reshape(1,4)))

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)
    env.close()

    print('Fitness Médio:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))