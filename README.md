#  Neuroevolucao-CartPole-OpenGymAI

Códigos usado no projeto da disciplina Sistemas Inteligentes, onde desenvolvemos uma serie de funções para aplicarmos Neuroevolução (Redes Neurais evoluídas por Algoritmos Géneticos) para resolver o problema do Cart-Pole, da plataforma OpenGymAI de Aprendizado por Reforço.

Mais detalhes podem ser encontrados em um pequeno [tutorial no meu site do github](https://vtoliveira.github.io/projetos/neuroevolucao).

#  Dependências

 - Python3.6
 - Numpy
 - Pandas
 - Gym
 - Keras
 - Tensowflow
 - Jupyter Notebook
 - Anaconda
 

# Uso

Basta colocar o arquivo *cartPoleGA.py* e o notebook *cart_pole_model.ipyb* na mesma pasta e rodar o notebook a partir do jupyter com python na versão 3.6.

A versão do Anaconda3 utilizada foi para Windows. Para instalação do gym, seguir recomendação do [site](https://gym.openai.com/envs/CartPole-v1/).

Para alterar o valor máximo de frames que o jogo roda, lembrando que a versão v0 roda 200, e a versão v1 roda 500, basta alterar o __init__.py, adicionando a seguinte linha de código dentro do seu diretório do anaconda, no meu caso foi
C:\Users\VictorVT\Anaconda3\Lib\site-packages\gym\envs:

    register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
        reward_threshold=4750.0,
    )
Ou se preferir, mude dentro do arquivo cartPoleGA.py para v1 ou v0:

    env = gym.make("CartPole-v2")
# Resultados

> Written with [StackEdit](https://stackedit.io/).