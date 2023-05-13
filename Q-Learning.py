import random
import numpy as np
import matplotlib.pyplot as plt

# Dimensiones del tablero
board_size = 5

# Posición inicial y objetivo aleatorias
initial_pos = (random.randint(0, board_size-1), random.randint(0, board_size-1))
goal_pos = (random.randint(0, board_size-1), random.randint(0, board_size-1))

# Función para mover el agente en el tablero
def move(pos, action):
    if action == 0: # arriba
        new_pos = (pos[0], max(pos[1]-1, 0))
    elif action == 1: # abajo
        new_pos = (pos[0], min(pos[1]+1, board_size-1))
    elif action == 2: # izquierda
        new_pos = (max(pos[0]-1, 0), pos[1])
    elif action == 3: # derecha
        new_pos = (min(pos[0]+1, board_size-1), pos[1])
    return new_pos

# Parámetros del algoritmo
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Inicialización de la tabla Q
Q = np.zeros((board_size, board_size, 4))

# Algoritmo de aprendizaje por refuerzo activo
scores = []
for i in range(num_episodes):
    pos = initial_pos
    done = False
    score = 0
    while not done:
        # Selección de acción
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            action = np.argmax(Q[pos[0], pos[1], :])

        # Ejecución de acción y actualización de tabla Q
        new_pos = move(pos, action)
        if new_pos == goal_pos:
            reward = 10
            done = True
        else:
            reward = -1
        Q[pos[0], pos[1], action] += learning_rate * (reward + discount_factor * np.max(Q[new_pos[0], new_pos[1], :]) - Q[pos[0], pos[1], action])
        pos = new_pos
        score += reward
    scores.append(score)

# Graficar los resultados
plt.plot(scores)
plt.xlabel("Episodio")
plt.ylabel("Puntaje")
plt.title("Aprendizaje por refuerzo activo para el problema del agente en el tablero")
plt.show()
