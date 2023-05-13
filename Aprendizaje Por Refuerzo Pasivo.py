import random
import matplotlib.pyplot as plt

# Dimensiones del tablero
board_size = 5

# Posición inicial y objetivo aleatorias
initial_pos = (random.randint(0, board_size-1), random.randint(0, board_size-1))
goal_pos = (random.randint(0, board_size-1), random.randint(0, board_size-1))
print("la pocicion inicial", initial_pos)
print("la pocicion final", goal_pos)
# Función para mover el agente en el tablero
def move(pos, action):
    if action == "up":
        new_pos = (pos[0], max(pos[1]-1, 0))
    elif action == "down":
        new_pos = (pos[0], min(pos[1]+1, board_size-1))
    elif action == "left":
        new_pos = (max(pos[0]-1, 0), pos[1])
    elif action == "right":
        new_pos = (min(pos[0]+1, board_size-1), pos[1])
    return new_pos

# Función para calcular el puntaje del agente en un episodio
def run_episode():
    pos = initial_pos
    done = False
    score = 0
    while not done:
        action = random.choice(["up", "down", "left", "right"])
        pos = move(pos, action)
        if pos == goal_pos:
            done = True
            score += 10
        else:
            score -= 1
    return score

# Parámetros del algoritmo
num_episodes = 1000

# Algoritmo de aprendizaje por refuerzo pasivo
scores = []
for i in range(num_episodes):
    score = run_episode()
    scores.append(score)

# Graficar los resultados
plt.plot(scores)
plt.xlabel("Episodio")
plt.ylabel("Puntaje")
plt.title("Aprendizaje por refuerzo pasivo para el problema del agente en el tablero")
plt.show()
