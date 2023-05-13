import gym
import numpy as np
import matplotlib.pyplot as plt

# Configuración del entorno
env = gym.make('Blackjack-v0')

# Definición de variables
num_episodes = 100000
returns_sum = np.zeros((32, 11, 2))
returns_count = np.zeros((32, 11, 2))
V = np.zeros((32, 11))

# Algoritmo de valoración de Monte Carlo
for i in range(num_episodes):
    # Generar un episodio
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        
    # Actualización de la función de valor
    states, _, rewards = zip(*episode)
    for j, state in enumerate(states):
        returns_sum[state][action][0 if rewards[j] == 0 else 1] += sum(rewards[j:])
        returns_count[state][action][0 if rewards[j] == 0 else 1] += 1
        V[state][action] = returns_sum[state][action][1] / returns_count[state][action][1]

# Graficar la función de valor final
x_range = np.arange(11)
y_range = np.arange(32)
X, Y = np.meshgrid(x_range, y_range)
Z = V[Y, X]
fig, ax = plt.subplots()
heatmap = ax.pcolormesh(X, Y, Z, cmap='coolwarm')
cbar = plt.colorbar(heatmap)
plt.xlabel('Dealer showing')
plt.ylabel('Player sum')
plt.title('Value function')
plt.show()
