import numpy as np
import random
import matplotlib.pyplot as plt

# Definition der Umgebungsparameter basierend auf dem neuen Grundriss
rooms = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
room_indices = {room: idx for idx, room in enumerate(rooms)}
num_rooms = len(rooms)

actions = ['left', 'right', 'up', 'down']
action_indices = {action: idx for idx, action in enumerate(actions)}
num_actions = len(actions)

# Übergangswahrscheinlichkeiten und Belohnungsfunktion
transition_prob = 0.9
stay_prob = 0.1
reward_step = -1
gamma = 0.9

# Q-Tabelleninitialisierung
Q = np.zeros((num_rooms, num_actions))

# Hilfsfunktion zur Wahl der Aktion basierend auf ε-greedy Politik
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(Q[state])]

# Simulationsfunktion für die Umgebung basierend auf dem neuen Grundriss
def get_next_state_and_reward(current_state, action):
    if action == 'left':
        if current_state == room_indices['D']:
            next_state = room_indices['B']
        elif current_state == room_indices['F']:
            next_state = room_indices['D']
        elif current_state == room_indices['C']:
            next_state = room_indices['A']
        elif current_state == room_indices['E']:
            next_state = room_indices['C']
        elif current_state == room_indices['G']:
            next_state = room_indices['E']
        else:
            next_state = current_state
    elif action == 'right':
        if current_state == room_indices['B']:
            next_state = room_indices['D']
        elif current_state == room_indices['D']:
            next_state = room_indices['F']
        elif current_state == room_indices['A']:
            next_state = room_indices['C']
        elif current_state == room_indices['C']:
            next_state = room_indices['E']
        elif current_state == room_indices['E']:
            next_state = room_indices['G']
        else:
            next_state = current_state
    elif action == 'up':
        if current_state == room_indices['A']:
            next_state = room_indices['B']
        elif current_state == room_indices['C']:
            next_state = room_indices['D']
        elif current_state == room_indices['E']:
            next_state = room_indices['F']
        else:
            next_state = current_state
    elif action == 'down':
        if current_state == room_indices['B']:
            next_state = room_indices['A']
        elif current_state == room_indices['D']:
            next_state = room_indices['C']
        elif current_state == room_indices['F']:
            next_state = room_indices['E']
        else:
            next_state = current_state
    else:
        next_state = current_state

    if random.uniform(0, 1) < stay_prob:
        next_state = current_state

    reward = reward_step
    return next_state, reward

# Q-Learning Algorithmus
def q_learning(episodes, alpha, epsilon):
    rewards_per_episode = []
    for _ in range(episodes):
        current_state = random.choice(range(num_rooms - 1))  # Startzustand zufällig wählen
        total_reward = 0
        while current_state != room_indices['G']:
            action = choose_action(current_state, epsilon)
            action_index = action_indices[action]
            next_state, reward = get_next_state_and_reward(current_state, action)
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[current_state, action_index]
            Q[current_state, action_index] += alpha * td_error
            current_state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)
    return rewards_per_episode

# Parameter für das Q-Learning
episodes = 1000
alpha = 0.1
epsilon = 0.9

# Q-Learning ausführen und Lernkurve speichern
rewards_per_episode = q_learning(episodes, alpha, epsilon)

# Ergebnisse anzeigen
print("Q-Tabelle nach Q-Learning:")
print(Q)

# Optimale Politik anzeigen
optimal_policy = {rooms[i]: actions[np.argmax(Q[i])] for i in range(num_rooms)}
print("Optimale Politik:")
print(optimal_policy)

# Lernkurve anzeigen
plt.figure(figsize=(10, 6))
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Gesamte Kosten')
plt.title('Lernkurve des Q-Learning Algorithmus')
plt.grid(True)
plt.show()

# Optimale Politik visuell darstellen
def plot_policy(optimal_policy):
    grid = np.array([
        ['B', 'D', 'F'],
        ['A', 'C', 'E', 'G']
    ])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != ' ':
                room = grid[i, j]
                action = optimal_policy[room]
                ax.text(j + 0.5, i + 0.5, f"{room}\n{action}", 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
                
                # Zeichne den Pfeil für die Aktion
                if action == 'left':
                    ax.arrow(j + 0.5, i + 0.5, -0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif action == 'right':
                    ax.arrow(j + 0.5, i + 0.5, 0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif action == 'up':
                    ax.arrow(j + 0.5, i + 0.5, 0, -0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif action == 'down':
                    ax.arrow(j + 0.5, i + 0.5, 0, 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('Optimale Politik')
    plt.show()

plot_policy(optimal_policy)