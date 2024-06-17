from _ast import arg

import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

class QLearningEnvironment:
    def __init__(self, rooms, actions, transition_prob, stay_prob, reward_step, gamma):
        self.rooms = rooms
        self.room_indices = {room: idx for idx, room in enumerate(rooms)}
        self.num_rooms = len(rooms)
        self.actions = actions
        self.action_indices = {action: idx for idx, action in enumerate(actions)}
        self.num_actions = len(actions)
        self.transition_prob = transition_prob
        self.stay_prob = stay_prob
        self.reward_step = reward_step
        self.gamma = gamma
        self.Q = np.zeros((self.num_rooms, self.num_actions))

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.Q[state]
            max_value = np.max(q_values)
            best_actions = [action for action, q in zip(self.actions, q_values) if q == max_value]
            return min(best_actions)

    # Simulationsfunktion für die Umgebung basierend auf dem neuen Grundriss
    def get_next_state_and_reward(self, current_state, action):
        if random.uniform(0, 1) < self.transition_prob:  # Bewegung mit Wahrscheinlichkeit 0.9
            if action == 'left':
                if current_state == self.room_indices['D']:
                    next_state = self.room_indices['B']
                elif current_state == self.room_indices['F']:
                    next_state = self.room_indices['D']
                elif current_state == self.room_indices['C']:
                    next_state = self.room_indices['A']
                elif current_state == self.room_indices['E']:
                    next_state = self.room_indices['C']
                elif current_state == self.room_indices['G']:
                    next_state = self.room_indices['E']
                else:
                    next_state = current_state
            elif action == 'right':
                if current_state == self.room_indices['B']:
                    next_state = self.room_indices['D']
                elif current_state == self.room_indices['D']:
                    next_state = self.room_indices['F']
                elif current_state == self.room_indices['A']:
                    next_state = self.room_indices['C']
                elif current_state == self.room_indices['C']:
                    next_state = self.room_indices['E']
                elif current_state == self.room_indices['E']:
                    next_state = self.room_indices['G']
                else:
                    next_state = current_state
            elif action == 'up':
                if current_state == self.room_indices['A']:
                    next_state = self.room_indices['B']
                elif current_state == self.room_indices['C']:
                    next_state = self.room_indices['D']
                elif current_state == self.room_indices['E']:
                    next_state = self.room_indices['F']
                else:
                    next_state = current_state
            elif action == 'down':
                if current_state == self.room_indices['B']:
                    next_state = self.room_indices['A']
                elif current_state == self.room_indices['D']:
                    next_state = self.room_indices['C']
                elif current_state == self.room_indices['F']:
                    next_state = self.room_indices['E']
                else:
                    next_state = current_state
            else:
                next_state = current_state

        else:  # Verbleiben im aktuellen Raum mit Wahrscheinlichkeit 0.1
            next_state = current_state

        reward = self.reward_step
        return next_state, reward

    # Q-Learning Algorithmus
    def q_learning(self, episodes, alpha, epsilon):
        rewards_per_episode = []
        for _ in range(episodes):
            current_state = random.choice(range(self.num_rooms - 1))  # Startzustand zufällig wählen
            total_reward = 0
            while current_state != self.room_indices['G']:
                action = self.choose_action(current_state, epsilon)
                action_index = self.action_indices[action]
                next_state, reward = self.get_next_state_and_reward(current_state, action)
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state, best_next_action]
                td_error = td_target - self.Q[current_state, action_index]
                print(td_error)
                self.Q[current_state, action_index] += alpha * td_error
                current_state = next_state
                total_reward += reward
            rewards_per_episode.append(total_reward)
        return rewards_per_episode

def main():
    parser = argparse.ArgumentParser(
        description='Trainiere einen Agenten mit Q-Learning in einer simulierten Umgebung.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Diskontierungsfaktor für zukünftige Belohnungen.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Lernrate für das Q-Learning.')
    parser.add_argument('--epsilon', type=float, default=0.9, help='Epsilon-Wert für die ε-greedy Politik.')
    parser.add_argument('--episodes', type=int, default=1000, help='Anzahl der Episoden für das Q-Learning.')
    parser.add_argument('--rooms', type=str, default=['A', 'B', 'C', 'D', 'E', 'F', 'G'], help='Räume.')
    parser.add_argument('--actions', type=str, default=['left', 'right', 'up', 'down'], help='Anzahl der Aktionen.')
    parser.add_argument('--transition_prob', type=float, default=0.9, help='Übergangswahrscheinlichkeit.')
    parser.add_argument('--stay_prob', type=float, default=0.1, help='Aktion schlägt fehl.')
    parser.add_argument('--reward_step', type=int, default=-1, help='Kosten.')

    args = parser.parse_args()

    rooms = args.rooms
    actions = args.actions
    transition_prob = args.transition_prob
    stay_prob = args.stay_prob
    reward_step = args.reward_step
    gamma = args.gamma

    env = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    episodes = args.episodes
    alpha = args.alpha
    epsilon = args.epsilon

    rewards_per_episode = env.q_learning(episodes, alpha, epsilon)

    # Ergebnisse anzeigen
    print("Q-Tabelle nach Q-Learning:")
    print(env.Q)

    # Optimale Politik anzeigen
    optimal_policy = {env.rooms[i]: env.actions[np.argmax(env.Q[i])] for i in range(env.num_rooms)}

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

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()

    # Neue Raumpositionen als Dictionary (alle auf einer Achse)
    positions = {
        'A': (0, 1), 'B': (0, 0), 'C': (1, 1), 'D': (1, 0), 'E': (2, 1), 'F': (2, 0), 'G': (3, 1)
    }

    # Zeichne Rechtecke und Aktionen
    for room, pos in positions.items():
        # Zeichne das Rechteck für den Raum
        rect = plt.Rectangle((pos[0] - 0.5, pos[1] - 0.25), 1, 0.5, edgecolor='black', facecolor='lightgrey', lw=2)
        ax.add_patch(rect)

        # Hole die beste Aktion für den Raum
        action = optimal_policy[room]

        # Zeichne den Raumname in die Mitte der linken Hälfte des Rechtecks
        ax.text(pos[0] - 0.25, pos[1], f"{room}",
                ha='center', va='center', fontsize=12, weight='bold')

        # Zeichne den Pfeil in die Mitte der rechten Hälfte des Rechtecks
        arrow_length = 0.15
        arrow_offset = 0.25  # Offset für den Pfeil, damit er in der rechten Hälfte liegt

        if action == 'left':
            ax.arrow(pos[0] + arrow_offset, pos[1], -arrow_length, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
        elif action == 'right':
            ax.arrow(pos[0] + arrow_offset, pos[1], arrow_length, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
        elif action == 'up':
            ax.arrow(pos[0] + arrow_offset, pos[1], 0, -arrow_length, head_width=0.05, head_length=0.05, fc='k', ec='k')
        elif action == 'down':
            ax.arrow(pos[0] + arrow_offset, pos[1], 0, arrow_length, head_width=0.05, head_length=0.05, fc='k', ec='k')

    # Setze die Achsen-Ticks und -Labels aus
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title('Optimale Politik')
    plt.show()


    plot_policy(optimal_policy)

if __name__ == "__main__":
    main()
