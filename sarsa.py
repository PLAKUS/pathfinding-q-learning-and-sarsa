import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

class SarsaEnvironment:
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
        self.update_counts = np.zeros((self.num_rooms, self.num_actions))  # Hinzufügen der Update-Count-Tabelle

    def choose_action(self, state, episode):
        dynamic_epsilon = 1 / (episode + 1)  # +1 um Division durch 0 zu vermeiden
        if random.uniform(0, 1) < dynamic_epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.Q[state]
            max_value = np.max(q_values)
            best_actions = [action for action, q in zip(self.actions, q_values) if q == max_value]
            return min(best_actions)

    def get_next_state_and_reward(self, current_state, action):
        # Definiere die Transition basierend auf dem jetzigen Zustand und Aktion
        transitions = {
            'A': {'left': 'A', 'right': 'C', 'up': 'B', 'down': 'A'},
            'B': {'left': 'B', 'right': 'B', 'up': 'B', 'down': 'A'},
            'C': {'left': 'A', 'right': 'E', 'up': 'D', 'down': 'C'},
            'D': {'left': 'B', 'right': 'F', 'up': 'D', 'down': 'C'},
            'E': {'left': 'C', 'right': 'G', 'up': 'F', 'down': 'E'},
            'F': {'left': 'D', 'right': 'F', 'up': 'F', 'down': 'E'},
            'G': {'left': 'G', 'right': 'G', 'up': 'G', 'down': 'G'}
        }
        # Bewegung mit Wahrscheinlichkeit 0.9
        if random.uniform(0, 1) < self.transition_prob:
            next_state = self.room_indices[transitions[self.rooms[current_state]][action]]
        else:
            next_state = current_state  # Verbleiben im aktuellen Raum mit Wahrscheinlichkeit 0.1

        reward = self.reward_step
        return next_state, reward

    # Implementierung des Sarsa-Algorithmus
    def sarsa(self, max_iterations, convergence_threshold=0.00000001):
        rewards_per_episode = []
        iteration = 0
        converged = False
        while not converged and iteration < max_iterations:
            iteration += 1
            current_state = random.choice(range(self.num_rooms - 1))
            action = self.choose_action(current_state, iteration)
            total_reward = 0

            while current_state != self.room_indices['G']:
                action_index = self.action_indices[action]
                next_state, reward = self.get_next_state_and_reward(current_state, action)
                next_action = self.choose_action(next_state, iteration)
                next_action_index = self.action_indices[next_action]

                # Q-Wert Berechnung
                predict = self.Q[current_state, action_index]
                target = reward + self.gamma * self.Q[next_state, next_action_index]

                # Anzahl der Aktualisierungen erhöhen
                self.update_counts[current_state, action_index] += 1
                alpha = 1 / self.update_counts[current_state, action_index]

                # Aktualisiere Q-Wert basierend auf der spezifizierten Lernregel
                self.Q[current_state, action_index] = (1 - alpha) * self.Q[
                    current_state, action_index] + alpha * target

                current_state = next_state
                action = next_action
                total_reward += reward

            rewards_per_episode.append(total_reward)

            # Check for convergence
            if iteration > 1:
                q_diff = np.mean(np.abs(self.Q - prev_Q))
                if q_diff < convergence_threshold:
                    converged = True
            prev_Q = np.copy(self.Q)

        return rewards_per_episode



def main():
    parser = argparse.ArgumentParser(description='Trainiere einen Agenten mit Sarsa in einer simulierten Umgebung.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Diskontierungsfaktor für zukünftige Belohnungen.')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Anzahl der Episoden für das Sarsa.')
    parser.add_argument('--rooms', type=str, default=['A', 'B', 'C', 'D', 'E', 'F', 'G'], help='Räume.')
    parser.add_argument('--actions', type=str, default=['left', 'right', 'up', 'down'], help='Aktionen.')
    parser.add_argument('--transition_prob', type=float, default=0.9, help='Übergangswahrscheinlichkeit.')
    parser.add_argument('--stay_prob', type=float, default=0.1, help='Wahrscheinlichkeit im aktuellen Raum zu bleiben.')
    parser.add_argument('--reward_step', type=int, default=-1, help='Kosten für eine Bewegung.')

    args = parser.parse_args()

    rooms = args.rooms
    actions = args.actions
    transition_prob = args.transition_prob
    stay_prob = args.stay_prob
    reward_step = args.reward_step
    gamma = args.gamma

    env = SarsaEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    max_iterations = args.max_iterations

    rewards_per_episode = env.sarsa(max_iterations)

    # Ergebnisse anzeigen
    print("Q-Tabelle nach Sarsa:")
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
    plt.title('Lernkurve des Sarsa-Algorithmus')
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

if __name__ == "__main__":
    main()
