import numpy as np
import random


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
        dynamic_epsilon = 1 / (episode/100)  # +1 um Division durch 0 zu vermeiden
        if random.uniform(0, 1) < dynamic_epsilon:
            return random.choice(self.actions)
        else:
            transitions = {
                'A': {'left': 'A', 'right': 'C', 'up': 'B', 'down': 'A'},
                'B': {'left': 'B', 'right': 'D', 'up': 'B', 'down': 'A'},
                'C': {'left': 'A', 'right': 'E', 'up': 'D', 'down': 'C'},
                'D': {'left': 'B', 'right': 'F', 'up': 'D', 'down': 'C'},
                'E': {'left': 'C', 'right': 'G', 'up': 'E', 'down': 'E'},
                'F': {'left': 'D', 'right': 'F', 'up': 'F', 'down': 'F'},
                'G': {'left': 'G', 'right': 'G', 'up': 'G', 'down': 'G'}
            }
            q_values = self.Q[state]
            max_value = np.max(q_values)
            best_actions = [action for action, q in zip(self.actions, q_values) if q == max_value]
            # Bei gleichem Q-Wert den Raum mit dem niedrigsten Index wählen
            next_state = len(self.rooms)
            if len(best_actions)>1:
                for action in best_actions:
                    new_room = self.room_indices[transitions[self.rooms[state]][action]]
                    if next_state > new_room:
                        next_state = new_room
                        best_action = action
                return best_action
            return min(best_actions)

    # Simulationsfunktion für die Umgebung basierend auf dem neuen Grundriss
    def get_next_state_and_reward(self, current_state, action):
        # Definiere die Transition basierend auf dem jetzigen Zustand und Aktion
        transitions = {
            'A': {'left': 'A', 'right': 'C', 'up': 'B', 'down': 'A'},
            'B': {'left': 'B', 'right': 'B', 'up': 'B', 'down': 'A'},
            'C': {'left': 'A', 'right': 'E', 'up': 'D', 'down': 'C'},
            'D': {'left': 'B', 'right': 'F', 'up': 'D', 'down': 'C'},
            'E': {'left': 'C', 'right': 'G', 'up': 'F', 'down': 'E'},
            'F': {'left': 'D', 'right': 'F', 'up': 'F', 'down': 'F'},
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
    def sarsa(self, max_iterations, convergence_threshold=0.0001, min_episodes=1):
    def sarsa(self, num_iterations):
        rewards_per_episode = []
        iteration = 0
        converged = False
        prev_Q = np.copy(self.Q)
        while not converged:
            iteration += 1
            current_state = random.choice(range(self.num_rooms - 1))
            action = self.choose_action(current_state, x)
            total_reward = 0

            while current_state != self.room_indices['G']:
                action_index = self.action_indices[action]
                next_state, reward = self.get_next_state_and_reward(current_state, action)
                next_action = self.choose_action(next_state, x)
                next_action_index = self.action_indices[next_action]

                # Q-Wert Berechnung
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

        return rewards_per_episode

    # Funktion, um Erwartungswert für die Kosten des kürzesten Pfades von start_state nach goal_state
    def simulate_path_costs(self, start_state, goal_state, num_simulations=1000):
        costs = []

        for _ in range(num_simulations):
            current_state = start_state
            total_cost = 0
            while current_state != goal_state:
                action_index = np.argmax(self.Q[current_state])  # Best action
                action = self.actions[action_index]
                next_state, reward = self.get_next_state_and_reward(current_state, action)
                total_cost += reward
                current_state = next_state

                # Sicherheitsabfrage, um eine Endlosschleife zu vermeiden
                if total_cost < -1000:  # anpassen, um zu lange Pfade zu vermeiden
                    break

            costs.append(total_cost)

        expected_cost = np.mean(costs)
        return expected_cost, costs
