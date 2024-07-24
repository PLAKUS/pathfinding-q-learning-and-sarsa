import numpy as np
from matplotlib import pyplot as plt

class Visuals:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.expected_costs = {}
        self.goal_state = 0

    def calc_expected_cost_q(self, env):
        self.goal_state = env.room_indices["G"]
        for i in range(env.num_rooms-1):
            expected_cost, costs_distribution = env.simulate_path_costs(i, self.goal_state)
            self.expected_costs[i] = expected_cost

# Aufgaben
###############################################################################################################
    # Aufgabe a ausgeben
    def print_a(self, env, rewards_per_episode):
        # Ergebnisse anzeigen
        print("\n")
        print("Aufgabe (A):")
        print("Trainiert mit Start in Raum: "+env.rooms[env.starting_room])
        print("Anzahl der durchlaufenen Trajektorien: "+format(len(rewards_per_episode)))
        print("Q-Tabelle nach Q-Lernen:")
        print(np.abs(np.array(env.Q)))

        # Optimale Politik anzeigen
        optimal_policy = {env.rooms[i]: env.actions[np.argmax(env.Q[i])] for i in range(env.num_rooms)}

        print("Optimale Politik:")
        print(optimal_policy)

        # Raumpositionen als Dictionary
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(1.5, 2.75)
        ax.invert_yaxis()
        positions = {
            'A': (0, 1.5), 'B': (0, 1), 'C': (1, 1.5), 'D': (1, 1), 'E': (2, 1.5), 'F': (2, 1), 'G': (3, 1.5)
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
                ax.arrow(pos[0] + arrow_offset, pos[1], -arrow_length, 0, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'right':
                ax.arrow(pos[0] + arrow_offset, pos[1], arrow_length, 0, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'up':
                ax.arrow(pos[0] + arrow_offset, pos[1], 0, -arrow_length, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'down':
                ax.arrow(pos[0] + arrow_offset, pos[1], 0, arrow_length, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')

        # Setze die Achsen-Ticks und -Labels aus
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.title('Optimale Politik')
        plt.show()
        print("\n")
###############################################################################################################

    # Aufgabe b ausgeben
    def print_b(self, env, rewards_per_episode):
        # Ergebnisse anzeigen
        print("\n")
        print("Aufgabe (B):")
        start_state = env.room_indices["A"]
        print(f"Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von {env.rooms[start_state]} nach {env.rooms[self.goal_state]}: {self.expected_costs[start_state]}")
        print("\n")
###############################################################################################################

    # Aufgabe c ausgeben
    def print_c(self, env, rewards_per_episode):
        # Ergebnisse anzeigen
        print("\n")
        print("Aufgabe (C):")
        print("Lernkurve siehe neues Fenster")
        # Lernkurve anzeigen
        plt.close('all')
        plt.figure(figsize=(10, 6))
        plot = np.array(rewards_per_episode)
        plot = np.abs(plot)
        plt.plot(plot)
        plt.xlabel('Episode')
        plt.ylabel('Gesamte Kosten')
        plt.title('Aufgabe (C): Lernkurve des Q-Lernen-Algorithmus A->G')
        plt.grid(True)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(1.5, 2.75)
        ax.invert_yaxis()
        print("\n")
###############################################################################################################

    # Aufgabe d ausgeben
    def  print_d(self, env, rewards_per_episode):
        print("\n")
        print("Aufgabe (D):")
        print("Trainiert mit Start in Raum: "+env.rooms[env.starting_room])
        print("Anzahl der durchlaufenen Episoden: "+format(len(rewards_per_episode)))
        print("Q-Tabelle nach Q-Lernen:")
        print(np.abs(np.array(env.Q)))

        # Optimale Politik anzeigen
        optimal_policy = {env.rooms[i]: env.actions[np.argmax(env.Q[i])] for i in range(env.num_rooms)}

        print("Optimale Politik:")
        print(optimal_policy)
        print(f"Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von {env.rooms[env.starting_room]} nach {env.rooms[self.goal_state]}: {self.expected_costs[env.starting_room]}")

        # Raumpositionen als Dictionary
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(1.5, 2.75)
        ax.invert_yaxis()
        positions = {
            'A': (0, 1.5), 'B': (0, 1), 'C': (1, 1.5), 'D': (1, 1), 'E': (2, 1.5), 'F': (2, 1), 'G': (3, 1.5)
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
                ax.arrow(pos[0] + arrow_offset, pos[1], -arrow_length, 0, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'right':
                ax.arrow(pos[0] + arrow_offset, pos[1], arrow_length, 0, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'up':
                ax.arrow(pos[0] + arrow_offset, pos[1], 0, -arrow_length, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'down':
                ax.arrow(pos[0] + arrow_offset, pos[1], 0, arrow_length, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')

        # Setze die Achsen-Ticks und -Labels aus
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.title('Optimale Politik')
        plt.show()
        # Lernkurve anzeigen
        plt.close('all')
        plt.figure(figsize=(10, 6))
        plot = np.array(rewards_per_episode)
        plot = np.abs(plot)
        plt.plot(plot)
        plt.xlabel('Episode')
        plt.ylabel('Gesamte Kosten')
        plt.title('Aufgabe (C): Lernkurve des Q-Lernen-Algorithmus F->G')
        plt.grid(True)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(1.5, 2.75)
        ax.invert_yaxis()
        print("\n")
###############################################################################################################

    # Aufgabe e ausgeben
    def print_e(self, env, env_01, env_05, rewards_per_episode, rewards_per_episode01, rewards_per_episode05):
        # Ergebnisse anzeigen
        print("\n")
        print("Aufgabe (E):")
        print(f"Anzahl der durchlaufenen Episoden ({env.gamma=}): "+format(len(rewards_per_episode)))
        optimal_policy = {env.rooms[i]: env.actions[np.argmax(env.Q[i])] for i in range(env.num_rooms)}
        print("Optimale Politik:")
        print(optimal_policy)
        print(f"Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von {env.rooms[env.starting_room]} nach {env.rooms[self.goal_state]}: {self.expected_costs[env.starting_room]}")
        print("\n")
        print(f"Anzahl der durchlaufenen Episoden ({env_01.gamma=}): "+format(len(rewards_per_episode01)))
        optimal_policy = {env_01.rooms[i]: env_01.actions[np.argmax(env_01.Q[i])] for i in range(env_01.num_rooms)}
        print("Optimale Politik:")
        print(optimal_policy)
        expected_cost01, costs_distribution = env_01.simulate_path_costs(env_01.starting_room, self.goal_state)
        print(f"Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von {env_01.rooms[env_01.starting_room]} nach {env_01.rooms[self.goal_state]}: {expected_cost01}")
        print("\n")
        print(f"Anzahl der durchlaufenen Episoden ({env_05.gamma=}): "+format(len(rewards_per_episode05)))
        optimal_policy = {env_05.rooms[i]: env_05.actions[np.argmax(env_05.Q[i])] for i in range(env_05.num_rooms)}
        print("Optimale Politik:")
        print(optimal_policy)
        expected_cost05, costs_distribution = env_05.simulate_path_costs(env_05.starting_room, self.goal_state)
        print(f"Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von {env_05.rooms[env_05.starting_room]} nach {env_05.rooms[self.goal_state]}: {expected_cost05}")

        data09 = np.abs(np.array(rewards_per_episode))
        data05 = np.abs(np.array(rewards_per_episode05))
        data01 = np.abs(np.array(rewards_per_episode01))
        # Lernkurve anzeigen
        # Compute moving average
        def compute_avg(data):
            window_size = 3  # Size of the moving window
            cumsum = np.cumsum(data)
            cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
            moving_avg = cumsum[window_size - 1:] / window_size
            x_values = np.arange(window_size - 1, len(data))
            return moving_avg, x_values
        avg09, x09 = compute_avg(data09)
        avg05, x05 = compute_avg(data05)
        avg01, x01 = compute_avg(data01)
        # X values (indices of the array, adjusted for moving average length)
        

        # Plotting
        plt.close('all')
        plt.figure(figsize=(10, 6))

        # Plot the moving average line
        #plt.plot(x09, avg09, label='Gamma=0.9', color='blue', linewidth=0.5)
        #plt.plot(x05, avg05, label='Gamma=0.5', color='red', linewidth=0.5)
        #plt.plot(x01, avg01, label='Gamma=0.1', color='black', linewidth=0.5)
        plt.plot(data09, label='Gamma=0.9', color='blue', linewidth=0.5)
        plt.plot(data05, label='Gamma=0.5', color='red', linewidth=0.5)
        plt.plot(data01, label='Gamma=0.1', color='black', linewidth=0.5)
        plt.plot(np.array([len(data01),len(data01)]),np.array([0,np.max(data01)]), color='black', linewidth=1, linestyle=":")
        plt.plot(np.array([len(data05),len(data05)]),np.array([0,np.max(data01)]), color='red', linewidth=1, linestyle=":")
        plt.plot(np.array([len(data09),len(data09)]),np.array([0,np.max(data01)]), color='blue', linewidth=1, linestyle=":")
        # Plot the original data
        #plt.plot(np.arange(len(data)), data, color='gray', linestyle='--', alpha=0.2, label='Original Data')
        plt.xlabel('Episode')
        plt.ylabel('Gesamte Kosten')
        plt.title('Aufgabe (e): Lernkurve des Q-Lernen-Algorithmus')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("\n")
###############################################################################################################

    # Aufgabe f ausgeben
    def print_f(self, env, rewards_per_episode, rewards_per_episode_q):
        # Ergebnisse anzeigen
        print("\n")
        print("Aufgabe (F):")
        print("Anzahl der durchlaufenen Episoden (Sarsa): "+format(len(rewards_per_episode)))
        print("Anzahl der durchlaufenen Episoden (Q-Lernen): "+format(len(rewards_per_episode_q)))
        print("Q-Tabelle nach Q-Lernen:")
        print(np.abs(np.array(env.Q)))
        expected_cost, costs_distribution = env.simulate_path_costs(env.room_indices["A"], env.room_indices["G"])
        print(f"Sarsa - Erwartungswert für die Kosten der kürzesten Pfades von {env.rooms[0]} nach G: {expected_cost}")

        print("Lernkurve siehe neues Fenster")
        # Lernkurve anzeigen
        plt.figure(figsize=(10, 6))
        data = np.abs(np.array(rewards_per_episode))
        data_q = np.abs(np.array(rewards_per_episode_q))
        plt.plot(data, label='Sarsa', color='red', linewidth=0.5)
        plt.plot(data_q, label='Q-Lernen', color='black', linewidth=0.5)
        plt.plot(np.array([len(data),len(data)]),np.array([0,np.max(data)]), color='red', linewidth=1, linestyle=":")
        plt.plot(np.array([len(data_q),len(data_q)]),np.array([0,np.max(data_q)]), color='black', linewidth=1, linestyle=":")
        plt.xlabel('Episode')
        plt.ylabel('Gesamte Kosten')
        plt.title('Aufgabe (F): Lernkurve des Sarsa-Algorithmus A->G')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Optimale Politik anzeigen
        optimal_policy = {env.rooms[i]: env.actions[np.argmax(env.Q[i])] for i in range(env.num_rooms)}

        print("Optimale Politik:")
        print(optimal_policy)

        # Raumpositionen als Dictionary
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(1.5, 2.75)
        ax.invert_yaxis()
        positions = {
            'A': (0, 1.5), 'B': (0, 1), 'C': (1, 1.5), 'D': (1, 1), 'E': (2, 1.5), 'F': (2, 1), 'G': (3, 1.5)
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
                ax.arrow(pos[0] + arrow_offset, pos[1], -arrow_length, 0, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'right':
                ax.arrow(pos[0] + arrow_offset, pos[1], arrow_length, 0, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'up':
                ax.arrow(pos[0] + arrow_offset, pos[1], 0, -arrow_length, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')
            elif action == 'down':
                ax.arrow(pos[0] + arrow_offset, pos[1], 0, arrow_length, head_width=0.05, head_length=0.05, fc='k',
                         ec='k')

        # Setze die Achsen-Ticks und -Labels aus
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.title('Optimale Politik')
        plt.show()
        print("\n")
