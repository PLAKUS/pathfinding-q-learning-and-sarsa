import numpy as np
from matplotlib import pyplot as plt


class Visuals:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

    # Ergebnisse anzeigen
    def print_q_table(self, env):
        print("Q-Tabelle nach Q-Learning:")
        print(env.Q)

    # Optimale Politik anzeigen
    def print_optimal_policy(self, env):
        optimal_policy = {env.rooms[i]: env.actions[np.argmax(env.Q[i])] for i in range(env.num_rooms)}
        print("Optimale Politik:")
        print(optimal_policy)

    # Lernkurve anzeigen
    def plot_learning_curve(self, env):
        plt.figure(figsize=(10, 6))
        plt.plot(env.sarsa(self.max_iterations))  # hier muss man den Algorithmus noch mit if abfragen
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
            optimal_policy = {env.rooms[i]: env.actions[np.argmax(env.Q[i])] for i in range(env.num_rooms)}
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