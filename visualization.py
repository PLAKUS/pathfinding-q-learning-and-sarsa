import numpy as np
from matplotlib import pyplot as plt


class Visuals:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

    # Ergebnisse anzeigen
    def print_q(self, env, rewards_per_episode):
        # Ergebnisse anzeigen
        print("Q-Tabelle nach Q-Lernen:")
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
        plt.title('Aufgabe (c): Lernkurve des Q-Lernen-Algorithmus')
        plt.grid(True)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(1.5, 2.75)
        ax.invert_yaxis()

        # Raumpositionen als Dictionary
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

    # Optimale Politik anzeigen
    def print_sarsa(self, env, rewards_per_episode):

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
        plt.title('Aufgabe (f): Lernkurve des Sarsa-Algorithmus')
        plt.grid(True)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(1.5, 2.75)
        ax.invert_yaxis()

        # Raumpositionen als Dictionary
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
