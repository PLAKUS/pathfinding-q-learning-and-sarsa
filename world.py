import argparse
from visualization import Visuals

from sarsa import SarsaEnvironment
from qlearning import QLearningEnvironment



if __name__ == "__main__":
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

    # Sarsa oder Q-Lernen
    sarsa_env = SarsaEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    q_env = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)

    max_iterations = args.max_iterations

    rewards_per_episode0 = sarsa_env.sarsa(max_iterations)
    rewards_per_episode1 = q_env.q_learning(max_iterations)

    # Erzeugen einer Visuals-Instanz
    visuals = Visuals(max_iterations)

    # Aufrufen der Visualisierungsmethoden
    visuals.print_sarsa(sarsa_env, rewards_per_episode0)

    print('---------------------------------------------------')

    visuals.print_sarsa(q_env, rewards_per_episode1)

