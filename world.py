import argparse
from visualization import Visuals

from sarsa import SarsaEnvironment
from qlearning import QLearningEnvironment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trainiere einen Agenten mit Sarsa in einer simulierten Umgebung.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Diskontierungsfaktor für zukünftige Belohnungen.')
    parser.add_argument('--max_iterations', type=int, default=10000, help='Anzahl der Episoden für das Sarsa.')
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

    # Initialisiere Umgebungen
    sarsa_env = SarsaEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    q_env_rand = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    q_env = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)

    max_iterations = args.max_iterations

    # Trainiere den Agenten
    rewards_per_episode_sarsa = sarsa_env.sarsa(max_iterations)
    rewards_per_episode_qrandom = q_env_rand.q_learning(max_iterations, True)
    rewards_per_episode_q = q_env.q_learning(max_iterations, False)

    # Berechne die Kosten des kürzesten Pfades von A nach G
    start_state = sarsa_env.room_indices['A']
    goal_state = sarsa_env.room_indices['G']

    # Erzeugen einer Visuals-Instanz
    visuals = Visuals(max_iterations)

    # Aufrufen der Visualisierungsmethoden
    print("Projektvorschlag 10:")
    print("Aufgabe (a):")

    visuals.print_q(q_env, rewards_per_episode_q)

    expected_cost, costs_distribution = q_env.simulate_path_costs(start_state, goal_state)
    print(f"Aufgabe (b): Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von A nach G: {expected_cost}")

    print('---------------------------------------------------')

    print("Aufgabe (d):")

    visuals.print_q(q_env, rewards_per_episode_qrandom)
    expected_cost, costs_distribution = q_env.simulate_path_costs(start_state, goal_state)
    print(f"Aufgabe (b): Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von A nach G: {expected_cost}")

    print('---------------------------------------------------')

    print("Aufgabe (e): Diskontierungsfaktor auf 0.1")
    q_env = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma=0.1)
    rewards_per_episode_qrandom = q_env.q_learning(max_iterations, True)
    visuals.print_q(q_env, rewards_per_episode_qrandom)

    # expected_cost, costs_distribution = q_env.simulate_path_costs(start_state, goal_state)
    # print(f"Aufgabe (b): Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von A nach G: {expected_cost}")
    print('---------------------------------------------------')

    print("Aufgabe (e): Diskontierungsfaktor auf 0.5")
    q_env = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma=0.5)
    rewards_per_episode_qrandom = q_env.q_learning(max_iterations, True)
    visuals.print_q(q_env, rewards_per_episode_qrandom)

    #expected_cost, costs_distribution = q_env.simulate_path_costs(start_state, goal_state)
    #print(f"Aufgabe (b): Q-Learning - Erwartungswert für die Kosten der kürzesten Pfades von A nach G: {expected_cost}")

    print('---------------------------------------------------')

    print("Aufgabe (f):")
    visuals.print_sarsa(q_env, rewards_per_episode_sarsa)

    expected_cost, costs_distribution = sarsa_env.simulate_path_costs(start_state, goal_state)
    print(f"Aufgabe (b): Sarsa - Erwartungswert für die Kosten der kürzesten Pfades von A nach G: {expected_cost}")
