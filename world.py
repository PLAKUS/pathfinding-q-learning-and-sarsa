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
    #sarsa_env = SarsaEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    #q_env_rand = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    q_env = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)
    sarsa_env = SarsaEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma)

    max_iterations = args.max_iterations

    # Trainiere den Agenten
    rewards_per_episode_q = q_env.q_learning(max_iterations, False)
    rewards_per_episode_sarsa = sarsa_env.sarsa(max_iterations)


    # Erzeugen einer Visuals-Instanz
    visuals_q = Visuals(len(rewards_per_episode_q))
    visuals_sarsa = Visuals(len(rewards_per_episode_sarsa))

    visuals_q.calc_expected_cost_q(q_env)

    # Aufrufen der Visualisierungsmethoden
    print("Projektvorschlag 10:")

    #Aufgabe (a):
    visuals_q.print_a(q_env, rewards_per_episode_q)

    print('---------------------------------------------------')

    #Aufgabe (b):
    visuals_q.print_b(q_env, rewards_per_episode_q)

    print('---------------------------------------------------')

    #Aufgabe (c):
    visuals_q.print_c(q_env, rewards_per_episode_q)

    print('---------------------------------------------------')

    #Aufgabe (d):
    visuals_q.print_d(q_env, rewards_per_episode_q)

    print('---------------------------------------------------')

    #Aufgabe (e):
    # Agent mit diskontierung 0.1
    q_env_01 = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma=0.1)
    rewards_per_episode_q01 = q_env_01.q_learning(max_iterations, False)
    # Agent mit diskontierung 0.5
    q_env_05 = QLearningEnvironment(rooms, actions, transition_prob, stay_prob, reward_step, gamma=0.5) 
    rewards_per_episode_q05 = q_env_05.q_learning(max_iterations, False)

    visuals_q.print_e(q_env, q_env_01, q_env_05, rewards_per_episode_q, rewards_per_episode_q01, rewards_per_episode_q05)

    print('---------------------------------------------------')

    #Aufgabe (f):
    visuals_sarsa.print_f(sarsa_env, rewards_per_episode_sarsa)

    print('---------------------------------------------------')
