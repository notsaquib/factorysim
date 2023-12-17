import argparse
import json
import logging
import os
from utils.configs import load_configs, set_logging_parameters
from factorygym.factoryGymA import FsimEnv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
import time
import cProfile
import matplotlib.pyplot as plt
import matplotlib
from models import QNetwork, QNetworkLSTM

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    #profiler.disable()
    #profiler.print_stats(sort='tottime')
    #except Exception as e:
     #   print(e)
     #   continue


# Define the epsilon-greedy policy
def epsilon_greedy(q_network, state, epsilon, n_actions=1):
    np.random.seed(int(time.time_ns() % 1000000))
    x = np.random.rand(1)

    if x < epsilon:
        # Random action
        # print(int(time.time_ns()%1000000))
        y = np.random.choice(n_actions + 1)
        # print(y)
        return y
    else:
        # Greedy action
        q_values = q_network(state).detach().cpu().numpy()
        return np.argmax(q_values)

def train(cfgs: dict, args: argparse.Namespace):
    env = FsimEnv(5, 1, './product/products.xml', './factory/machine/machines.xml', './energy/day_ahead.csv')
    # profiler = cProfile.Profile()
    # profiler.enable()
    # Define the environment

    # Define the number of agents and the action space
    n_agents = len(env.machines_number)
    n_actions = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the Q-network for each agent
    if cfgs["general"]["model"] == 0:
        q_network = [QNetwork().to(device) for _ in range(n_agents)]
    elif cfgs["general"]["model"] == 1:
        q_network = [QNetworkLSTM().to(device) for _ in range(n_agents)]
    optimizer = [optim.Adam(q_network[i].parameters(), lr=0.01) for i in range(n_agents)]
    loss_fn = torch.nn.MSELoss()

    # Define the hyperparameters
    gamma = cfgs["hyperparam"]["gamma"]
    epsilon = cfgs["hyperparam"]["epsilon"]
    min_epsilon = cfgs["hyperparam"]["min_epsilon"]
    epsilon_decay = cfgs["hyperparam"]["epsilon_decay"]
    episodes = cfgs["hyperparam"]["episodes"]
    batch_size = cfgs["hyperparam"]["batch_size"]

    # Define the replay buffer
    replay_buffer = []
    record_replay_buffer = []


    prices = []
    state = env.reset()
    # Train the agents using Q-learning with neural networks
    for episode in range(episodes):
        try:
            state, _, _, record = env.reset()
            state = list(state.items())
            for i in range(len(state)):
                state[i] = list(state[i][1].values())
                state[i] = np.array(
                    [item for sublist in state[i] for item in (sublist if isinstance(sublist, list) else [sublist])])
            record = [state]
            record = np.array(record)
            done = False

            while not done:

                # Choose an action for each agent using epsilon-greedy policy
                # actions = [epsilon_greedy(q_network[i], torch.tensor(state[i]).float().to(device), epsilon) for i in
                #           range(n_agents)]
                actions = [epsilon_greedy(q_network[i], torch.tensor(record[:, i]).float().to(device), epsilon) for i in
                           range(n_agents)]
                # print(actions)
                # Take the actions and observe the next state and rewards

                next_state, rewards, done, next_record = env.step(actions)

                next_state = list(next_state.items())
                for i in range(len(next_state)):
                    next_state[i] =list(next_state[i][1].values())
                    next_state[i] = np.array(
                        [item for sublist in next_state[i] for item in
                         (sublist if isinstance(sublist, list) else [sublist])])
                record_list = []
                for record_item in next_record:
                    record_state = list(record_item.items())
                    for i in range(len(record_state)):
                        record_state[i] = list(record_state[i][1].values())
                        record_state[i] = np.array(
                            [item for sublist in record_state[i] for item in
                             (sublist if isinstance(sublist, list) else [sublist])])
                    record_list.append(record_state)

                # Add the transition to the replay buffer
                replay_buffer.append((state, actions, rewards, next_state, done))
                record_replay_buffer.append((record, actions, rewards, record_list, done))
                # Sample a minibatch from the replay buffer
                minibatch = np.random.choice(len(replay_buffer), size=batch_size, replace=True)
                minibatch = np.random.choice(len(record_replay_buffer), size=batch_size, replace=True)

                states, actions, rewards, next_states, dones = zip(*[record_replay_buffer[i] for i in minibatch])

                max_len1 = max(len(l) for l in next_states)
                # max_len2 = max(len(l) for l in states)
                next_states = [l + [[np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)]] * (max_len1 - len(l))
                               for l in next_states]
                # states = [l + [np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)] * (max_len1 - len(l)) for l
                #               in states]

                # Convert the minibatch to NumPy arrays
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)

                # Update the Q-network for each agent
                for i in range(n_agents):
                    # Compute the target Q-values
                    with torch.no_grad():
                        target_q_values = rewards[:, i] + gamma * np.max(
                            q_network[i](torch.tensor(next_states[:, :, i, :]).float().to(device)).cpu().numpy(),
                            axis=1) * (
                                                  1 - dones)

                        # Compute the current Q-values
                        current_q_values = q_network[i](torch.tensor(states[:, :, i, :]).float().to(device)).cpu().numpy()
                        ground_q_values = current_q_values
                        ground_q_values[np.arange(batch_size), actions[:, i]] = target_q_values

                    # Train the Q-network using the minibatch
                    optimizer[i].zero_grad()
                    loss = loss_fn(torch.tensor(current_q_values.squeeze(), requires_grad=True).float().to(device),
                                   torch.tensor(ground_q_values.squeeze()).float().to(device))
                    loss.backward()
                    optimizer[i].step()
                # Update the state
                state = next_state
                epsilon *= epsilon_decay
                epsilon = max(epsilon, min_epsilon)
                replay_buffer = replay_buffer[-1024 * 16:]
            print(f"Energy price based schedule in episode {episode}: " + str(env.factory_sim.energy_price))
            prices.append(env.factory_sim.energy_price)
            plot(prices)
        except:
            pass

def plot(prices):
    plt.plot(list(range(len(prices))), prices)
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.savefig('prices.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logging",
        help="choose logging level",
        choices=["debug", "info", "release"],
        default="info",
    )
    parser.add_argument("--configs", help="config.json file")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="runs some samples for profiling",
    )

    args = parser.parse_args()

    configs = load_configs(args.configs)
    set_logging_parameters(configs, args.logging)

    train(configs, args)

