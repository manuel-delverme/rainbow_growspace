import growspace  # noqa
import gym
import numpy as np
import torch

from env_wrapper import WrapPyTorch
import growspace  # noqa
import gym
import numpy as np
import torch

from env_wrapper import WrapPyTorch


# Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10
# test_iter = 0


# Test DQN
def test(config, time_step, agent, val_mem, evaluate=False):
    # global Ts, rewards, Qs, best_avg_reward, test_iter

    env = gym.make(config.env_name)
    env.seed(config.seed)
    env = WrapPyTorch(env)
    # action_space = env.action_space.n

    # Ts.append(T)
    # T_rewards, T_Qs = [], []
    Qs = []
    rewards = []

    # Test performance over several episodes
    state, reward_sum, done = env.reset(), 0, False
    state = torch.FloatTensor(state).to(config.device)
    for _ in range(config.evaluation_episodes):
        while not done:
            action = agent.act_e_greedy(state)  # Choose an action ε-greedily
            state, reward, done, info = env.step(action)
            state = torch.FloatTensor(state).to(config.device)
            reward_sum += reward
            # if args.render:
            #     env.render()
        rewards.append(reward_sum)
    env.close()

    # Test Q-values over validation memory
    for state in val_mem:  # Iterate over valid states
        Qs.append(agent.evaluate_q(state))

    avg_reward, avg_Q = np.mean(rewards), np.mean(Qs)
    return avg_reward, avg_Q
