import random
from datetime import datetime
from test import test

import growspace  # noqa
import gym
import torch

import config
from agent import Agent
from env import WrapPyTorch
from memory import ReplayMemory

random.seed(config.seed)
torch.manual_seed(random.randint(1, 10000))


def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


# Environment
env = gym.make(config.env_name)
env.seed(config.seed)
env = WrapPyTorch(env, stack_size=4)
action_space = env.action_space.n

# Agent
dqn = Agent(config, env)
mem = ReplayMemory(config, config.memory_capacity)
priority_weight_increase = (1 - config.priority_weight) / (config.T_max - config.learn_start)

# Construct validation memory
val_mem = ReplayMemory(config, config.evaluation_size)
T, done = 0, True
while T < config.evaluation_size:
    if done:
        state, done = env.reset(), False
        state = torch.FloatTensor(state).to(config.device)
    next_state, _, done, _ = env.step(random.randint(0, action_space - 1))
    next_state = torch.FloatTensor(next_state).to(config.device)
    val_mem.append(state, None, None, done)
    state = next_state
    T += 1

if config.evaluate:
    dqn.eval()  # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test(config, 0, dqn, val_mem, evaluate=True)  # Test
    print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
    # Training loop
    dqn.train()
    T, done = 0, True
    while T < config.T_max:
        if done:
            state, done = env.reset(), False
            state = torch.FloatTensor(state).to(config.device)
        if T % config.replay_frequency == 0:
            dqn.reset_noise()  # Draw a new set of noisy weights
        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, _ = env.step(action)  # Step
        next_state = torch.FloatTensor(next_state).to(config.device)
        if config.reward_clip > 0:
            reward = max(min(reward, config.reward_clip), -config.reward_clip)  # Clip rewards
        mem.append(state, action, reward, done)  # Append transition to memory
        T += 1

        if T % config.log_interval == 0:
            log('T = ' + str(T) + ' / ' + str(config.T_max))

        # Train and test
        if T >= config.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            if T % config.replay_frequency == 0:
                dqn.learn(mem)  # Train with n-step distributional double-Q learning

            if T % config.evaluation_interval == 0:
                dqn.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(config, T, dqn, val_mem)  # Test
                log('T = ' + str(T) + ' / ' + str(config.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                dqn.train()  # Set DQN (online network) back to training mode

            # Update target network
            if T % config.target_update == 0:
                dqn.update_target_net()

        state = next_state

env.close()
