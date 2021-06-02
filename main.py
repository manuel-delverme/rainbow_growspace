import random

import growspace  # noqa
import gym
import numpy as np
import torch
import tqdm

import agent
import config
import env_wrapper
import memory
import testing


def main():
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Environment
    env = gym.make(config.env_name)
    env.seed(config.seed)
    env = env_wrapper.WrapPyTorch(env, stack_size=config.stack_size)

    # Agent
    dqn_agent = agent.Agent(config, env)
    replay_memory = memory.ReplayMemory(config, config.memory_capacity)
    priority_weight_increase = (1 - config.priority_weight) / (config.T_max - config.learn_start)

    validation_replay_memory = memory.ReplayMemory(config, config.evaluation_size)
    state, done = env.reset(), False
    state = torch.FloatTensor(state).to(config.device)

    for env_steps in range(config.evaluation_size):
        next_state, _, done, _ = env.step(env.action_space.sample())
        next_state = torch.FloatTensor(next_state).to(config.device)
        validation_replay_memory.append(state, None, None, done)
        state = next_state

        if done:
            state, done = env.reset(), False
            state = torch.FloatTensor(state).to(config.device)

    # Training loop
    dqn_agent.train()
    env_steps, done = 0, True

    for env_steps in tqdm.trange(config.T_max):
        if done:
            state, done = env.reset(), False
            state = torch.FloatTensor(state).to(config.device)

        if env_steps % config.replay_frequency == 0:
            dqn_agent.reset_noise()  # Draw a new set of noisy weights
        action = dqn_agent.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, _ = env.step(action)  # Step
        next_state = torch.FloatTensor(next_state).to(config.device)

        if config.reward_clip > 0:
            reward = np.clip(reward, -config.reward_clip, config.reward_clip)

        replay_memory.append(state, action, reward, done)

        if env_steps >= config.learn_start:
            replay_memory.priority_weight = min(replay_memory.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            if env_steps % config.replay_frequency == 0:
                dqn_agent.learn(replay_memory)  # Train with n-step distributional double-Q learning

            if env_steps % config.evaluation_interval == 0:
                assert env_steps // int(1e5) == 0

                dqn_agent.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = testing.test(config, env_steps, dqn_agent, validation_replay_memory)  # Test
                config.tensorboard.add_scalar("train/avg_reward", avg_reward, env_steps)
                config.tensorboard.add_scalar("train/avg_reward", avg_Q, env_steps)

                if env_steps % int(1e6) == 0:
                    config.tensorboard.add_scalar("test/1e6_mean_reward", avg_reward, env_steps)
                dqn_agent.train()  # Set DQN (online network) back to training mode

            # Update target network
            if env_steps % config.target_update == 0:
                dqn_agent.update_target_net()
        state = next_state
    env.close()


if __name__ == "__main__":
    main()
