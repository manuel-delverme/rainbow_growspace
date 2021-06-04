import random

import growspace  # noqa
import gym
import gym.spaces
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
    done = 0

    episode_rewards = []
    episode_branches = []
    # episode_branch1 = []
    # episode_branch2 = []
    episode_light_width = []
    episode_light_move = []
    episode_success = []
    episode_plantpixel = []

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dist = np.zeros(env.action_space.n)

    state, done = env.reset(), False
    state = torch.FloatTensor(state).to(config.device)

    for env_steps in tqdm.trange(config.T_max):
        if env_steps % config.replay_frequency == 0:
            dqn_agent.reset_noise()  # Draw a new set of noisy weights

        action = dqn_agent.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, info = env.step(action)  # Step
        next_state = torch.FloatTensor(next_state).to(config.device)

        if config.reward_clip > 0:
            reward = np.clip(reward, -config.reward_clip, config.reward_clip)

        replay_memory.append(state, action, reward, done)
        episode_rewards.append(reward)

        if 'episode' in info.keys():
            episode_rewards.append(info['episode']['r'])
            episode_length.append(info['episode']['l'])
            # wandb.log({"Episode_Reward": info['episode']['r']}, step=total_num_steps)

        if 'new_branches' in info.keys():
            episode_branches.append(info['new_branches'])

        if 'new_b1' in info.keys():
            episode_branch1.append(info['new_b1'])

        if 'new_b2' in info.keys():
            episode_branch2.append(info['new_b2'])

        if 'light_width' in info.keys():
            episode_light_width.append(info['light_width'])

        if 'light_move' in info.keys():
            episode_light_move.append(info['light_move'])

        if 'success' in info.keys():
            episode_success.append(info['success'])

        if 'plant_pixel' in info.keys():
            episode_plantpixel.append(info['plant_pixel'])

        if env_steps >= config.learn_start:
            replay_memory.priority_weight = min(replay_memory.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            if env_steps % config.replay_frequency == 0:
                dqn_agent.learn(replay_memory)  # Train with n-step distributional double-Q learning

            if env_steps % config.evaluation_interval == 0:
                assert env_steps // int(1e5) == 0

                dqn_agent.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = testing.test(config, env_steps, dqn_agent, validation_replay_memory)  # Test
                config.tensorboard.add_scalar("train/sum_reward", avg_reward, env_steps)
                config.tensorboard.add_scalar("train/sum_reward", avg_Q, env_steps)

                if env_steps % int(1e6) == 0:
                    config.tensorboard.add_scalar("test/1e6_sum_reward", avg_reward, env_steps)
                dqn_agent.train()  # Set DQN (online network) back to training mode

            # Update target network
            if env_steps % config.target_update == 0:
                dqn_agent.update_target_net()

        state = next_state

        if done:
            if isinstance(env.action_space, gym.spaces.Discrete):
                config.tensorboard.add_histogram("Discrete_Actions", action_dist, env_steps)

            config.tensorboard.add_scalar("Reward_Min", np.min(episode_rewards), env_steps)
            config.tensorboard.add_scalar("Summed_Reward", np.sum(episode_rewards), env_steps)
            config.tensorboard.add_scalar("Reward_Mean", np.mean(episode_rewards), env_steps)
            config.tensorboard.add_scalar("Reward_Max", np.max(episode_rewards), env_steps)
            config.tensorboard.add_scalar("Number_of_Mean_New_Branches", np.mean(episode_branches), env_steps)
            config.tensorboard.add_scalar("Number_of_Max_New_Branches", np.max(episode_branches), env_steps)
            config.tensorboard.add_scalar("Number_of_Min_New_Branches", np.min(episode_branches), env_steps)
            # config.tensorboard.add_scalar("Number_of_Mean_New_Branches_of_Plant_1", np.mean(episode_branch1), env_steps)
            # config.tensorboard.add_scalar("Number_of_Mean_New_Branches_of_Plant_2", np.mean(episode_branch2), env_steps)
            config.tensorboard.add_scalar("Number_of_Total_Displacement_of_Light", np.sum(episode_light_move), env_steps)
            config.tensorboard.add_scalar("Mean_Light_Displacement", np.mean(episode_light_move), env_steps)
            config.tensorboard.add_scalar("Mean_Light_Width", np.mean(episode_light_width), env_steps)
            config.tensorboard.add_scalar("Number_of_Steps_in_Episode_with_Tree_is_as_close_as_possible", np.sum(episode_success), env_steps)
            config.tensorboard.add_scalar("Mean_Plant_Pixel", np.mean(episode_plantpixel), env_steps)
            config.tensorboard.add_scalar("Summed_Plant_Pixel", np.sum(episode_plantpixel), env_steps)

            # config.tensorboard.add_histogram("Displacement of Light Position", env_steps)
            # config.tensorboard.add_histogram("Displacement of Beam Width", env_steps)
            # config.tensorboard.add_histogram("Plant Pixel Histogram", env_steps)

            episode_rewards.clear()
            episode_branches.clear()
            # episode_branch2.clear()
            # episode_branch1.clear()
            episode_light_move.clear()
            episode_light_width.clear()
            episode_success.clear()
            episode_plantpixel.clear()

            state, done = env.reset(), False
            state = torch.FloatTensor(state).to(config.device)

    env.close()


if __name__ == "__main__":
    main()
