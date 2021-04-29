from mappo.mappo_trainer import MAPPOTrainer
from unityagents import UnityEnvironment
from mappo.ppo_model import PolicyNormal
from mappo.ppo_model import CriticNet
from mappo.ppo_model import ActorNet
from mappo.ppo_agent import PPOAgent
import numpy as np
import torch
import sys
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_env(env_loc):
    """
    Initializes the UnityEnviornment and corresponding environment variables
    based on the running operating system.

    Arguments:
        env_loc: A string designating unity environment directory.

    Returns:
        env: A UnityEnvironment used for Agent evaluation and training.
        num_agents: Integer number of Agents to be trained in environment.
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
    """

    # Set path for unity environment based on operating system.
    if sys.platform == 'linux':
        f = os.path.join(env_loc, r'Tennis_Linux/Tennis.x86_64')
    elif sys.platform == 'darwin':
        f = os.path.join(env_loc, r'Tennis.app')
    else:
        f = os.path.join(env_loc, r'Tennis_Windows_x86_64/Tennis.exe')

    # Initialize unity environment, return message if error thrown.
    try:
        env = UnityEnvironment(file_name=f)

    except UnityEnvironmentException:
        print('\nEnvironment not found or Operating System not supported.\n')
        sys.exit()

    # Extract state dimensionality from env.
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = env_info.vector_observations.shape[-1]

    # Extract action dimensionality and number of agents from env.
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)

    # Display relevant environment information.
    print('\nNumber of Agents: {}, State Size: {}, Action Size: {}\n'.format(
        num_agents, state_size, action_size))

    return env, num_agents, state_size, action_size


def create_agent(state_size, action_size, actor_fc1_units=512,
                 actor_fc2_units=256, actor_lr=1e-4, critic_fc1_units=512,
                 critic_fc2_units=256, critic_lr=1e-4, gamma=0.99,
                 num_updates=10, max_eps_length=500, eps_clip=0.3,
                 critic_loss=0.5, entropy_bonus=0.01, batch_size=256):
    """
    This function creates an agent with specified parameters for training.

    Arguments:
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
        actor_fc1_units: An integer number of units used in the first FC
            layer for the Actor object.
        actor_fc2_units: An integer number of units used in the second FC
            layer for the Actor object.
        actor_lr: A float designating the learning rate of the Actor's
            optimizer.
        critic_fc1_units: An integer number of units used in the first FC
            layer for the Critic object.
        critic_fc2_units: An integer number of units used in the second FC
            layer for the Critic object.
        critic_lr: A float designating the learning rate of the Critic's
            optimizer.
        gamma: A float designating the discount factor.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        eps_clip: Float designating range for clipping surrogate objective.
        critic_loss: Float designating initial Critic loss.
        entropy_bonus: Float increasing Actor's tendency for exploration.
        batch_size: An integer for minibatch size.

    Returns:
        agent: An Agent object used for training.
    """

    # Create Actor/Critic networks based on designated parameters.
    actor_net = ActorNet(state_size, action_size, actor_fc1_units,
                         actor_fc2_units).to(device)
    critic_net = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)

    # Create copy of Actor/Critic networks for action prediction.
    actor_net_old = ActorNet(state_size, action_size, actor_fc1_units,
                             actor_fc2_units).to(device)
    critic_net_old = CriticNet(state_size, critic_fc1_units, critic_fc2_units)\
        .to(device)
    actor_net_old.load_state_dict(actor_net.state_dict())
    critic_net_old.load_state_dict(critic_net.state_dict())

    # Create PolicyNormal objects containing both sets of Actor/Critic nets.
    actor_critic = PolicyNormal(actor_net, critic_net)
    actor_critic_old = PolicyNormal(actor_net_old, critic_net_old)

    # Initialize optimizers for Actor and Critic networks.
    actor_optimizer = torch.optim.Adam(
        actor_net.parameters(),
        lr=actor_lr
    )
    critic_optimizer = torch.optim.Adam(
        critic_net.parameters(),
        lr=critic_lr
    )

    # Create and return PPOAgent with relevant parameters.
    agent = PPOAgent(
        device=device,
        actor_critic=actor_critic,
        actor_critic_old=actor_critic_old,
        gamma=gamma,
        num_updates=num_updates,
        eps_clip=eps_clip,
        critic_loss=critic_loss,
        entropy_bonus=entropy_bonus,
        batch_size=batch_size,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer
    )

    return agent


def create_trainer(env, agents, save_dir, update_frequency=5000,
                   max_eps_length=500, score_window_size=100):
    """
    Initializes trainer to train agents in specified environment.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        agents: Agent objects used for training.
        save_dir: Path designating directory to save resulting files.
        update_frequency: An integer designating the step frequency of
            updating target network parameters.
        max_eps_length: An integer for maximum number of timesteps per
            episode.
        score_window_size: Integer window size used in order to gather
            max mean score to evaluate environment solution.

    Returns:
        trainer: A MAPPOTrainer object used to train agents in environment.
    """

    # Initialize MAPPOTrainer object with relevant arguments.
    trainer = MAPPOTrainer(
        env=env,
        agents=agents,
        score_window_size=score_window_size,
        max_episode_length=max_eps_length,
        update_frequency=update_frequency,
        save_dir=save_dir
    )

    return trainer


def train_agents(env, trainer, n_episodes=8000, target_score=0.5,
                 score_window_size=100):
    """
    This function carries out the training process with specified trainer.

    Arguments:
        env: A UnityEnvironment used for Agent evaluation and training.
        trainer: A MAPPOTrainer object used to train agents in environment.
        n_episodes: An integer for maximum number of training episodes.
        target_score: An float max mean target score to be achieved over
            the last score_window_size episodes.
        score_window_size: The integer number of past episode scores
            utilized in order to calculate the current mean scores.
    """

    # Train the agent for n_episodes.
    for i_episode in range(1, n_episodes + 1):

        # Step through the training process.
        trainer.step()

        # Print status of training every 100 episodes.
        if i_episode % 100 == 0:
            scores = np.max(trainer.score_history, axis=1).tolist()
            trainer.print_status()

        # If target achieved, print and plot reward statistics.
        mean_reward = np.max(
            trainer.score_history[-score_window_size:], axis=1
        ).mean()
        if mean_reward >= target_score:
            print('Environment is solved.')
            env.close()
            trainer.print_status()
            trainer.plot()
            trainer.save()
            break


if __name__ == '__main__':

    # Initialize environment, extract state/action dimensions and num agents.
    env, num_agents, state_size, action_size = load_env(os.getcwd())

    # Initialize agents for training.
    agents = [create_agent(state_size, action_size) for _ in range(num_agents)]

    # Create MAPPOTrainer object to train agents.
    save_dir = os.path.join(os.getcwd(), r'saved_files')
    trainer = create_trainer(env, agents, save_dir)

    # Train agent in specified environment.
    train_agents(env, trainer)
