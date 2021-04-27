from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch


class Memory:
    """Memory object which stores past data for experiential learning."""

    def __init__(self):
        """Initializes arrays for storing experience variables."""

        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        """Reinitializes experience variables."""

        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]


class ExperienceDataset(Dataset):
    """Object to create data batches for training."""

    def __init__(self, old_states, old_actions, old_log_probs, rewards):
        """Initializes experience variables for training."""

        self.old_states = old_states
        self.old_actions = old_actions
        self.old_log_probs = old_log_probs
        self.rewards = rewards

    def __getitem__(self, ix):
        """Extracts experience for learning."""

        return (self.old_states[ix], self.old_actions[ix],
                self.old_log_probs[ix], self.rewards[ix])

    def __len__(self):
        """Returns length of experiences."""

        return len(self.old_states)


class PPOAgent:
    """
    A class to implement an Agent which interacts with and learns from the
    environment!

    Attributes:
        device: Designated processor for executing program.
        actor_critic: PolicyNormal object containing current Actor/Critic
            networks.
        actor_critic_old: PolicyNormal object containing old Actor/Critic
            networks.
        gamma: A float designating the discount factor.
        num_updates: Integer number of updates desired for every
            update_frequency steps.
        eps_clip: Float designating range for clipping surrogate objective.
        critic_loss: Float designating initial Critic loss.
        entropy_bonus: Float increasing Actor's tendency for exploration.
        batch_size: An integer for minibatch size used for training.
        actor_optimizer: Optimizer used when training Actor network.
        critic_optimizer: Optimizer used when training Critic network.
    """

    def __init__(self, device, actor_critic, actor_critic_old, gamma,
                 num_updates, eps_clip, critic_loss, entropy_bonus,
                 batch_size, actor_optimizer, critic_optimizer):
        """Initializes relevant variables for Agent object."""

        self.device = device
        self.actor_critic = actor_critic
        self.actor_critic_old = actor_critic_old
        self.gamma = gamma
        self.num_updates = num_updates
        self.eps_clip = eps_clip
        self.critic_loss = critic_loss
        self.entropy_bonus = entropy_bonus
        self.batch_size = batch_size
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.memory = Memory()

    def add_memory(self, state, action, log_prob, reward, done):
        """Stores experience variables to memory."""

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.log_probs.append(log_prob)
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)

    def get_actions(self, state):
        """Samples actions from current policy for the given state."""

        # Generates actions and log probs from current Normal distribution.
        self.actor_critic_old.eval()
        with torch.no_grad():
            actions, log_probs = \
                self.actor_critic_old.act(state.to(self.device))
        self.actor_critic_old.train()
        actions = actions.cpu().detach()
        log_probs = log_probs.cpu().detach()

        return actions, log_probs

    def discount_rewards(self, rewards, dones):
        """
        Computes cumulative rewards after each step of each episode.

        Parameters:
            rewards: Array with stored rewards from episode.
            dones: Array with stored done boolean values.
        """

        # Initialize current return and array for reward storage.
        current_return = 0
        returns = []

        # Compute returns for each timestep in reverse order.
        for reward, done in zip(rewards[::-1], dones[::-1]):

            # Restart return calculation if episode is completed.
            if done:
                current_return = 0

            # Calculate return based on discount factor and store value.
            current_return = reward + (self.gamma * current_return)
            returns.append(current_return)

        return returns[::-1]

    def update_batch(self, old_states, old_actions, old_log_probs, rewards):
        """Execute gradient steps for Actor/Critic networks."""

        # Compute new state, log_prob estimates based on old states, actions.
        log_probs, state_values, dist_entropy = \
            self.actor_critic.evaluate(old_states, old_actions)

        # Compute probability ratio between old and new policies.
        ratios = torch.exp(log_probs - old_log_probs.detach())

        # Compute advantages and consequent surrogate losses.
        advantages = (rewards - state_values.detach()).clone()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                            1 + self.eps_clip) * advantages

        # Compute loss variables for gradient step.
        l1 = -torch.min(surr1, surr2)
        l2 = self.critic_loss * (state_values - rewards) ** 2
        l3 = -self.entropy_bonus * dist_entropy

        # Compute Actor loss with entropy bonus and Critic loss.
        loss1 = (l1 + l3).mean()
        loss2 = l2.mean()

        # Execute gradient step.
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def update(self):
        """Carries out updates on Actor/Critic Networks utilizing Memory."""

        # Calculate and normalize Monte Carlo estimates of state rewards.
        rewards = self.discount_rewards(self.memory.rewards, self.memory.dones)
        rewards = torch.tensor(rewards).to(self.device).unsqueeze(1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list of tensor variables to tensors.
        old_states = torch.stack(self.memory.states).to(self.device)
        old_actions = torch.stack(self.memory.actions).to(self.device)
        old_log_probs = torch.stack(self.memory.log_probs).to(self.device)

        # Initialize ExperienceDataset and DataLoader objects for training.
        dataset = ExperienceDataset(old_states, old_actions,
                                    old_log_probs, rewards)
        data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 shuffle=True)

        # Update Actor/Critic networks num_updates number of times.
        for _ in range(self.num_updates):
            for old_states_batch, old_actions_batch, old_log_probs_batch, \
                    rewards_batch in data_loader:
                self.update_batch(
                    old_states=old_states_batch,
                    old_actions=old_actions_batch,
                    old_log_probs=old_log_probs_batch,
                    rewards=rewards_batch
                )

        # Update old Actor/Critic networks to match current ones.
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())

        # Reset Memory information.
        self.memory = Memory()
        self.memory.clear_memory()
