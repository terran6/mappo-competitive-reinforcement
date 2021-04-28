from torch.distributions import Normal
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch


def hidden_init(layer):
    """
    Provides limits for Uniform distribution which reinitializes parameters
    for neural network layers.

    Arguments:
        layer: Neural network layer to be reinitialized.

    Returns:
        limits: Upper and lower limits used for Uniform distribution.
    """

    # Calculate limits for Uniform distribution sampling.
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNet(nn.Module):
    """
    Initializes an Actor (Policy) Model.

    Arguments:
        state_size: Integer number of possible states.
        action_size: Integer number of possible actions.
        fc1_units: An integer number of nodes in first hidden layer.
        fc2_units: An integer number of nodes in second hidden layer.
    """

    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        """Initializes parameters and builds model."""

        # Initialize inheritance and relevant variables.
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.mu = nn.Linear(fc2_units, action_size)
        self.sigma = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def forward(self, state):
        """
        Builds an actor (policy) network that evaluates the mean and sigma for
        an action distribution based on the given state.

        Parameters:
            state: An instance of state gathered for the environent.

        Returns:
            mu: Mean for Normal distribution used for action selection.
            sigma: Sigma for Normal distribution used for action selection.
        """

        # Build base Actor neural network architecture.
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))

        # Evaluate the mean for the action distribution.
        mu = torch.tanh(self.mu(x))

        # Evaluate the sigma for the action distribution.
        log_sigma = -torch.relu(self.sigma(x))
        sigma = torch.exp(log_sigma)

        return mu, sigma

    def reset_parameters(self):
        """Reinitializes the parameters of each hidden layer."""

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-3e-3, 3e-3)
        self.sigma.weight.data.uniform_(-3e-3, 3e-3)


class CriticNet(nn.Module):
    """
    Initializes a Critic (Value) Model.

    Arguments:
        state_size: An integer count of dimensions for each state.
        fc1_units: An integer number of nodes in first hidden layer.
        fc2_units: An integer number of nodes in second hidden layer.
        state: An instance of state gathered for the environent.
    """

    def __init__(self, state_size, fc1_units, fc2_units):
        """Initializes parameters and builds model."""

        # Initialize inheritance and relevant variables.
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def forward(self, state):
        """
        Builds a critic (value) network evaluates the given state.

        Parameters:
            state: An instance of state gathered for the environent.

        Returns:
            out: Layer evaluating the quality of selected state.
        """

        # Build base Critic neural network architecture.
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # Output critique of state for Actor training.
        out = self.fc3(x)

        return out

    def reset_parameters(self):
        """Reinitializes the parameters of each hidden layer."""

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class PolicyNormal(nn.Module):
    """
    A class which generates, and evaluates the effectiveness of, a Normal
    distribution for action selection through its interaction with Actor
    and Critic networks.

    Arguments:
        actor: Actor network to generate Normal distribution.
        critic: Critic network to evaluate Normal distribution.
    """

    def __init__(self, actor, critic):
        """Specifies Actor and Critic networks for model evaluation."""

        # Initialize inheritance and relevant variables.
        super().__init__()
        self.actor = actor
        self.critic = critic

    def get_dist(self, mu, sigma):
        """
        Initializes a Normal distribution based on given mu and sigma values.

        Parameters:
            mu: Mean for Normal distribution used for action selection.
            sigma: Sigma for Normal distribution used for action selection.
        """

        # Generate Normal distribution based on mu and sigma values.
        sigma = sigma.expand_as(mu)
        dist = Normal(mu, sigma)

        return dist

    def act(self, state):
        """
        Samples action value from given state for environment interaction.

        Parameters:
            state: An instance of state gathered for the environent.

        Returns:
            action: Normal distribution sampled value used for action.
            log_prob: Log probability of sampled action.
        """

        # Generate Normal distribution based on current state.
        action_mu, action_sigma = self.actor(state)
        dist = self.get_dist(action_mu, action_sigma)

        # Sample action value from generated distribution.
        action = dist.sample()

        # Calculate the log probability of the sampled action.
        log_prob = dist.log_prob(action).sum(-1).reshape(-1)

        return action, log_prob

    def evaluate(self, state, action):
        """
        Compute the log probability of the sampled action and the value of the
        given state.

        Parameters:
            state: An instance of state gathered for the environent.
            action: Normal distribution sampled value used for action.

        Returns:
            log_prob: Log probability of sampled action.
            state_value: Quality of given state based on value.
            dist_entropy: Calculated entropy used for environment exploration.
        """

        # Generate Normal distribution based on current state.
        action_mu, action_sigma = self.actor(state)
        dist = self.get_dist(action_mu, action_sigma)

        # Calculate the log probability of the sampled action.
        log_prob = dist.log_prob(action).sum(1, keepdim=True)

        # Compute entropy used in loss function, dictates exploration.
        dist_entropy = dist.entropy().sum(1, keepdim=True)

        # Calculate the quality of state value based on value.
        state_value = self.critic(state)

        return log_prob, state_value, dist_entropy
