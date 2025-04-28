import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)  # Actor output layer
        self.critic = nn.Linear(hidden_size, 1)  # Critic output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)  
        state_value = self.critic(x)  # State value from the Critic
        return action_probs, state_value
