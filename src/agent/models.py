import torch
import torch.nn as nn
# import torch.nn.functional as F # Not used directly here, but often useful
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, h1, h2):
        super(ActorCritic, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(input_dims, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            # experiment w/ this
            # nn.Linear(h2, h2),
            # nn.ReLU()
        )
        self.actor_head = nn.Linear(h2, n_actions) # Outputs logits for actions
        self.critic_head = nn.Linear(h2, 1)      # Outputs a single state value

        self._initialize_weights() # Corrected typo: _initialize_weights

    def _initialize_weights(self): # Corrected typo
        for module in self.shared_net.modules():
            if isinstance(module, nn.Linear):
                # orthogonal initialization is good for policy gradients
                nn.init.orthogonal_(module.weight, gain = nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Specific initialization for actor and critic heads
        nn.init.orthogonal_(self.actor_head.weight, gain = .01) # Smaller gain for action logits
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain = 1.)  # Standard gain for value output
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, state_tensor):
        shared_features = self.shared_net(state_tensor)
        action_logits = self.actor_head(shared_features)
        state_values = self.critic_head(shared_features) # Shape: (batch_size, 1)
        return action_logits, state_values
    
    def get_action_and_value(self, state_input, action_token=None, deterministic=False, device='cpu'): # Added deterministic
        if isinstance(state_input, np.ndarray):
            state_tensor = torch.tensor(state_input, dtype=torch.float32).to(device)
            if state_tensor.ndim == 1: 
                state_tensor = state_tensor.unsqueeze(0)
        elif torch.is_tensor(state_input):
            state_tensor = state_input.to(device)
            if state_tensor.ndim == 1: 
                state_tensor = state_tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported state type: {type(state_input)}. Must be np.ndarray or torch.Tensor.")
        
        action_logits, state_value_batch = self.forward(state_tensor)
        prob_dist = Categorical(logits=action_logits)

        if action_token is None:
            if deterministic: # <-- Use deterministic flag
                actual_action_tensor = torch.argmax(action_logits, dim=-1)
            else:
                actual_action_tensor = prob_dist.sample()
        else:
            # ... (your existing logic for handling action_token) ...
            if not torch.is_tensor(action_token):
                action_token = torch.tensor(action_token, device=device, dtype=torch.long)
            if action_token.ndim == 0: action_token = action_token.unsqueeze(0)
            if action_token.ndim > 1 and action_token.shape[1] == 1: action_token = action_token.squeeze(-1)
            actual_action_tensor = action_token.to(device, dtype=torch.long)

        log_probs = prob_dist.log_prob(actual_action_tensor)
        entropy = prob_dist.entropy().mean() 
        final_state_value = state_value_batch.squeeze(-1)

        return actual_action_tensor, log_probs, entropy, final_state_value