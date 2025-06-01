import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .models import ActorCritic
import logging
import yaml

class PPOAgent:
    def __init__(self, input_dims, n_actions, config_path, device: str = 'auto'):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        agent_config = full_config.get('agent', {})
        actor_critic_config = agent_config.get('actor_critic', {})
        ppo_config = agent_config.get('ppo', {})

        self.gamma = float(ppo_config.get('gamma', 0.99))
        self.gae_lambda = float(ppo_config.get('gae_lambda', 0.95))
        self.policy_clip = float(ppo_config.get('policy_clip', 0.2))
        self.ppo_batch_size = int(ppo_config.get('ppo_batch_size', 64))
        self.ppo_epochs = int(ppo_config.get('ppo_epochs', 10))
        self.entropy_coeff = float(ppo_config.get('entropy_coeff', 0.01))
        self.value_loss_coeff = float(ppo_config.get('value_loss_coeff', 0.5))
        self.max_grad_norm = float(ppo_config.get('max_grad_norm', 0.5))
        lr = float(ppo_config.get('lr', 3e-4))

        hidden_size1 = int(actor_critic_config.get('hidden_size1', 512))
        hidden_size2 = int(actor_critic_config.get('hidden_size2', 256))

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor_critic = ActorCritic(input_dims, n_actions, hidden_size1, hidden_size2).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self._clear_memory()

    def _clear_memory(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def store_transition(self, state_np, action_item, log_prob_tensor, reward_float, done_bool, value_tensor_scalar):
        self.states.append(state_np) 
        self.actions.append(action_item) 
        self.log_probs.append(log_prob_tensor.detach().cpu()) 
        self.rewards.append(reward_float)
        self.dones.append(done_bool)
        self.values.append(value_tensor_scalar.detach().cpu()) # value_tensor_scalar is V(s_t)

    def select_action(self, observation_np, deterministic=False): # Added deterministic parameter
        self.actor_critic.eval()
        with torch.no_grad():
            action_tensor, log_prob_tensor, _, value_tensor_scalar = \
                self.actor_critic.get_action_and_value(
                    observation_np, 
                    deterministic=deterministic, # Pass it down
                    device=self.device
                )
        # Ensure action_tensor is scalar before .item() if batch size is 1
        action_item = action_tensor.squeeze().item() if action_tensor.numel() == 1 else action_tensor.tolist()

        return action_item, log_prob_tensor, value_tensor_scalar
    
    def learn(self, last_value_for_bootstrap: torch.Tensor, last_done_for_bootstrap: bool):
        if not self.states:
            logging.warning("PPO learn called with no states in memory. Skipping update.")
            return 0.0, 0.0, 0.0  # Return zero losses
        
        self.actor_critic.train() # Set model to training mode

        # Convert stored experience to tensors
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(self.actions, dtype=torch.long).to(self.device) # Shape: (N,)
        # Ensure old_log_probs_t has shape (N,) if stored log_probs are scalar tensors
        old_log_probs_t = torch.stack(self.log_probs).to(self.device).squeeze() 
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32).to(self.device) # Shape: (N,)
        # dones_t[i] is True if state S_i led to a terminal state S_{i+1}
        dones_t = torch.tensor(self.dones, dtype=torch.bool).to(self.device)    # Shape: (N,) 
        values_t = torch.stack(self.values).to(self.device).squeeze() # V(S_t), Shape: (N,)

        # --- Calculate Generalized Advantage Estimation (GAE) ---
        advantages = torch.zeros_like(rewards_t)
        last_gae_lam = 0.0 # Initialize as float
        num_steps = len(rewards_t)

        # last_value_for_bootstrap is V(S_N) from the perspective of the (N-1)th transition.
        # It should already be 0.0 if last_done_for_bootstrap is True (handled in training loop).
        current_last_value = last_value_for_bootstrap.to(self.device).squeeze()

        for t in reversed(range(num_steps)):
            # Determine V(S_{t+1})
            # If t is the last step index (num_steps - 1), then S_{t+1} is S_N.
            # Its value is current_last_value (which is V(S_N), already 0 if S_N is terminal/truncated).
            # If t is not the last step, V(S_{t+1}) is values_t[t+1].
            # And if S_{t+1} was terminal (dones_t[t] is True), then V(S_{t+1}) for GAE should be 0.
            
            if t == num_steps - 1:
                next_value = current_last_value 
                # not_done_mask_for_next_state effectively checks if S_N was terminal/truncated
                not_done_mask_for_next_state = torch.tensor(0.0 if last_done_for_bootstrap else 1.0, device=self.device)
            else:
                next_value = values_t[t+1]
                # not_done_mask_for_next_state checks if S_{t+1} was terminal based on dones_t[t]
                not_done_mask_for_next_state = (~dones_t[t]).float() 
                
            current_value_St = values_t[t] # V(S_t)
            
            # Calculate delta (TD error)
            # delta_t = r_t + gamma * V(S_{t+1}) * (1 if S_{t+1} not done else 0) - V(S_t)
            delta = rewards_t[t] + self.gamma * next_value * not_done_mask_for_next_state - current_value_St
            
            # Calculate GAE: A_t = delta_t + gamma * lambda * A_{t+1} * (1 if S_{t+1} not done else 0)
            last_gae_lam = delta + self.gamma * self.gae_lambda * not_done_mask_for_next_state * last_gae_lam
            advantages[t] = last_gae_lam
        
        # --- Calculate Returns (targets for value function) ---
        # Returns_t = Advantage_t + V(S_t)
        returns_t = advantages + values_t 
        
        # Normalize advantages (common practice for PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss_epoch, total_critic_loss_epoch, total_entropy_loss_epoch = 0.0, 0.0, 0.0
        num_minibatches = 0

        # --- PPO Update Epochs ---
        for _ in range(self.ppo_epochs):
            # Create random minibatches from the collected trajectory
            indices = torch.randperm(num_steps).to(self.device)
            for start in range(0, num_steps, self.ppo_batch_size):
                end = start + self.ppo_batch_size
                batch_indices = indices[start:end]

                # Get new log_probs, entropy, and values for the states in the batch, using the *actions taken*
                # actions_t[batch_indices] provides the actions taken for states_t[batch_indices]
                # new_values_batch will be V_current_policy(S_t)
                # entropy_batch is the mean entropy from the policy distribution
                _, new_log_probs_batch, entropy_batch, new_values_batch = \
                    self.actor_critic.get_action_and_value(states_t[batch_indices], 
                                                        actions_t[batch_indices], 
                                                        self.device)

                # --- Actor (Policy) Loss ---
                # r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
                prob_ratio = torch.exp(new_log_probs_batch - old_log_probs_t[batch_indices]) 
                # Advantage for this batch
                batch_advantages = advantages[batch_indices]
                
                surr1 = prob_ratio * batch_advantages
                surr2 = torch.clamp(prob_ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic (Value) Loss ---
                # new_values_batch is V_theta_new(S_t) from the current network
                # returns_t[batch_indices] is the GAE-based target for V(S_t)
                critic_loss = F.mse_loss(new_values_batch, returns_t[batch_indices])
                
                # --- Entropy Loss (for exploration) ---
                # entropy_batch is already the mean entropy from get_action_and_value
                entropy_loss = -self.entropy_coeff * entropy_batch 

                # --- Total Loss ---
                total_loss = actor_loss + self.value_loss_coeff * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_actor_loss_epoch += actor_loss.item()
                total_critic_loss_epoch += critic_loss.item()
                total_entropy_loss_epoch += entropy_loss.item() # This is coeff * H
                num_minibatches +=1
        
        self._clear_memory() # Clear memory after learning from the batch
        
        avg_actor_loss = total_actor_loss_epoch / num_minibatches if num_minibatches > 0 else 0
        avg_critic_loss = total_critic_loss_epoch / num_minibatches if num_minibatches > 0 else 0
        # If you want to log pure average entropy (H) rather than entropy_coeff * H
        avg_pure_entropy = (total_entropy_loss_epoch / (-self.entropy_coeff if self.entropy_coeff != 0 else 1)) / num_minibatches if num_minibatches > 0 else 0
        
        # For consistency, let's return the scaled entropy loss as calculated
        avg_entropy_loss_scaled = total_entropy_loss_epoch / num_minibatches if num_minibatches > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_entropy_loss_scaled

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        logging.info(f"PPO model saved: {path}")

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval() 
        logging.info(f"PPO model loaded: {path}")