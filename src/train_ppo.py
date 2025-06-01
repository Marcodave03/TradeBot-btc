import pandas as pd
import numpy as np
import torch
import time
import logging
import os
import yaml
import json

from data.processor import DataProcessor
from environment.trading_env import TradingEnv 
from agent.ppo_agent import PPOAgent 

def evaluate_agent(env: TradingEnv, agent: PPOAgent, num_episodes=10, deterministic_eval=True):
    total_rewards = []
    total_net_worths = []
    original_mode_is_train = agent.actor_critic.training
    agent.actor_critic.eval() 

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False; truncated = False; episode_reward = 0
        with torch.no_grad():
            while not (done or truncated):
                action, _, _ = agent.select_action(obs, deterministic=deterministic_eval) 
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                obs = next_obs; episode_reward += reward
        total_rewards.append(episode_reward)
        total_net_worths.append(info.get('net_worth', env.initial_balance))
        logging.debug(f"Eval Ep {episode+1}: Reward={episode_reward:.3f}, Final NW={info.get('net_worth', env.initial_balance):.2f}")
    
    if original_mode_is_train:
        agent.actor_critic.train()
        
    return np.mean(total_rewards), np.mean(total_net_worths)

def main_training_loop(config_path="config/config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None: 
            logging.error(f"Config file {config_path} is empty or invalid. Cannot proceed.")
            return
    except FileNotFoundError:
        logging.error(f"FATAL: Config file {config_path} not found. Creating a default example. Please review and run again.")
        _default_raw_dir = os.path.join("data", "historical_raw")
        _default_processed_dir = os.path.join("data", "historical_processed")
        os.makedirs(_default_raw_dir, exist_ok=True)
        os.makedirs(_default_processed_dir, exist_ok=True)
        os.makedirs(os.path.join("src", "models", "saved_models", "PPO"), exist_ok=True)

        default_config_content = {
            "project_root": ".", "logging_level": "INFO",
            "data_paths": {
                "raw_data_directory": "data/historical_raw",
                "processed_data_directory": "data/historical_processed",
                "features_list_path": "data/historical_processed/final_agent_features.json",
                "scaler_params_path": "data/historical_processed/custom_fe_scaler_params.joblib",
                "scaler_colnames_path": "data/historical_processed/custom_fe_scaler_colnames.joblib",
                "model_save_directory": "src/models/saved_models/PPO" 
            },
            "data_processing": {
                "feature_engineer_timeframes": ["1h", "15m", "5m"],
                "primary_timeframe": "15m",
                "train_split_ratio": 0.7, "validation_split_ratio": 0.15
            },
            "feature_engineering": { 
                "settings": {}, 
                "indicators": [ # --- Comprehensive list of indicators ---
                    {"name": "SMA", "params": {"length": 10}, "on_column": "Close"},
                    {"name": "SMA", "params": {"length": 30}, "on_column": "Close"},
                    {"name": "EMA", "params": {"length": 10}, "on_column": "Close"},
                    {"name": "EMA", "params": {"length": 30}, "on_column": "Close"},
                    {"name": "RSI", "params": {"length": 14}, "on_column": "Close"},
                    {"name": "ATR", "params": {"length": 14}}, # Assumes HLC columns are present
                    {"name": "BBANDS", "params": {"length": 20, "std": 2}, "on_column": "Close"}, # Produces BBP, BBW
                    {"name": "SMA", "params": {"length": 20}, "on_column": "Volume", "output_name_override": "Vol_SMA_20"},
                    {"name": "PCT_CHANGE", "params": {"periods": 1}, "on_column": "Volume", "output_name_override": "Vol_Change_1"},
                    {"name": "ROC", "params": {"length": 1}, "on_column": "Close"},
                    {"name": "ROC", "params": {"length": 5}, "on_column": "Close"}
                ]
            },
            "environment": {
                "initial_balance": 10000., "lookback_window": 30, "commission_fee": .001,
                "episode_max_steps": 1000, "holding_penalty_ratio": .000001,
                "trade_penalty_ratio": .00001, "bankruptcy_penalty": -1.,
                "stop_loss_pct": .05, "short_sell_margin_floor_pct": .1,
                "position_size_fractions": [.125, .25],
                "slippage_factor_per_1k_value": .00001,
                "volatility_penalty_coeff": .01, "atr_period_for_regime": 14,
                "regime_atr_threshold_pct": .015
            },
            "agent": {
                "actor_critic": {"hidden_size1": 512, "hidden_size2": 256},
                "ppo": {
                    "lr": .0003, "ppo_batch_size": 64, "ppo_epochs": 10, "gamma": .99,
                    "gae_lambda": .95, "policy_clip": .2, "entropy_coeff": .02,
                    "value_loss_coeff": .5, "max_grad_norm": 0.5
                }
            },
            "training_loop": {
                "max_total_timesteps": 2000000, "num_episodes_fallback": 10000,
                "rollout_steps": 2048, "save_model_freq_episodes": 50,
                "log_freq_episodes": 5, "eval_freq_episodes": 20, "eval_episodes": 10,
                "early_stopping_patience": 20
            },
            "device": "auto"
        }
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir): os.makedirs(config_dir, exist_ok=True)
        with open(config_path, "w") as f: yaml.dump(default_config_content, f, sort_keys=False, default_flow_style=False)
        return # Exit so user can review/populate the new default config
    except yaml.YAMLError as e: logging.error(f"FATAL: Error parsing YAML {config_path}: {e}"); return
    except Exception as e: logging.error(f"FATAL: Could not load or process config {config_path}: {e}"); return


    log_level_str = config.get("logging_level", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__) 

    data_paths_cfg = config.get('data_paths', {})
    dp_cfg = config.get('data_processing', {})
    env_cfg = config.get('environment', {})
    agent_cfg = config.get('agent', {})
    loop_cfg = config.get('training_loop', {})

    project_root_str = config.get("project_root", ".")
    project_root = os.path.abspath(project_root_str)

    def resolve_path(key, default_subpath, cfg_section):
        path_val = cfg_section.get(key)
        if path_val:
            return os.path.join(project_root, path_val) if not os.path.isabs(path_val) else path_val

        processed_dir = os.path.join(project_root, data_paths_cfg.get("processed_data_directory", "data/historical_processed"))
        return os.path.join(processed_dir, default_subpath)

    model_save_dir = resolve_path("model_save_directory", os.path.join("src", "models", "saved_models", "PPO"), loop_cfg) # Prefer loop_cfg then data_paths_cfg
    if not os.path.exists(model_save_dir) : os.makedirs(model_save_dir, exist_ok=True)
    
    features_list_path = resolve_path("features_list_path", "final_agent_features.json", data_paths_cfg)
    if os.path.dirname(features_list_path) and not os.path.exists(os.path.dirname(features_list_path)):
         os.makedirs(os.path.dirname(features_list_path), exist_ok=True)


    # 1 -> data processing
    logger.info(f"Initializing DataProcessor with main config: {config_path}")
    processor = DataProcessor(config_path=config_path) 
    
    
    df_train_scaled, df_val_scaled, df_test_scaled, final_agent_features_list = \
        processor.process_all_timeframes(config_path=config_path) 
    
    primary_timeframe_str_for_filename = dp_cfg.get("primary_timeframe", "15m") # Get from data_processing config

    datasets_to_save = [
        (df_train_scaled, "train_processed", f"train"),
        (df_val_scaled, "val_processed", f"val"),
        (df_test_scaled, "test_processed", f"test")
    ]
    for df_to_save, name_prefix, log_info in datasets_to_save:
        filename = f"df_{name_prefix}_{primary_timeframe_str_for_filename}.csv"
        processor.save_processed_data(df_to_save, filename, log_info)
    
    loaded_agent_features_list = final_agent_features_list 
    if not final_agent_features_list and os.path.exists(features_list_path):
        try:
            with open(features_list_path, 'r') as f: loaded_agent_features_list = json.load(f)
            logger.info(f"Loaded agent features from {features_list_path}")
        except Exception as e: logger.error(f"TRAIN: Failed to load agent features from {features_list_path}: {e}")

    if df_train_scaled.empty or not loaded_agent_features_list:
        logger.error("TRAIN: Training data empty or no features list. Exiting."); return
    if df_val_scaled.empty: logger.warning("TRAIN: Validation data is empty. Evaluation will be skipped.")

    primary_timeframe_str = dp_cfg.get("primary_timeframe", "15m")
    
    price_col_to_use = f"Close_{primary_timeframe_str}" 
    if price_col_to_use not in df_train_scaled.columns:
        if "Close" in df_train_scaled.columns:
            price_col_to_use = "Close"
            logger.warning(f"Suffixed price column '{f'Close_{primary_timeframe_str}'}' not found. Using generic 'Close'. Verify compatibility with TradingEnv.")
        else:
            logger.error(f"TRAIN: Price column for primary_tf '{primary_timeframe_str}' (expected: '{price_col_to_use}' or 'Close') not in df_train_scaled. Cols: {df_train_scaled.columns.tolist()[:10]}. Exiting.")
            return

    train_env_params = {
        "features_to_use": loaded_agent_features_list,
        "price_column": price_col_to_use,
        "primary_timeframe_str": primary_timeframe_str, 
        **env_cfg
    }
    logger.info(f"Initializing Training Environment with params: {json.dumps({k: str(v) if isinstance(v, list) else v for k,v in train_env_params.items() if k!='features_to_use'}, indent=2)}")
    train_env = TradingEnv(df=df_train_scaled, serial=False, **train_env_params)
    
    val_env = None
    if not df_val_scaled.empty:
        logger.info("Initializing Validation Environment...")
        val_env = TradingEnv(df=df_val_scaled, serial=True, **train_env_params) 

    input_dims = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n 
    logger.info(f"TRAIN: Env obs_shape: {input_dims}, action_space: {n_actions}")

    device = config.get("device", "auto")
    if device == "auto": device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(device)
    
    agent = PPOAgent(input_dims=input_dims, n_actions=n_actions, 
                     config_path=config_path, device=device)

    max_total_timesteps = int(loop_cfg.get("max_total_timesteps", 2000000))
    num_episodes_fallback = int(loop_cfg.get("num_episodes_fallback", 10000))
    rollout_steps = int(loop_cfg.get("rollout_steps", 2048))
    save_model_freq_episodes = int(loop_cfg.get("save_model_freq_episodes", 50))
    log_freq_episodes = int(loop_cfg.get("log_freq_episodes", 5))
    eval_freq_episodes = int(loop_cfg.get("eval_freq_episodes", 20))
    eval_episodes = int(loop_cfg.get("eval_episodes", 10))
    max_patience = int(loop_cfg.get("early_stopping_patience", 20))

    total_timesteps_collected = 0; episode_count = 0; all_episode_rewards = []
    best_val_reward = -float('inf'); patience_counter = 0
    
    logger.info(f"TRAIN: Starting PPO. Max steps: {max_total_timesteps}, Max eps: {num_episodes_fallback}, Device: {device}")
    logger.info(f"Saving models to: {model_save_dir}")
    start_time_total_training = time.time()
    action_counts_rollout = np.zeros(n_actions, dtype=int)

    while total_timesteps_collected < max_total_timesteps and episode_count < num_episodes_fallback:
        obs, _ = train_env.reset()
        done_episode = False; current_episode_reward = 0; current_episode_steps = 0
        
        last_value_for_bootstrap = torch.tensor(0.0, device=agent.device)
        
        for _rollout_step_idx in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs, deterministic=False)
            action_counts_rollout[action] += 1
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done_step = terminated or truncated
            agent.store_transition(obs, action, log_prob, reward, done_step, value)
            obs = next_obs; current_episode_reward += reward; total_timesteps_collected += 1; current_episode_steps +=1
            if done_step: done_episode = True; break
            if total_timesteps_collected >= max_total_timesteps: break
        
        if not done_episode and len(agent.states) > 0: 
            with torch.no_grad():
                agent.actor_critic.eval()
                obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
                _, _, _, last_val_tensor = agent.actor_critic.get_action_and_value(obs_np, device=agent.device, deterministic=True)
                last_value_for_bootstrap = last_val_tensor.squeeze() if last_val_tensor.numel() > 0 else torch.tensor(0.0, device=agent.device)
                agent.actor_critic.train()
        elif done_episode:
             last_value_for_bootstrap = torch.tensor(0.0, device=agent.device)
        
        actor_loss, critic_loss, entropy_val = 0.0, 0.0, 0.0
        if len(agent.states) > 0: 
            actor_loss, critic_loss, entropy_val = agent.learn(last_value_for_bootstrap, done_episode)
        
        if done_episode: 
            all_episode_rewards.append(current_episode_reward)
            episode_count += 1
            if episode_count % log_freq_episodes == 0:
                avg_reward = np.mean(all_episode_rewards[-log_freq_episodes:]) if all_episode_rewards else 0.0
                action_dist_percent = (action_counts_rollout / np.sum(action_counts_rollout) * 100) if np.sum(action_counts_rollout) > 0 else np.zeros(n_actions)
                action_labels = ["Hold"]
                if hasattr(train_env, 'position_size_fractions') and train_env.position_size_fractions:
                    for frac in train_env.position_size_fractions: action_labels.append(f"Buy_{int(frac*100)}%")
                    for frac in train_env.position_size_fractions: action_labels.append(f"Sell_{int(frac*100)}%")
                else: action_labels.extend([f"Act_{i}" for i in range(1,n_actions)])
                log_action_dist_str = ", ".join([f"{action_labels[i]}: {action_dist_percent[i]:.1f}%" for i in range(n_actions) if i < len(action_labels)]) if len(action_labels) == n_actions else "Action label/count mismatch"
                
                logger.info(f"Ep: {episode_count}, Steps: {total_timesteps_collected}, AvgRew(last {log_freq_episodes}): {avg_reward:.3f}")
                raw_entropy = entropy_val / (-agent.entropy_coeff) if agent.entropy_coeff != 0 else 0.0
                info_nw = info.get('net_worth', train_env.initial_balance) if info else train_env.initial_balance
                info_trades = info.get('trades', 0) if info else 0
                logger.info(f"  LastEp: Rew: {current_episode_reward:.3f}, EpSteps: {current_episode_steps}, ALoss: {actor_loss:.4f}, CLoss: {critic_loss:.4f}, EntTerm: {entropy_val:.4f}, RawEnt: {raw_entropy:.4f}")
                logger.info(f"  LastEpInfo: NW: {info_nw:.2f}, Trades: {info_trades}")
                logger.info(f"  Action Dist: {log_action_dist_str}")
                action_counts_rollout = np.zeros(n_actions, dtype=int)

            if val_env and episode_count > 0 and episode_count % eval_freq_episodes == 0:
                logger.info(f"--- Validating (Ep: {episode_count}, Steps: {total_timesteps_collected}) ---")
                avg_val_rew, avg_val_nw = evaluate_agent(val_env, agent, num_episodes=eval_episodes, deterministic_eval=True)
                logger.info(f"Validation: AvgReward: {avg_val_rew:.3f}, AvgNW: {avg_val_nw:.2f}")
                if avg_val_rew > best_val_reward:
                    best_val_reward = avg_val_rew; patience_counter = 0
                    logger.info(f"*** New best validation reward: {best_val_reward:.3f}. Saving model. ***")
                    agent.save_model(os.path.join(model_save_dir, "ppo_trading_agent_best_val.pth"))
                else:
                    patience_counter += 1; logger.info(f"Validation reward did not improve. Patience: {patience_counter}/{max_patience}")
                if patience_counter >= max_patience: logger.info(f"EARLY STOPPING due to no validation improvement."); break 
            
            if episode_count > 0 and episode_count % save_model_freq_episodes == 0:
                agent.save_model(os.path.join(model_save_dir, f"ppo_trading_agent_ep{episode_count}.pth"))
        
        if total_timesteps_collected >= max_total_timesteps: logger.info("Max total timesteps reached."); break
            
    training_duration = time.time() - start_time_total_training
    logger.info(f"TRAIN: Finished. Episodes: {episode_count}, Timesteps: {total_timesteps_collected}. Duration: {training_duration:.2f}s")
    agent.save_model(os.path.join(model_save_dir, "ppo_trading_agent_final.pth"))
    
    train_env.close(); 
    if val_env: val_env.close()

if __name__ == "__main__":
    main_training_loop()