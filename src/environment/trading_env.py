import gym
from gym import spaces
import numpy as np
import pandas as pd
import logging
from collections import deque

class TradingEnv(gym.Env): # Using the name you provided
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, features_to_use: list, price_column: str,
                 primary_timeframe_str: str, 
                 # These will now come from a dictionary, but keep defaults for standalone use
                 initial_balance=10000, lookback_window=30, commission_fee=0.001,
                 serial=False, episode_max_steps=None,
                 holding_penalty_ratio=0.00001, trade_penalty_ratio=0.0001,
                 bankruptcy_penalty=-1.0, stop_loss_pct=0.05,
                 short_sell_margin_floor_pct=0.1, position_size_fractions=[0.125, 0.25],
                 slippage_factor_per_1k_value=0.00001, 
                 volatility_penalty_coeff=0.01,
                 atr_period_for_regime=14, regime_atr_threshold_pct=0.015,
                 **kwargs): # Absorb any other params from config if passed via **train_env_params
        
        super(TradingEnv, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing TradingEnv (Advanced Configurable) for primary TF: {primary_timeframe_str}...")

        self.initial_balance = float(kwargs.get('initial_balance', initial_balance))
        self.lookback_window = int(kwargs.get('lookback_window', lookback_window))
        self.commission_fee = float(kwargs.get('commission_fee', commission_fee))
        self.serial = kwargs.get('serial', serial)
        self.episode_max_steps = kwargs.get('episode_max_steps', episode_max_steps)
        
        self.holding_penalty_ratio = float(kwargs.get('holding_penalty_ratio', holding_penalty_ratio))
        self.trade_penalty_ratio = float(kwargs.get('trade_penalty_ratio', trade_penalty_ratio))
        self.bankruptcy_penalty = float(kwargs.get('bankruptcy_penalty', bankruptcy_penalty))
        self.volatility_penalty_coeff = float(kwargs.get('volatility_penalty_coeff', volatility_penalty_coeff))
        
        self.stop_loss_pct = float(kwargs.get('stop_loss_pct', stop_loss_pct))
        self.short_sell_margin_floor_pct = float(kwargs.get('short_sell_margin_floor_pct', short_sell_margin_floor_pct))
        self.short_sell_margin_floor = self.initial_balance * self.short_sell_margin_floor_pct
        
        _pos_fracs = kwargs.get('position_size_fractions', position_size_fractions)
        self.position_size_fractions = sorted(list(set(f for f in _pos_fracs if 0 < f <= 1.0)))
        if not self.position_size_fractions: self.position_size_fractions = [1.0]; self.logger.warning("Pos fractions invalid, defaulted to [1.0]")

        self.slippage_factor_per_1k_value = float(kwargs.get('slippage_factor_per_1k_value', slippage_factor_per_1k_value))
        self.slippage_value_unit = 1000.0 

        self.atr_period_for_regime = int(kwargs.get('atr_period_for_regime', atr_period_for_regime))
        self.regime_atr_threshold_pct = float(kwargs.get('regime_atr_threshold_pct', regime_atr_threshold_pct))

        if df.empty: raise ValueError("DataFrame for TradingEnv cannot be empty.")
        self.features_to_use = features_to_use 
        self.price_column = price_column 
        self.primary_timeframe_str = primary_timeframe_str
        if self.price_column not in df.columns: raise ValueError(f"Price column '{self.price_column}' not found.")
        
        missing_in_df = [f for f in features_to_use if f not in df.columns]
        if missing_in_df:
            self.logger.warning(f"Provided features_to_use not found in input df: {missing_in_df}. They will be ignored for observation construction.")
        
        self.df = df.copy().reset_index(drop=True)
        self.logger.debug(f"DataFrame shape: {self.df.shape}. Lookback: {self.lookback_window}")

        self.provided_atr_col_name = f'ATR_{self.atr_period_for_regime}_{self.primary_timeframe_str}'
        self.internal_atr_col_name = f'_internal_ATR_{self.atr_period_for_regime}'
        self.atr_col_to_use_for_regime = None 
        self.has_atr_for_regime = False
        if self.atr_period_for_regime > 0:
            if self.provided_atr_col_name in self.df.columns and self.provided_atr_col_name in self.features_to_use:
                self.atr_col_to_use_for_regime = self.provided_atr_col_name; self.has_atr_for_regime = True
                self.logger.info(f"Using provided (scaled) '{self.atr_col_to_use_for_regime}' for regime.")
            else:
                if self._calculate_atr_for_regime(): 
                    self.atr_col_to_use_for_regime = self.internal_atr_col_name; self.has_atr_for_regime = True
                else: self.logger.error(f"Failed to use/calculate ATR. Regime static.")
        
        self.provided_atr_col_name = f'ATR_{self.atr_period_for_regime}_{self.primary_timeframe_str}'
        self.internal_atr_col_name = f'_internal_ATR_{self.atr_period_for_regime}'
        self.atr_col_to_use_for_regime = None 
        self.has_atr_for_regime = False
        if self.atr_period_for_regime > 0:
            if self.provided_atr_col_name in self.df.columns and self.provided_atr_col_name in self.features_to_use:
                self.atr_col_to_use_for_regime = self.provided_atr_col_name; self.has_atr_for_regime = True
                self.logger.info(f"Using provided (scaled) '{self.atr_col_to_use_for_regime}' for regime.")
            else:
                if self._calculate_atr_for_regime(): 
                    self.atr_col_to_use_for_regime = self.internal_atr_col_name; self.has_atr_for_regime = True
                else: self.logger.error(f"Failed to use/calculate ATR. Regime static.")
        
        self.n_position_fractions = len(self.position_size_fractions)
        self.action_space = spaces.Discrete(1 + 2 * self.n_position_fractions)
        self.logger.info(f"Action space: Discrete({self.action_space.n}) using fractions: {self.position_size_fractions}")
        self.actual_features_in_df_for_obs = [f for f in self.features_to_use if f in self.df.columns]
        num_tech_features_shape = len(self.actual_features_in_df_for_obs)
        self.observation_space_shape = (4 + 1 + self.lookback_window * num_tech_features_shape,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_space_shape, dtype=np.float32)
        self.logger.info(f"Final Obs space shape: {self.observation_space_shape} ({num_tech_features_shape} tech features)")

        self.current_df_idx = 0; self.current_episode_step = 0
        self.recent_net_worth_changes_for_vol_penalty = deque(maxlen=self.lookback_window)
        self._reset_trading_state()

    def _calculate_atr_for_regime(self):
        base_price_col_name = self.price_column
        # Construct High, Low column names. Assumes consistent naming like "Close_15m" -> "High_15m"
        if base_price_col_name.startswith("Close_") and "_" in base_price_col_name:
            suffix = base_price_col_name.split("_",1)[1]
            high_col = f"High_{suffix}"
            low_col = f"Low_{suffix}"
        elif base_price_col_name == "Close": # Generic name, assume High/Low are also generic
            high_col = "High"
            low_col = "Low"
        else:
            self.logger.error(f"Cannot reliably derive High/Low column names from price_column '{self.price_column}' for ATR calculation.")
            return False

        if not all(col in self.df.columns for col in [high_col, low_col, self.price_column]):
            self.logger.error(f"Required raw columns ('{high_col}', '{low_col}', '{self.price_column}') not all found in DataFrame for internal ATR calculation.")
            return False
        try:
            # Ensure no inplace modifications on slices if df is a slice (though self.df is a copy)
            df_copy = self.df # Work on self.df directly as it's already a copy
            
            prev_close = df_copy[self.price_column].shift(1)
            hl_range = df_copy[high_col] - df_copy[low_col]
            hc_range = np.abs(df_copy[high_col] - prev_close)
            lc_range = np.abs(df_copy[low_col] - prev_close)
            
            tr_series = pd.concat([hl_range, hc_range, lc_range], axis=1).max(axis=1, skipna=False)
            
            atr_series = tr_series.ewm(span=self.atr_period_for_regime, adjust=False, min_periods=self.atr_period_for_regime).mean()
            
            self.df[self.internal_atr_col_name] = atr_series # Assign directly to self.df
            # Fill NaNs robustly
            self.df[self.internal_atr_col_name] = self.df[self.internal_atr_col_name].bfill()
            self.df[self.internal_atr_col_name] = self.df[self.internal_atr_col_name].ffill()
            if self.df[self.internal_atr_col_name].isnull().any(): # Should be rare after bfill then ffill
                self.df[self.internal_atr_col_name] = self.df[self.internal_atr_col_name].fillna(0)

            self.logger.info(f"Successfully calculated and added '{self.internal_atr_col_name}' to DataFrame.")
            return True
        except Exception as e:
            self.logger.error(f"Error during internal ATR calculation for '{self.internal_atr_col_name}': {e}", exc_info=True)
            if self.internal_atr_col_name in self.df.columns: # Cleanup if partially created
                self.df.drop(columns=[self.internal_atr_col_name], inplace=True)
            return False

    def _get_market_regime(self, df_idx):
        if not self.has_atr_for_regime or self.atr_col_to_use_for_regime is None:
            return 0 

        safe_idx = min(max(0, df_idx), len(self.df) - 1)
        
        if self.atr_col_to_use_for_regime not in self.df.columns:
             self.logger.warning(f"ATR column '{self.atr_col_to_use_for_regime}' not found at df_idx {df_idx} for regime. Defaulting regime to 0.")
             return 0

        atr_val = self.df.loc[safe_idx, self.atr_col_to_use_for_regime]
        current_price = self.df.loc[safe_idx, self.price_column]
        
        if pd.isna(atr_val) or pd.isna(current_price) or current_price <= 1e-9:
            return 0 

        if (atr_val / current_price) > self.regime_atr_threshold_pct:
            return 1 
        return 0
    
    def _reset_trading_state(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0.0
        self.avg_entry_price = 0.0
        self.current_position = 0
        self.total_trades = 0
        self.total_fees = 0.0
        self.episode_realized_pnl = 0.0
        self.current_episode_step = 0
        self.current_stop_loss_price = None
        self.recent_net_worth_changes_for_vol_penalty.clear()

    def _get_price_at_idx(self, df_idx):
        safe_idx = min(max(0, df_idx), len(self.df) - 1)
        return self.df.loc[safe_idx, self.price_column]

    def _get_observation_at_idx(self, df_idx):
        safe_idx_features_end = min(max(self.lookback_window - 1, df_idx), len(self.df) - 1)
        frame_start_idx = max(0, safe_idx_features_end - self.lookback_window + 1)
        
        # Use self.actual_features_in_df_for_obs which was filtered in __init__
        if not self.actual_features_in_df_for_obs and self.lookback_window > 0:
            # if features_to_use was non-empty but none are in df, this creates zeros for requested number of features
            # if features_to_use was empty, num_technical_features_for_shape is 0, so flat part is empty.
            num_orig_requested_features = len(self.features_to_use)
            obs_features_flat = np.zeros(self.lookback_window * num_orig_requested_features, dtype=np.float32)
            if num_orig_requested_features > 0 : # Only log warning if features were expected
                self.logger.warning(f"No features from 'features_to_use' list available in self.df at idx {df_idx}. Using zeros for feature part of obs.")
        elif self.lookback_window == 0: # No lookback, features are from current step only (or empty if none)
            if not self.actual_features_in_df_for_obs:
                obs_features_flat = np.array([], dtype=np.float32)
            else:
                current_features_series = self.df.loc[safe_idx_features_end, self.actual_features_in_df_for_obs]
                obs_features_flat = current_features_series.values.astype(np.float32).flatten()
        else: # Default case: features available and lookback > 0
            obs_features_df = self.df.loc[frame_start_idx : safe_idx_features_end, self.actual_features_in_df_for_obs]
            obs_features_values = obs_features_df.values.astype(np.float32)
            if len(obs_features_values) < self.lookback_window:
                padding_rows = self.lookback_window - len(obs_features_values)
                num_cols = obs_features_df.shape[1] if not obs_features_df.empty else len(self.actual_features_in_df_for_obs)
                padding = np.zeros((padding_rows, num_cols), dtype=np.float32)
                obs_features_values = np.vstack((padding, obs_features_values))
            obs_features_flat = obs_features_values.flatten()

        current_price_for_portfolio_state = self._get_price_at_idx(df_idx)
        norm_balance = self.balance / self.initial_balance if self.initial_balance > 0 else 0.0
        unrealized_pnl = 0.0
        if self.current_position == 1: 
            unrealized_pnl = (current_price_for_portfolio_state - self.avg_entry_price) * self.shares_held
        elif self.current_position == -1: 
            unrealized_pnl = (self.avg_entry_price - current_price_for_portfolio_state) * self.shares_held
        norm_unrealized_pnl = unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0.0
        norm_entry_price = 0.0
        if self.current_position != 0 and current_price_for_portfolio_state > 1e-9 and self.avg_entry_price > 1e-9:
            norm_entry_price = (self.avg_entry_price / current_price_for_portfolio_state) 
        
        portfolio_state_np = np.array([norm_balance, norm_unrealized_pnl, float(self.current_position), norm_entry_price], dtype=np.float32)
        market_regime_np = np.array([float(self._get_market_regime(df_idx))], dtype=np.float32)

        obs = np.concatenate([portfolio_state_np, market_regime_np, obs_features_flat])
        
        if obs.shape != self.observation_space_shape:
            expected_total_len = self.observation_space_shape[0]
            current_total_len = obs.shape[0]
            expected_feat_len_calc = self.lookback_window * len(self.actual_features_in_df_for_obs)
            
            self.logger.error(
                f"Obs shape mismatch: expected {self.observation_space_shape} (total len {expected_total_len}), "
                f"got {obs.shape} (total len {current_total_len}) at df_idx {df_idx}. "
                f"Portfolio: {portfolio_state_np.shape}, Regime: {market_regime_np.shape}, Features_flat: {obs_features_flat.shape} "
                f"(Expected flat feature part based on actual features in df: {expected_feat_len_calc}). "
                f"Num actual features used for obs_space_shape: {len(self.actual_features_in_df_for_obs)}"
            )
            raise ValueError(f"Observation shape mismatch. Expected {self.observation_space_shape}, got {obs.shape}")
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self._reset_trading_state()
        min_start_idx = self.lookback_window - 1 
        min_data_len_required = self.lookback_window if self.episode_max_steps == 0 else self.lookback_window + 1
        if len(self.df) < min_data_len_required:
             raise ValueError(f"DataFrame too short ({len(self.df)} rows). Needs {min_data_len_required} for lookback={self.lookback_window} + potential step.")
        max_random_start_idx = len(self.df) - 2 
        if self.serial: 
            self.current_df_idx = min_start_idx
        else: 
            if max_random_start_idx < min_start_idx: 
                 self.current_df_idx = min_start_idx 
                 self.logger.debug(f"DataFrame very short (len {len(self.df)}), serial start forced at idx {min_start_idx}.")
            else:
                 self.current_df_idx = np.random.randint(min_start_idx, max_random_start_idx + 1) 
        self.logger.debug(f"Env reset. Serial: {self.serial}. Start df_idx: {self.current_df_idx}. DF len: {len(self.df)}")
        return self._get_observation_at_idx(self.current_df_idx), {}

    def _apply_slippage(self, price, shares_traded, action_direction):
        """ 
        Apply slippage based on trade value. 
        action_direction: 1 for buy (price increases), -1 for sell (price decreases).
        """
        if self.slippage_factor_per_1k_value <= 0 or shares_traded <= 0 or price <= 1e-9:
            return price
        
        trade_value = shares_traded * price
        
        # Calculate how many "value units" this trade represents
        num_value_units = trade_value / self.slippage_value_unit
        
        # Total slippage percentage for this trade
        total_slippage_pct = self.slippage_factor_per_1k_value * num_value_units
        
        slippage_amount = price * total_slippage_pct
        
        if action_direction == 1: # Buying, price gets worse (higher)
            slipped_price = price + slippage_amount
            self.logger.debug(f"Slippage (BUY): Orig Price: {price:.2f}, Shares: {shares_traded:.4f}, Value: {trade_value:.2f}, Slipped Price: {slipped_price:.2f}")
            return slipped_price
        elif action_direction == -1: # Selling, price gets worse (lower)
            slipped_price = price - slippage_amount
            # Ensure price doesn't go negative due to slippage, though unlikely with small factors
            slipped_price = max(1e-9, slipped_price) 
            self.logger.debug(f"Slippage (SELL): Orig Price: {price:.2f}, Shares: {shares_traded:.4f}, Value: {trade_value:.2f}, Slipped Price: {slipped_price:.2f}")
            return slipped_price
        return price

    def _execute_trade(self, action_choice, current_market_price):
        action_taken_flag = False
        realized_pnl_from_trade = 0.0
        
        action_type = "HOLD"
        position_fraction_idx = -1
        chosen_fraction = 0.0

        if 0 < action_choice <= self.n_position_fractions:
            action_type = "BUY_OR_COVER"
            position_fraction_idx = action_choice - 1
            chosen_fraction = self.position_size_fractions[position_fraction_idx]
        elif self.n_position_fractions < action_choice < (1 + 2 * self.n_position_fractions) : # Corrected upper bound
            action_type = "SELL_OR_SHORT"
            position_fraction_idx = action_choice - (self.n_position_fractions + 1)
            chosen_fraction = self.position_size_fractions[position_fraction_idx]

        if current_market_price <= 1e-9 or action_type == "HOLD" or chosen_fraction <= 1e-8: # Also check if fraction is meaningful
            return realized_pnl_from_trade, action_taken_flag

        if action_type == "BUY_OR_COVER":
            if self.current_position == -1: # Cover Short
                shares_to_cover = self.shares_held * chosen_fraction
                if shares_to_cover < 1e-8: return realized_pnl_from_trade, action_taken_flag
                execution_price = self._apply_slippage(current_market_price, shares_to_cover, 1) # Buying back
                cost_to_buy_back = shares_to_cover * execution_price
                fees = cost_to_buy_back * self.commission_fee
                if self.balance >= cost_to_buy_back + fees :
                    self.balance -= (cost_to_buy_back + fees)
                    realized_pnl_from_trade = (self.avg_entry_price - execution_price) * shares_to_cover - fees
                    self.episode_realized_pnl += realized_pnl_from_trade; self.shares_held -= shares_to_cover
                    if self.shares_held < 1e-8: self.shares_held = 0.0; self.current_position = 0; self.avg_entry_price = 0.0; self.current_stop_loss_price = None
                    self.total_trades += 1; self.total_fees += fees; action_taken_flag = True
            elif self.current_position == 0: # Open Long
                if self.balance > 10: # Arbitrary min balance check
                    investable_balance = self.balance * chosen_fraction
                    amount_for_shares = investable_balance / (1 + self.commission_fee) # Max amount for shares after fee
                    shares_to_buy_est = amount_for_shares / current_market_price
                    if shares_to_buy_est < 1e-8: return realized_pnl_from_trade, action_taken_flag
                    
                    execution_price = self._apply_slippage(current_market_price, shares_to_buy_est, 1)
                    shares_to_buy = amount_for_shares / execution_price # Recalculate with slippage
                    cost = shares_to_buy * execution_price; fees = cost * self.commission_fee

                    if self.balance >= cost + fees and shares_to_buy > 1e-8:
                        self.balance -= (cost + fees); 
                        # for opening new position, shares_held is set, not added to.
                        # if allowing averaging up, this logic would change.
                        self.shares_held = shares_to_buy 
                        self.avg_entry_price = execution_price; self.current_position = 1
                        self.current_stop_loss_price = execution_price * (1 - self.stop_loss_pct)
                        realized_pnl_from_trade = -fees # Cost of opening
                        self.total_trades += 1; self.total_fees += fees; action_taken_flag = True
            # else: current_position == 1 (already long), BUY_OR_COVER might mean "add to position" - not implemented here.

        elif action_type == "SELL_OR_SHORT":
            if self.current_position == 1: # Sell Long
                shares_to_sell = self.shares_held * chosen_fraction
                if shares_to_sell < 1e-8: return realized_pnl_from_trade, action_taken_flag
                execution_price = self._apply_slippage(current_market_price, shares_to_sell, -1) # Selling
                sale_value = shares_to_sell * execution_price; fees = sale_value * self.commission_fee
                realized_pnl_from_trade = (execution_price - self.avg_entry_price) * shares_to_sell - fees
                self.episode_realized_pnl += realized_pnl_from_trade; self.balance += (sale_value - fees)
                self.shares_held -= shares_to_sell
                if self.shares_held < 1e-8: self.shares_held = 0.0; self.current_position = 0; self.avg_entry_price = 0.0; self.current_stop_loss_price = None
                self.total_trades += 1; self.total_fees += fees; action_taken_flag = True
            elif self.current_position == 0: # open Short
                if self.balance > 10:
                    value_to_short_basis = self.balance * chosen_fraction
                    amount_for_shares_short = value_to_short_basis
                    shares_to_short_est = amount_for_shares_short / current_market_price
                    if shares_to_short_est < 1e-8: return realized_pnl_from_trade, action_taken_flag

                    execution_price = self._apply_slippage(current_market_price, shares_to_short_est, -1)
                    shares_to_short = amount_for_shares_short / execution_price
                    proceeds = shares_to_short * execution_price; fees = proceeds * self.commission_fee
                    
                    if shares_to_short > 1e-8: 
                        self.balance += (proceeds - fees);
                        self.shares_held = shares_to_short
                        self.avg_entry_price = execution_price; self.current_position = -1
                        self.current_stop_loss_price = execution_price * (1 + self.stop_loss_pct)
                        realized_pnl_from_trade = -fees # cost of opening
                        self.total_trades += 1; self.total_fees += fees; action_taken_flag = True
        
        # update Net Worth after any trade
        price_for_nw_update = self._get_price_at_idx(self.current_df_idx) # use current index's price
        if self.current_position == 1: self.net_worth = self.balance + (self.shares_held * price_for_nw_update)
        elif self.current_position == -1: self.net_worth = self.balance - (self.shares_held * price_for_nw_update) # liability
        else: self.net_worth = self.balance
        
        return realized_pnl_from_trade, action_taken_flag

    def step(self, action_choice):
        self.current_episode_step += 1; net_worth_before_step = self.net_worth
        reward_penalty_component = 0.0; forced_liquidation = False
        self.current_df_idx += 1
        if self.current_df_idx >= len(self.df):
            truncated = True; terminated = False
            if self.current_position != 0: reward_penalty_component -= self.holding_penalty_ratio * self.initial_balance
            reward = reward_penalty_component / self.initial_balance if self.initial_balance > 0 else 0.0
            obs = self._get_observation_at_idx(self.current_df_idx - 1)
            info = self._get_info(self._get_price_at_idx(self.current_df_idx -1) , reward * self.initial_balance, net_worth_before_step, reward_penalty_component)
            return obs, reward, terminated, truncated, info

        current_market_price = self._get_price_at_idx(self.current_df_idx)
        try: idx_100_percent_fraction = self.position_size_fractions.index(1.0)
        except ValueError: idx_100_percent_fraction = self.n_position_fractions - 1
        action_cover_100_percent = 1 + idx_100_percent_fraction
        action_sell_100_percent = 1 + self.n_position_fractions + idx_100_percent_fraction

        if self.current_position == 1 and self.current_stop_loss_price and current_market_price <= self.current_stop_loss_price:
            self.logger.info(f"SL LONG at {current_market_price:.2f} (SL: {self.current_stop_loss_price:.2f})")
            self._execute_trade(action_sell_100_percent , current_market_price); forced_liquidation = True
        elif self.current_position == -1:
            if self.current_stop_loss_price and current_market_price >= self.current_stop_loss_price:
                self.logger.info(f"SL SHORT at {current_market_price:.2f} (SL: {self.current_stop_loss_price:.2f})")
                self._execute_trade(action_cover_100_percent, current_market_price); forced_liquidation = True
            if not forced_liquidation and (self.balance - (self.shares_held * current_market_price)) < self.short_sell_margin_floor:
                self.logger.warning(f"MARGIN CALL SHORT at {current_market_price:.2f}. Forcing cover.")
                self._execute_trade(action_cover_100_percent, current_market_price); forced_liquidation = True
        
        action_executed_flag_agent = False
        if not forced_liquidation: _, action_executed_flag_agent = self._execute_trade(action_choice, current_market_price)

        if action_executed_flag_agent or forced_liquidation: reward_penalty_component -= self.trade_penalty_ratio * self.initial_balance
        elif self.current_position != 0: reward_penalty_component -= self.holding_penalty_ratio * self.initial_balance
        
        step_pnl = (self.net_worth - net_worth_before_step)
        mtm_reward = step_pnl / self.initial_balance if self.initial_balance > 0 else 0.0
        self.recent_net_worth_changes_for_vol_penalty.append(step_pnl)
        volatility_reward_penalty = 0.0
        if len(self.recent_net_worth_changes_for_vol_penalty) >= max(2, self.lookback_window // 20): # Adjusted window
            pnl_std_dev = np.std(list(self.recent_net_worth_changes_for_vol_penalty))
            if self.initial_balance > 0: volatility_reward_penalty = -self.volatility_penalty_coeff * (pnl_std_dev / self.initial_balance)
        reward = mtm_reward + (reward_penalty_component / self.initial_balance if self.initial_balance > 0 else 0) + volatility_reward_penalty

        terminated = False; truncated = False
        if self.current_df_idx >= len(self.df) - 1: truncated = True
        if self.episode_max_steps and self.current_episode_step >= self.episode_max_steps and not truncated: truncated = True
        if self.net_worth <= self.initial_balance * 0.5 and not (terminated or truncated): 
            terminated = True; reward += self.bankruptcy_penalty
            self.logger.info(f"BANKRUPTCY idx {self.current_df_idx}. NW: {self.net_worth:.2f}")
        
        obs = self._get_observation_at_idx(self.current_df_idx)
        info = self._get_info(current_market_price, reward * self.initial_balance, net_worth_before_step, 
                              reward_penalty_component, volatility_reward_penalty * self.initial_balance)
        return obs, reward, terminated, truncated, info

    def _get_info(self, current_price, final_step_reward_unnormalized, nw_before_step, penalty_comp_unnormalized, vol_penalty_comp_unnormalized=0.0):
        return {
            'net_worth': self.net_worth, 'episode_realized_pnl': self.episode_realized_pnl,
            'current_position': self.current_position, 'trades': self.total_trades,
            'current_price': current_price, 'df_idx': self.current_df_idx, 'balance': self.balance,
            'shares_held': self.shares_held, 'avg_entry_price': self.avg_entry_price,
            'stop_loss_price': self.current_stop_loss_price,
            'reward_mtm_step': (self.net_worth - nw_before_step),
            'reward_penalty_component': penalty_comp_unnormalized,
            'reward_volatility_penalty_component': vol_penalty_comp_unnormalized,
            'reward_final_step': final_step_reward_unnormalized
        }

    def render(self, mode='human'):
        price = self._get_price_at_idx(self.current_df_idx)
        pos_map = {0: "FLAT", 1: "LONG", -1: "SHORT"}
        sl_price = self.current_stop_loss_price if self.current_stop_loss_price else 0.0
        print(
            f"Stp: {self.current_episode_step:03d}|Idx: {self.current_df_idx:05d}|Prc: {price:<7.2f}|"
            f"Pos: {pos_map[self.current_position]:<5}|Shr: {self.shares_held:<8.4f}|Entry: {self.avg_entry_price:<7.2f}|"
            f"SL: {sl_price:<7.2f}|Bal: {self.balance:<9.2f}|NW: {self.net_worth:<9.2f}|Trd: {self.total_trades:02d}"
        )

    def close(self):
        self.logger.info("TradingEnv (Advanced Version) closed.")