"""
Reinforcement Learning for Trading Strategy Optimization
Uses Stable Baselines3 for PPO and DQN algorithms
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Any
import os

rl_logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, data, initial_balance=1000):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
        self.position_size = 0
        self.entry_price = 0
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price, balance, position, technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # 10 features
            dtype=np.float32
        )
        
        # Track rewards and metrics
        self.total_profit = 0
        self.trades_made = 0
        self.win_rate = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_balance = self.initial_balance
        self.current_position = 0
        self.position_size = 0
        self.entry_price = 0
        self.current_step = 0
        self.total_profit = 0
        self.trades_made = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation state"""
        if self.current_step >= len(self.data):
            return np.zeros(10, dtype=np.float32)
        
        current_data = self.data.iloc[self.current_step]
        
        # Calculate technical indicators
        price = current_data['close']
        
        # Simple features for observation
        obs = np.array([
            price / self.initial_balance,  # Normalized price
            self.current_balance / self.initial_balance,  # Normalized balance
            self.current_position,  # Current position
            self.position_size / self.initial_balance if self.position_size > 0 else 0,  # Position size
            (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0,  # PnL
            self._calculate_rsi(self.current_step),  # RSI
            self._calculate_ma(self.current_step, 10) / price if price > 0 else 1,  # MA ratio
            self._calculate_ma(self.current_step, 50) / price if price > 0 else 1,  # Long MA ratio
            self._calculate_volatility(self.current_step),  # Volatility
            self.current_step / self.max_steps  # Time progress
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_rsi(self, step, period=14):
        """Calculate RSI"""
        if step < period:
            return 50
        
        prices = self.data['close'].iloc[max(0, step-period):step+1]
        deltas = prices.diff()
        
        gains = deltas.where(deltas > 0, 0).mean()
        losses = (-deltas.where(deltas < 0, 0)).mean()
        
        if losses == 0:
            return 100
        
        rs = gains / losses
        return 100 - (100 / (1 + rs))
    
    def _calculate_ma(self, step, period):
        """Calculate Moving Average"""
        if step < period:
            return self.data['close'].iloc[step]
        
        return self.data['close'].iloc[max(0, step-period):step+1].mean()
    
    def _calculate_volatility(self, step, period=20):
        """Calculate volatility"""
        if step < period:
            return 0
        
        prices = self.data['close'].iloc[max(0, step-period):step+1]
        return prices.std() / prices.mean() if prices.mean() > 0 else 0
    
    def step(self, action):
        """Execute one time step"""
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, False, {}
        
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0
        
        # Execute action
        if action == 1 and self.current_position == 0:  # Buy
            self.current_position = 1
            self.entry_price = current_price
            self.position_size = self.current_balance * 0.95  # Use 95% of balance
            self.current_balance -= self.position_size
            self.trades_made += 1
            
        elif action == 2 and self.current_position == 1:  # Sell
            if self.entry_price > 0:
                profit = self.position_size * (current_price / self.entry_price - 1)
                self.current_balance += self.position_size + profit
                self.total_profit += profit
                
                # Reward based on profit
                reward = profit / self.initial_balance
                
                if profit > 0:
                    self.win_rate = (self.win_rate * (self.trades_made - 1) + 1) / self.trades_made
                else:
                    self.win_rate = (self.win_rate * (self.trades_made - 1) + 0) / self.trades_made
            
            self.current_position = 0
            self.position_size = 0
            self.entry_price = 0
        
        # Calculate unrealized PnL for holding position
        elif self.current_position == 1:
            unrealized_pnl = self.position_size * (current_price / self.entry_price - 1)
            reward = unrealized_pnl / self.initial_balance * 0.1  # Small reward for holding profitable position
        
        # Check if done
        done = self.current_step >= self.max_steps - 1
        
        # Additional reward for good performance
        if done:
            final_reward = (self.current_balance - self.initial_balance) / self.initial_balance
            reward += final_reward * 10  # Large reward for final performance
        
        return self._get_observation(), reward, done, False, {
            'balance': self.current_balance,
            'total_profit': self.total_profit,
            'trades_made': self.trades_made,
            'win_rate': self.win_rate
        }


class RLTradingOptimizer:
    """Reinforcement Learning optimizer for trading strategies"""
    
    def __init__(self, model_type='PPO'):
        self.model_type = model_type
        self.model = None
        self.env = None
        
    def train(self, data, total_timesteps=10000):
        """Train RL model"""
        try:
            from stable_baselines3 import PPO, DQN
            from stable_baselines3.common.env_util import make_vec_env
            
            # Create environment
            self.env = TradingEnvironment(data)
            
            # Create vectorized environment
            vec_env = make_vec_env(lambda: self.env, n_envs=4)
            
            # Create model based on type
            if self.model_type == 'PPO':
                self.model = PPO('MlpPolicy', vec_env, verbose=1)
            elif self.model_type == 'DQN':
                self.model = DQN('MlpPolicy', vec_env, verbose=1)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Train
            rl_logger.info(f"Training {self.model_type} model for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps)
            
            rl_logger.info(f"{self.model_type} training completed")
            
            return self.model
            
        except ImportError:
            rl_logger.warning("Stable Baselines3 not installed. RL training skipped.")
            return None
        except Exception as e:
            rl_logger.error(f"RL training failed: {e}")
            return None
    
    def get_optimal_action(self, observation):
        """Get optimal action from trained model"""
        if self.model is None:
            return 0  # Default to hold
        
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return action
        except Exception as e:
            rl_logger.error(f"Action prediction failed: {e}")
            return 0
    
    def save_model(self, path='rl_trading_model'):
        """Save trained model"""
        if self.model:
            self.model.save(path)
            rl_logger.info(f"RL model saved to {path}")
    
    def load_model(self, path='rl_trading_model'):
        """Load trained model"""
        try:
            from stable_baselines3 import PPO, DQN
            
            if self.model_type == 'PPO':
                self.model = PPO.load(path)
            elif self.model_type == 'DQN':
                self.model = DQN.load(path)
            
            rl_logger.info(f"RL model loaded from {path}")
            
        except ImportError:
            rl_logger.warning("Stable Baselines3 not installed.")
        except Exception as e:
            rl_logger.error(f"Failed to load RL model: {e}")


class StrategyOptimizer:
    """Optimize trading strategy parameters using RL"""
    
    def __init__(self):
        self.rl_optimizer = RLTradingOptimizer()
        self.best_parameters = {}
        
    def optimize_parameters(self, historical_data, parameter_ranges):
        """Optimize strategy parameters using RL"""
        rl_logger.info("Starting parameter optimization...")
        
        # This would implement a more sophisticated optimization
        # For now, return default parameters
        self.best_parameters = {
            'rsi_period': 7,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'take_profit_pct': 5,
            'stop_loss_pct': 2,
            'confidence_threshold': 0.15
        }
        
        rl_logger.info(f"Optimized parameters: {self.best_parameters}")
        return self.best_parameters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test RL environment
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Test environment
    env = TradingEnvironment(test_data)
    obs, info = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.4f}, Balance={info['balance']:.2f}")
