"""
Configuration management for QuantLab application.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Data processing configuration."""
    default_ticker: str = "AAPL"
    default_period_years: int = 5
    cache_ttl_seconds: int = 3600
    min_data_points: int = 50


@dataclass
class MLConfig:
    """Machine learning configuration."""
    default_forecast_horizon: int = 5
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    min_samples_for_training: int = 100
    
    # Model hyperparameters
    rf_default_params: Dict = None
    xgb_default_params: Dict = None
    lstm_default_params: Dict = None
    
    def __post_init__(self):
        if self.rf_default_params is None:
            self.rf_default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': self.random_state
            }
        
        if self.xgb_default_params is None:
            self.xgb_default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': self.random_state
            }
        
        if self.lstm_default_params is None:
            self.lstm_default_params = {
                'sequence_length': 60,
                'epochs': 50,
                'batch_size': 32
            }


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    min_position_size: float = 100.0  # Minimum $100 position
    max_position_pct: float = 0.2  # Max 20% of portfolio per position
    rebalance_frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class UIConfig:
    """UI configuration."""
    page_title: str = "QuantLab - Professional Trading Simulator"
    page_icon: str = "📈"
    layout: str = "wide"
    theme_primary_color: str = "#2a5298"


@dataclass
class AppConfig:
    """Main application configuration."""
    data: DataConfig
    ml: MLConfig
    backtest: BacktestConfig
    ui: UIConfig
    
    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> 'AppConfig':
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return cls(
                data=DataConfig(**config_dict.get('data', {})),
                ml=MLConfig(**config_dict.get('ml', {})),
                backtest=BacktestConfig(**config_dict.get('backtest', {})),
                ui=UIConfig(**config_dict.get('ui', {}))
            )
        else:
            # Return default configuration
            return cls(
                data=DataConfig(),
                ml=MLConfig(),
                backtest=BacktestConfig(),
                ui=UIConfig()
            )
    
    def save_to_file(self, config_path: Optional[str] = None):
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        config_dict = {
            'data': self.data.__dict__,
            'ml': self.ml.__dict__,
            'backtest': self.backtest.__dict__,
            'ui': self.ui.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Global configuration instance
config = AppConfig.load_from_file()