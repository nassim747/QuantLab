import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    mse: float
    rmse: float
    mae: float
    r2: float
    hit_rate: float
    directional_accuracy: float


class MLModelTrainer:
    """
    Enhanced ML model trainer with proper return prediction and validation.
    
    Features:
    - Return-based prediction (not price prediction)
    - Feature scaling and normalization
    - Time-series aware validation
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, use_scaling: bool = True):
        self.model: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = StandardScaler() if use_scaling else None
        self.use_scaling = use_scaling
        self.is_trained = False
        
        logger.info(f"Initialized MLModelTrainer with scaling: {use_scaling}")
    
    def get_model(self, model_type: str, **hyperparams) -> Any:
        """
        Get model instance with hyperparameters.
        
        Args:
            model_type: Type of model to create
            **hyperparams: Model-specific hyperparameters
        
        Returns:
            Configured model instance
        """
        logger.debug(f"Creating {model_type} with params: {hyperparams}")
        
        if model_type == "Linear Regression":
            return LinearRegression()
        elif model_type == "Random Forest Regressor":
            return RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 10),
                min_samples_split=hyperparams.get('min_samples_split', 5),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "XGBoost Regressor":
            return XGBRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 6),
                subsample=hyperparams.get('subsample', 0.8),
                colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
                use_label_encoder=False,
                eval_metric='rmse',
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(self, X_train, y_train, model_type, **hyperparams):
        """Train the selected model with optional feature scaling."""
        # Fit/transform scaler if enabled
        if self.use_scaling and self.scaler is not None:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
        else:
            X_train_scaled = X_train

        self.model = self.get_model(model_type, **hyperparams)
        self.model_type = model_type
        self.feature_names = X_train.columns.tolist()

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        return self.model
    
    def predict(self, X):
        """Make predictions on new data with the proper scaling pipeline."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if self.use_scaling and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=X.columns
            )
        else:
            X_scaled = X

        return self.model.predict(X_scaled)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Hit-rate: correct forecast of return direction
        actual_direction = np.sign(y_test)
        pred_direction = np.sign(y_pred)
        hit_rate = np.mean(actual_direction == pred_direction)
        metrics['hit_rate'] = hit_rate if not np.isnan(hit_rate) else 0.0

        return metrics, y_pred
    
    def get_feature_importance(self):
        """Get feature importance if available."""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': self.model.coef_
            })
            return importance_df
        else:
            return None


class TimeSeriesValidator:
    """Time series cross-validation for model selection."""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def validate_model(self, X, y, model_type, param_grid=None):
        """Perform time series cross-validation."""
        model = self._get_base_model(model_type)
        
        if param_grid is None:
            param_grid = self._get_default_param_grid(model_type)
        
        # Perform grid search with time series CV
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=self.tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def _get_base_model(self, model_type):
        """Get base model for validation."""
        if model_type == "Random Forest Regressor":
            return RandomForestRegressor(random_state=42)
        elif model_type == "XGBoost Regressor":
            return XGBRegressor(use_label_encoder=False, eval_metric='rmse', random_state=42)
        else:
            return LinearRegression()
    
    def _get_default_param_grid(self, model_type):
        """Get default parameter grid for hyperparameter tuning."""
        if model_type == "Random Forest Regressor":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == "XGBoost Regressor":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        else:
            return {}


def calculate_trading_metrics(returns):
    """Calculate trading strategy metrics."""
    if len(returns) == 0:
        return {}
    
    returns = pd.Series(returns)
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + returns).mean() ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = ann_return / downside_std if downside_std > 0 else 0
    
    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }