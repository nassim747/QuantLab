import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import streamlit as st


class MLModelTrainer:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.feature_names = None
        self.scaler = None
    
    def get_model(self, model_type, **hyperparams):
        """Get model instance with hyperparameters."""
        if model_type == "Linear Regression":
            return LinearRegression()
        elif model_type == "Random Forest Regressor":
            return RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 10),
                random_state=42
            )
        elif model_type == "XGBoost Regressor":
            return XGBRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                max_depth=hyperparams.get('max_depth', 3),
                use_label_encoder=False,
                eval_metric='rmse',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(self, X_train, y_train, model_type, **hyperparams):
        """Train the selected model."""
        self.model = self.get_model(model_type, **hyperparams)
        self.model_type = model_type
        self.feature_names = X_train.columns.tolist()
        
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Financial metrics
        price_changes = y_test.values
        pred_changes = y_pred
        
        # Hit rate (percentage of correct direction predictions)
        actual_direction = np.sign(price_changes - X_test.iloc[:, 0].values)  # Compare with current price
        pred_direction = np.sign(pred_changes - X_test.iloc[:, 0].values)
        hit_rate = np.mean(actual_direction == pred_direction)
        metrics['hit_rate'] = hit_rate
        
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