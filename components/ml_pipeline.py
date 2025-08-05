"""
ML Pipeline component for QuantLab Professional.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from ml_models import MLModelTrainer
from data_utils import prepare_ml_features
from utils.logger import get_logger

logger = get_logger(__name__)


class MLPipeline:
    """ML pipeline for training and evaluating models."""
    
    def __init__(self):
        self.trainer = MLModelTrainer(use_scaling=True)
        self.results = {}
        logger.info("MLPipeline initialized")
    
    def train_models(self, 
                    data: pd.DataFrame,
                    forecast_horizon: int = 5,
                    use_log_returns: bool = True,
                    model_types: List[str] = None,
                    enable_advanced: bool = False) -> Dict[str, Any]:
        """
        Train multiple ML models and return results.
        
        Args:
            data: Stock price data
            forecast_horizon: Days ahead to predict
            use_log_returns: Whether to use log returns
            model_types: List of model types to train
            enable_advanced: Whether to train advanced models
        
        Returns:
            Dictionary with model results
        """
        if model_types is None:
            model_types = ["Random Forest Regressor"]
        
        logger.info(f"Training {len(model_types)} models")
        
        # Prepare ML features
        ml_data, feature_cols = prepare_ml_features(
            data, forecast_horizon, use_log_returns
        )
        
        if len(ml_data) < 100:
            logger.warning("Insufficient data for ML training")
            return {}
        
        # Split data
        split_idx = int(len(ml_data) * 0.8)
        X_train = ml_data[feature_cols].iloc[:split_idx]
        X_test = ml_data[feature_cols].iloc[split_idx:]
        y_train = ml_data['Target'].iloc[:split_idx]
        y_test = ml_data['Target'].iloc[split_idx:]
        
        # Remove NaN values
        train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]
        
        results = {}
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type}")
                
                # Train model
                model = self.trainer.train_model(
                    X_train_clean, y_train_clean, model_type
                )
                
                # Evaluate
                metrics, predictions = self.trainer.evaluate_model(
                    X_test_clean, y_test_clean
                )
                
                results[model_type] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': predictions,
                    'test_data': {
                        'X_test': X_test_clean,
                        'y_test': y_test_clean
                    }
                }
                
                logger.info(f"{model_type} - R²: {metrics['r2']:.3f}, Hit Rate: {metrics['hit_rate']:.2%}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue
        
        self.results = results
        return results