import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml_models import MLModelTrainer, TimeSeriesValidator, calculate_trading_metrics


class TestMLModels:
    
    def create_sample_ml_data(self, n_samples=100, n_features=5):
        """Create sample ML data for testing."""
        np.random.seed(42)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Generate features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=feature_names,
            index=pd.date_range('2023-01-01', periods=n_samples, freq='D')
        )
        
        # Generate target with some relationship to features
        y = pd.Series(
            X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5,
            index=X.index,
            name='target'
        )
        
        return X, y
    
    def test_ml_model_trainer_initialization(self):
        """Test MLModelTrainer initialization."""
        trainer = MLModelTrainer()
        
        assert trainer.model is None
        assert trainer.model_type is None
        assert trainer.feature_names is None
        assert trainer.scaler is None
    
    def test_get_model_linear_regression(self):
        """Test getting Linear Regression model."""
        trainer = MLModelTrainer()
        model = trainer.get_model("Linear Regression")
        
        assert isinstance(model, LinearRegression)
    
    def test_get_model_random_forest(self):
        """Test getting Random Forest model with hyperparameters."""
        trainer = MLModelTrainer()
        model = trainer.get_model(
            "Random Forest Regressor",
            n_estimators=200,
            max_depth=15
        )
        
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 200
        assert model.max_depth == 15
        assert model.random_state == 42
    
    def test_get_model_invalid_type(self):
        """Test getting model with invalid type."""
        trainer = MLModelTrainer()
        
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer.get_model("Invalid Model")
    
    def test_train_model(self):
        """Test model training."""
        trainer = MLModelTrainer()
        X_train, y_train = self.create_sample_ml_data(80, 3)
        
        # Train Linear Regression
        model = trainer.train_model(X_train, y_train, "Linear Regression")
        
        assert trainer.model is not None
        assert trainer.model_type == "Linear Regression"
        assert trainer.feature_names == X_train.columns.tolist()
        assert hasattr(model, 'coef_')
    
    def test_predict_without_training(self):
        """Test prediction without training model first."""
        trainer = MLModelTrainer()
        X_test, _ = self.create_sample_ml_data(20, 3)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            trainer.predict(X_test)
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        trainer = MLModelTrainer()
        X_train, y_train = self.create_sample_ml_data(80, 3)
        X_test, _ = self.create_sample_ml_data(20, 3)
        
        # Train model
        trainer.train_model(X_train, y_train, "Linear Regression")
        
        # Make predictions
        predictions = trainer.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        trainer = MLModelTrainer()
        X_train, y_train = self.create_sample_ml_data(80, 3)
        X_test, y_test = self.create_sample_ml_data(20, 3)
        
        # Train model
        trainer.train_model(X_train, y_train, "Random Forest Regressor", n_estimators=10)
        
        # Evaluate model
        metrics, predictions = trainer.evaluate_model(X_test, y_test)
        
        # Check that all expected metrics are present
        expected_metrics = ['mse', 'rmse', 'mae', 'r2', 'hit_rate']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check predictions
        assert len(predictions) == len(y_test)
        assert isinstance(predictions, np.ndarray)
        
        # Check metric ranges
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert 0 <= metrics['hit_rate'] <= 1
    
    def test_get_feature_importance_random_forest(self):
        """Test feature importance for Random Forest."""
        trainer = MLModelTrainer()
        X_train, y_train = self.create_sample_ml_data(80, 3)
        
        # Train Random Forest model
        trainer.train_model(X_train, y_train, "Random Forest Regressor", n_estimators=10)
        
        # Get feature importance
        importance = trainer.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == len(X_train.columns)
        
        # Check that importance values are non-negative and sorted
        assert all(importance['importance'] >= 0)
        assert importance['importance'].is_monotonic_decreasing
    
    def test_get_feature_importance_linear_regression(self):
        """Test feature importance for Linear Regression (coefficients)."""
        trainer = MLModelTrainer()
        X_train, y_train = self.create_sample_ml_data(80, 3)
        
        # Train Linear Regression model
        trainer.train_model(X_train, y_train, "Linear Regression")
        
        # Get feature importance (coefficients)
        importance = trainer.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns
        assert len(importance) == len(X_train.columns)
    
    def test_get_feature_importance_no_model(self):
        """Test feature importance when no model is trained."""
        trainer = MLModelTrainer()
        importance = trainer.get_feature_importance()
        
        assert importance is None


class TestTimeSeriesValidator:
    
    def create_time_series_data(self, n_samples=100):
        """Create time series data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        X = pd.DataFrame(
            np.random.randn(n_samples, 3),
            columns=['feature_1', 'feature_2', 'feature_3'],
            index=dates
        )
        
        y = pd.Series(
            X['feature_1'] + np.random.randn(n_samples) * 0.1,
            index=dates
        )
        
        return X, y
    
    def test_time_series_validator_initialization(self):
        """Test TimeSeriesValidator initialization."""
        validator = TimeSeriesValidator(n_splits=3)
        
        assert validator.n_splits == 3
        assert validator.tscv.n_splits == 3
    
    def test_validate_model_linear_regression(self):
        """Test model validation for Linear Regression."""
        validator = TimeSeriesValidator(n_splits=3)
        X, y = self.create_time_series_data(50)
        
        results = validator.validate_model(X, y, "Linear Regression")
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'cv_results' in results
        assert isinstance(results['best_score'], float)
        assert results['best_score'] >= 0  # MSE should be non-negative
    
    def test_validate_model_random_forest(self):
        """Test model validation for Random Forest."""
        validator = TimeSeriesValidator(n_splits=3)
        X, y = self.create_time_series_data(100)
        
        # Use a small parameter grid for faster testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        results = validator.validate_model(X, y, "Random Forest Regressor", param_grid)
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'cv_results' in results
        
        # Check that best parameters are from our grid
        assert results['best_params']['n_estimators'] in [10, 20]
        assert results['best_params']['max_depth'] in [3, 5]
    
    def test_get_default_param_grid(self):
        """Test default parameter grids."""
        validator = TimeSeriesValidator()
        
        # Test Random Forest grid
        rf_grid = validator._get_default_param_grid("Random Forest Regressor")
        expected_rf_keys = ['n_estimators', 'max_depth', 'min_samples_split']
        for key in expected_rf_keys:
            assert key in rf_grid
        
        # Test XGBoost grid
        xgb_grid = validator._get_default_param_grid("XGBoost Regressor")
        expected_xgb_keys = ['n_estimators', 'learning_rate', 'max_depth']
        for key in expected_xgb_keys:
            assert key in xgb_grid
        
        # Test Linear Regression (should be empty)
        lr_grid = validator._get_default_param_grid("Linear Regression")
        assert len(lr_grid) == 0


class TestTradingMetrics:
    
    def test_calculate_trading_metrics_empty_returns(self):
        """Test trading metrics with empty returns."""
        metrics = calculate_trading_metrics([])
        assert metrics == {}
    
    def test_calculate_trading_metrics_positive_returns(self):
        """Test trading metrics with positive returns."""
        # Create sample returns (daily)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
        
        metrics = calculate_trading_metrics(returns)
        
        expected_keys = [
            'total_return', 'ann_return', 'ann_vol', 'sharpe',
            'sortino', 'calmar', 'max_drawdown', 'win_rate'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Check metric ranges
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['max_drawdown'] <= 0  # Drawdown should be negative
        assert metrics['ann_vol'] >= 0  # Volatility should be non-negative
    
    def test_calculate_trading_metrics_negative_returns(self):
        """Test trading metrics with mostly negative returns."""
        returns = [-0.01] * 100  # Consistent negative returns
        
        metrics = calculate_trading_metrics(returns)
        
        # Total return should be negative
        assert metrics['total_return'] < 0
        assert metrics['win_rate'] == 0  # No positive returns
        assert metrics['max_drawdown'] < 0
    
    def test_calculate_trading_metrics_mixed_returns(self):
        """Test trading metrics with mixed returns."""
        returns = [0.02, -0.01, 0.015, -0.005, 0.01] * 50  # Mixed returns
        
        metrics = calculate_trading_metrics(returns)
        
        # Win rate should be between 0 and 1
        assert 0 < metrics['win_rate'] < 1
        
        # Check Sharpe ratio calculation
        assert isinstance(metrics['sharpe'], float)
        
        # Check Sortino ratio (should handle downside deviation)
        assert isinstance(metrics['sortino'], float)
    
    def test_calculate_trading_metrics_zero_volatility(self):
        """Test trading metrics with zero volatility."""
        returns = [0.01] * 100  # Constant positive returns
        
        metrics = calculate_trading_metrics(returns)
        
        # With zero volatility, Sharpe ratio should be very high or infinite
        # Our implementation should handle this gracefully
        assert metrics['ann_vol'] >= 0
        assert metrics['win_rate'] == 1  # All returns are positive
    
    def test_calculate_trading_metrics_single_return(self):
        """Test trading metrics with single return."""
        returns = [0.05]
        
        metrics = calculate_trading_metrics(returns)
        
        assert metrics['total_return'] == 0.05
        assert metrics['win_rate'] == 1.0
        # Other metrics should be calculated appropriately for single observation


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])