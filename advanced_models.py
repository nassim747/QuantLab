import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class LSTMModel:
    """LSTM model for time series prediction."""
    
    def __init__(self, sequence_length=60, epochs=50, batch_size=32):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def prepare_sequences(self, data, target_col='Close'):
        """Prepare sequences for LSTM training."""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_col]])
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train(self, data, target_col='Close', validation_split=0.2):
        """Train the LSTM model."""
        X, y = self.prepare_sequences(data, target_col)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Reshape for LSTM input
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self.build_model((X.shape[1], 1))
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        return history
    
    def predict(self, data, target_col='Close'):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X, _ = self.prepare_sequences(data, target_col)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def predict_future(self, data, target_col='Close', days=5):
        """Predict future values."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Get the last sequence
        scaled_data = self.scaler.transform(data[[target_col]])
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()


class ProphetModel:
    """Facebook Prophet model for time series forecasting."""
    
    def __init__(self):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required. Install with: pip install prophet")
        
        self.model = None
        self.fitted = False
    
    def prepare_data(self, data, target_col='Close'):
        """Prepare data for Prophet (requires 'ds' and 'y' columns)."""
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data[target_col]
        })
        return prophet_data
    
    def train(self, data, target_col='Close', **prophet_kwargs):
        """Train the Prophet model."""
        prophet_data = self.prepare_data(data, target_col)
        
        # Default Prophet parameters
        default_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05
        }
        default_params.update(prophet_kwargs)
        
        self.model = Prophet(**default_params)
        
        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_data)
        
        self.fitted = True
        return self.model
    
    def predict(self, periods=30):
        """Make future predictions."""
        if not self.fitted:
            raise ValueError("Model must be trained before making predictions")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast
    
    def predict_on_data(self, data, target_col='Close'):
        """Make predictions on existing data."""
        if not self.fitted:
            raise ValueError("Model must be trained before making predictions")
        
        prophet_data = self.prepare_data(data, target_col)
        forecast = self.model.predict(prophet_data[['ds']])
        
        return forecast['yhat'].values


class EnsembleModel:
    """Ensemble model combining multiple predictors."""
    
    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights or []
        self.trained_models = []
        
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)
    
    def train(self, X_train, y_train, **kwargs):
        """Train all models in the ensemble."""
        self.trained_models = []
        
        for model in self.models:
            if hasattr(model, 'fit'):
                # Sklearn-style model
                trained_model = model.fit(X_train, y_train)
                self.trained_models.append(trained_model)
            else:
                # Custom model with train method
                trained_model = model.train(X_train, y_train, **kwargs)
                self.trained_models.append(model)  # Store the model object itself
    
    def predict(self, X_test):
        """Make ensemble predictions."""
        if not self.trained_models:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        
        for i, model in enumerate(self.trained_models):
            if hasattr(model, 'predict'):
                pred = model.predict(X_test)
                predictions.append(pred * self.weights[i])
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble models")
        
        # Weighted average of predictions
        ensemble_pred = np.sum(predictions, axis=0) / np.sum(self.weights)
        return ensemble_pred


class AdvancedMLPipeline:
    """Advanced ML pipeline with multiple model types."""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def prepare_time_series_split(self, data, test_size=0.2):
        """Create time-series aware train/test split."""
        split_idx = int(len(data) * (1 - test_size))
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    def train_all_models(self, data, target_col='Close', feature_cols=None):
        """Train all available models."""
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]
        
        # Prepare data
        train_data, test_data = self.prepare_time_series_split(data)
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        results = {}
        
        # Train LSTM if available
        if TENSORFLOW_AVAILABLE and len(train_data) > 100:
            try:
                lstm_model = LSTMModel(sequence_length=min(60, len(train_data)//4))
                lstm_model.train(train_data, target_col)
                
                # Make predictions
                lstm_pred = lstm_model.predict(data, target_col)
                
                # Align predictions with test data
                if len(lstm_pred) >= len(y_test):
                    lstm_pred_aligned = lstm_pred[-len(y_test):]
                    
                    self.models['LSTM'] = lstm_model
                    self.predictions['LSTM'] = lstm_pred_aligned
                    self.metrics['LSTM'] = self._calculate_metrics(y_test, lstm_pred_aligned)
                    
                    results['LSTM'] = {
                        'model': lstm_model,
                        'predictions': lstm_pred_aligned,
                        'metrics': self.metrics['LSTM']
                    }
            except Exception as e:
                print(f"LSTM model training failed: {e}")
        
        # Train Prophet if available
        if PROPHET_AVAILABLE and len(train_data) > 30:
            try:
                prophet_model = ProphetModel()
                prophet_model.train(train_data, target_col)
                
                # Make predictions on test data
                prophet_pred = prophet_model.predict_on_data(test_data, target_col)
                
                self.models['Prophet'] = prophet_model
                self.predictions['Prophet'] = prophet_pred
                self.metrics['Prophet'] = self._calculate_metrics(y_test, prophet_pred)
                
                results['Prophet'] = {
                    'model': prophet_model,
                    'predictions': prophet_pred,
                    'metrics': self.metrics['Prophet']
                }
            except Exception as e:
                print(f"Prophet model training failed: {e}")
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate model performance metrics."""
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Direction accuracy
            direction_actual = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            direction_accuracy = np.mean(direction_actual == direction_pred) if len(direction_actual) > 0 else 0
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_best_model(self, metric='rmse'):
        """Get the best performing model based on specified metric."""
        if not self.metrics:
            return None
        
        best_model = None
        best_score = float('inf') if metric in ['mse', 'rmse', 'mae'] else float('-inf')
        
        for model_name, metrics in self.metrics.items():
            if metric in metrics:
                score = metrics[metric]
                if (metric in ['mse', 'rmse', 'mae'] and score < best_score) or \
                   (metric in ['r2', 'direction_accuracy'] and score > best_score):
                    best_score = score
                    best_model = model_name
        
        return best_model
    
    def create_ensemble(self, model_names=None):
        """Create ensemble from trained models."""
        if model_names is None:
            model_names = list(self.models.keys())
        
        ensemble_models = [self.models[name] for name in model_names if name in self.models]
        
        if len(ensemble_models) > 1:
            # Weight models based on their performance (inverse of RMSE)
            weights = []
            for name in model_names:
                if name in self.metrics and 'rmse' in self.metrics[name]:
                    rmse = self.metrics[name]['rmse']
                    weight = 1 / (rmse + 1e-8)  # Add small epsilon to avoid division by zero
                    weights.append(weight)
                else:
                    weights.append(1.0)
            
            ensemble = EnsembleModel(ensemble_models, weights)
            return ensemble
        
        return None