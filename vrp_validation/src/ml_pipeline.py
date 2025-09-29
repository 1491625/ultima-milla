import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class VRPPredictor:
    """
    A class to train, validate, and use ML models for VRP.
    It predicts travel times and customer availability.
    """

    def __init__(self, config):
        """
        Initializes the predictor with model configurations.
        """
        self.config = config
        rf_config = self.config['ml_pipeline']['random_forest']

        self.travel_model = RandomForestRegressor(
            n_estimators=rf_config['travel_model']['n_estimators'],
            max_depth=rf_config['travel_model']['max_depth'],
            random_state=rf_config['travel_model']['random_state']
        )
        self.availability_model = RandomForestClassifier(
            n_estimators=rf_config['availability_model']['n_estimators'],
            max_depth=rf_config['availability_model']['max_depth'],
            random_state=rf_config['availability_model']['random_state'],
            class_weight=rf_config['availability_model']['class_weight']
        )
        self.label_encoders = {}
        self.metrics = {}
        self.trained = False

        # Define feature lists here so they are part of the instance
        # regardless of whether it's trained or loaded.
        self.travel_features = [
            'distance_km', 'hour_of_day', 'day_of_week', 'weather_encoded', 'traffic_level', 'previous_stop_delay'
        ]
        self.availability_features = [
            'hour_of_day', 'day_of_week', 'weather_encoded'
            # Note: More features from README would require more complex data generation
        ]

    def _preprocess_data(self, df):
        """
        Preprocesses the raw historical data for feature engineering.
        """
        df['date'] = pd.to_datetime(df['date'])

        # Feature Engineering
        for col in ['weather']:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        return df

    def train_and_validate(self, data):
        """
        Trains and validates the models using a time-based split.
        - Training: Older data
        - Validation: More recent data
        """
        df = self._preprocess_data(data)

        # Temporal Split
        df_sorted = df.sort_values(by='date').reset_index(drop=True)
        split_point = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_point]
        val_df = df_sorted.iloc[split_point:]

        X_train_travel = train_df[self.travel_features]
        y_train_travel = train_df['travel_time_min']
        X_val_travel = val_df[self.travel_features]
        y_val_travel = val_df['travel_time_min']

        X_train_avail = train_df[self.availability_features]
        y_train_avail = train_df['customer_available']
        X_val_avail = val_df[self.availability_features]
        y_val_avail = val_df['customer_available']

        # Train Travel Model
        print("Training travel time prediction model...")
        self.travel_model.fit(X_train_travel, y_train_travel)

        # Train Availability Model
        print("Training customer availability prediction model...")
        self.availability_model.fit(X_train_avail, y_train_avail)

        self.trained = True
        print("Models trained successfully.")

        # Validation
        self._validate(X_val_travel, y_val_travel, X_val_avail, y_val_avail)

        return self.metrics

    def _validate(self, X_val_travel, y_val_travel, X_val_avail, y_val_avail):
        """Calculates and stores validation metrics."""
        print("\nValidating models...")
        # Travel model validation
        travel_preds = self.travel_model.predict(X_val_travel)
        self.metrics['travel_time_mae'] = mean_absolute_error(y_val_travel, travel_preds)
        self.metrics['travel_time_r2'] = r2_score(y_val_travel, travel_preds)

        # Availability model validation
        avail_preds_proba = self.availability_model.predict_proba(X_val_avail)[:, 1]
        avail_preds_class = self.availability_model.predict(X_val_avail)
        self.metrics['availability_auc'] = roc_auc_score(y_val_avail, avail_preds_proba)
        self.metrics['availability_accuracy'] = accuracy_score(y_val_avail, avail_preds_class)

        print("Validation complete. Metrics:")
        for key, value in self.metrics.items():
            print(f"- {key}: {value:.4f}")

    def predict(self, X_travel, X_avail):
        """
        Makes predictions on new data.

        Returns:
            dict: A dictionary containing 'travel_time' and 'availability_prob'.
        """
        if not self.trained:
            raise RuntimeError("Models must be trained before making predictions.")

        travel_time = self.travel_model.predict(X_travel)
        availability_prob = self.availability_model.predict_proba(X_avail)[:, 1]

        return {
            'travel_time': travel_time,
            'availability_prob': availability_prob
        }

    def save_models(self, path):
        """Saves the trained models to the specified path."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.travel_model, os.path.join(path, 'travel_model.pkl'))
        joblib.dump(self.availability_model, os.path.join(path, 'availability_model.pkl'))
        joblib.dump(self.label_encoders, os.path.join(path, 'label_encoders.pkl'))
        print(f"Models saved to {path}")

    def load_models(self, path):
        """Loads trained models from the specified path."""
        self.travel_model = joblib.load(os.path.join(path, 'travel_model.pkl'))
        self.availability_model = joblib.load(os.path.join(path, 'availability_model.pkl'))
        self.label_encoders = joblib.load(os.path.join(path, 'label_encoders.pkl'))
        self.trained = True
        print(f"Models loaded from {path}")


if __name__ == '__main__':
    import yaml

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'validation_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'historical_deliveries.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run data_generator.py first.")
    else:
        historical_data = pd.read_csv(data_path)

        # Initialize and run pipeline
        predictor = VRPPredictor(config)
        validation_metrics = predictor.train_and_validate(historical_data)

        # Save models
        model_save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
        predictor.save_models(model_save_path)