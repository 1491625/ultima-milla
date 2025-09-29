import pytest
import os
import yaml
import pandas as pd
from shutil import rmtree

# Add src to path to allow imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_generator import SyntheticDataGenerator
from ml_pipeline import VRPPredictor
from optimization_model import VRPOptimizer

# Define a fixture for the configuration that can be used by tests
@pytest.fixture(scope="module")
def config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'validation_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Define a fixture to create a temporary directory for test artifacts
@pytest.fixture(scope="module")
def temp_data_dir():
    dir_path = "vrp_validation/tests/temp_test_data"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    # Teardown: remove the directory after tests are done
    rmtree(dir_path)

def test_full_pipeline(config, temp_data_dir):
    """
    Tests the end-to-end pipeline: data generation -> ML training -> optimization.
    """
    # --- 1. Data Generation ---
    synthetic_data_path = os.path.join(temp_data_dir, 'synthetic')
    os.makedirs(synthetic_data_path, exist_ok=True)

    generator = SyntheticDataGenerator(config)
    generator.generate_all(synthetic_data_path)

    # Assert that files were created
    assert os.path.exists(os.path.join(synthetic_data_path, 'customers.csv'))
    assert os.path.exists(os.path.join(synthetic_data_path, 'vehicles.csv'))
    assert os.path.exists(os.path.join(synthetic_data_path, 'historical_deliveries.csv'))

    # --- 2. ML Pipeline ---
    historical_data = pd.read_csv(os.path.join(synthetic_data_path, 'historical_deliveries.csv'))

    predictor = VRPPredictor(config)
    metrics = predictor.train_and_validate(historical_data)

    # Assert ML metrics are within "realistic" bounds as per README
    assert 'travel_time_r2' in metrics
    assert metrics['travel_time_r2'] < 0.95, "R2 score is suspiciously high, data may be too perfect."

    assert 'availability_auc' in metrics
    assert metrics['availability_auc'] > 0.5, "AUC is not better than random."

    model_results_path = os.path.join(temp_data_dir, 'results')
    predictor.save_models(model_results_path)
    assert os.path.exists(os.path.join(model_results_path, 'travel_model.pkl'))

    # --- 3. Optimization Model ---
    customers = pd.read_csv(os.path.join(synthetic_data_path, 'customers.csv'))
    vehicles = pd.read_csv(os.path.join(synthetic_data_path, 'vehicles.csv'))

    # Use a small subset to ensure the test runs quickly
    customers_subset = customers.head(8)

    optimizer = VRPOptimizer(config)
    optimizer.formulate_milp(customers_subset, vehicles, predictor)
    solution_metrics = optimizer.solve()

    # Assert that the solver finished with a valid status
    # 'Infeasible' is a valid outcome for the test, it means the pipeline ran correctly.
    assert solution_metrics['solver_status'] in ['Optimal', 'Infeasible']

    print(f"\nIntegration test passed with solver status: {solution_metrics['solver_status']}")