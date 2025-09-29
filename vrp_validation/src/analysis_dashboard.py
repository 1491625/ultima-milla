import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import yaml
import joblib

# Add src to path to allow imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ml_pipeline import VRPPredictor
from optimization_model import VRPOptimizer

class ValidationDashboard:
    """
    Generates a validation report with metrics and visualizations
    for the VRP ML and optimization pipeline.
    """

    def __init__(self, config, ml_metrics, opt_metrics, ml_predictor=None, customers=None, solution_routes=None):
        self.config = config
        self.ml_metrics = ml_metrics
        self.opt_metrics = opt_metrics
        self.ml_predictor = ml_predictor
        self.customers = customers
        self.solution_routes = solution_routes
        # Construct path relative to the project root (vrp_validation)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.output_dir = os.path.join(project_root, self.config['dashboard']['reports']['output_dir'])
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_ml_performance(self):
        """Visualizes ML model performance metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("Machine Learning Performance", fontsize=16)

        # Travel Time R^2 Score
        r2_score = self.ml_metrics.get('travel_time_r2', 0)
        axes[0].barh(['R^2 Score'], [r2_score], color='skyblue')
        axes[0].set_title('Travel Time Prediction (R^2)')
        axes[0].set_xlim(0, 1)
        axes[0].axvline(self.config['dashboard']['benchmarks']['travel_time_r2_min'], color='red', linestyle='--', label=f"Min Target ({self.config['dashboard']['benchmarks']['travel_time_r2_min']})")
        axes[0].text(r2_score + 0.02, 0, f"{r2_score:.3f}", va='center')
        axes[0].legend()

        # Availability AUC Score
        auc_score = self.ml_metrics.get('availability_auc', 0)
        axes[1].barh(['AUC Score'], [auc_score], color='salmon')
        axes[1].set_title('Availability Prediction (AUC)')
        axes[1].set_xlim(0, 1)
        axes[1].axvline(self.config['dashboard']['benchmarks']['availability_auc_min'], color='red', linestyle='--', label=f"Min Target ({self.config['dashboard']['benchmarks']['availability_auc_min']})")
        axes[1].text(auc_score + 0.02, 0, f"{auc_score:.3f}", va='center')
        axes[1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.output_dir, "ml_performance.png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_feature_importance(self):
        """Visualizes feature importances for the ML models."""
        if not self.ml_predictor:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Feature Importances", fontsize=16)

        # Travel model
        importances = self.ml_predictor.travel_model.feature_importances_
        indices = np.argsort(importances)
        axes[0].barh(range(len(indices)), importances[indices], align='center')
        axes[0].set_yticks(range(len(indices)))
        axes[0].set_yticklabels(np.array(self.ml_predictor.travel_features)[indices])
        axes[0].set_title('Travel Time Model')

        # Availability model
        importances = self.ml_predictor.availability_model.feature_importances_
        indices = np.argsort(importances)
        axes[1].barh(range(len(indices)), importances[indices], align='center')
        axes[1].set_yticks(range(len(indices)))
        axes[1].set_yticklabels(np.array(self.ml_predictor.availability_features)[indices])
        axes[1].set_title('Availability Model')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_route_visualization(self):
        """Visualizes the optimized routes on a 2D grid."""
        if self.opt_metrics.get('solver_status') != 'Optimal' or not self.solution_routes or not self.customers is not None:
            return None

        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='lon', y='lat', data=self.customers, hue='customer_id', palette='viridis', s=100, legend=None)
        plt.scatter(self.config['dataset']['grid_size_km']/2, self.config['dataset']['grid_size_km']/2, c='red', marker='s', s=200, label='Depot')

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.solution_routes)))
        for (k, route), color in zip(self.solution_routes.items(), colors):
            if len(route) <= 2: continue

            route_coords = []
            for node_id in route:
                if node_id == 0:
                    route_coords.append((self.config['dataset']['grid_size_km']/2, self.config['dataset']['grid_size_km']/2))
                else:
                    cust = self.customers[self.customers['customer_id'] == node_id]
                    route_coords.append((cust['lon'].iloc[0], cust['lat'].iloc[0]))

            route_coords = np.array(route_coords)
            plt.plot(route_coords[:, 0], route_coords[:, 1], color=color, label=f'Vehicle {k}')

        plt.title("Optimized Routes")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.output_dir, "route_visualization.png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    def generate_full_report(self):
        """Generates a text summary and all visualizations."""
        report_lines = ["# VRP Validation Report\n"]

        # --- Executive Summary ---
        report_lines.append("## 1. Executive Summary")
        status = self.opt_metrics.get('solver_status', 'UNKNOWN')
        if status == 'Optimal':
            summary = "SUCCESS: The pipeline ran successfully, and the optimizer found an optimal solution."
        elif status == 'Infeasible':
            summary = "WARNING: The pipeline ran successfully, but the optimization problem was INFEASIBLE. This suggests conflicting constraints (e.g., time windows, capacity)."
        else:
            summary = "ERROR: The pipeline may have issues, as the optimizer did not find an optimal solution."
        report_lines.append(summary)

        r2 = self.ml_metrics.get('travel_time_r2', 0)
        if r2 > 0.9:
            report_lines.append("- High R^2 score for travel time prediction (>0.9) suggests the synthetic data might be too predictable.")

        auc = self.ml_metrics.get('availability_auc', 0)
        if auc < self.config['dashboard']['benchmarks']['availability_auc_min']:
            report_lines.append(f"- Low AUC score for availability prediction ({auc:.3f}) indicates the model is not learning effectively. This is a RED FLAG.")

        # --- ML Performance ---
        report_lines.append("\n## 2. ML Performance Analysis")
        for key, val in self.ml_metrics.items():
            report_lines.append(f"- {key}: {val:.4f}")
        ml_plot = self.plot_ml_performance()
        report_lines.append(f"![ML Performance]({os.path.basename(ml_plot)})")

        fi_plot = self.plot_feature_importance()
        if fi_plot:
            report_lines.append(f"![Feature Importance]({os.path.basename(fi_plot)})")

        # --- Optimization Performance ---
        report_lines.append("\n## 3. Optimization Performance Analysis")
        for key, val in self.opt_metrics.items():
            report_lines.append(f"- {key}: {val}")

        route_plot = self.plot_route_visualization()
        if route_plot:
             report_lines.append(f"![Route Map]({os.path.basename(route_plot)})")

        # --- Recommendations ---
        report_lines.append("\n## 4. Recommendations")
        report_lines.append("- **Proceed with Caution**. The pipeline is technically functional, but the results show significant weaknesses.")
        report_lines.append("- **Revisit Data Generation**: The availability model's failure and the travel model's high performance suggest the synthetic data needs more complexity and noise.")
        report_lines.append("- **Feature Engineering**: More complex features are needed for the availability model.")
        report_lines.append("- **Constraint Analysis**: The 'Infeasible' result requires analyzing which constraints are too tight.")

        # Save report
        report_path = os.path.join(self.output_dir, "validation_report.md")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))

        print(f"\nFull report generated at: {report_path}")
        return report_path

if __name__ == '__main__':
    # This block demonstrates how to run the dashboard
    print("Running dashboard generation...")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'validation_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Load all necessary data and models ---
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    customers_full = pd.read_csv(os.path.join(data_dir, 'customers.csv'))
    vehicles = pd.read_csv(os.path.join(data_dir, 'vehicles.csv'))
    historical_data = pd.read_csv(os.path.join(data_dir, 'historical_deliveries.csv'))

    # --- Run ML Pipeline ---
    predictor = VRPPredictor(config)
    ml_metrics = predictor.train_and_validate(historical_data)

    # --- Run Optimization ---
    # Use a subset of customers for the optimization run
    customers_opt = customers_full.head(10)
    optimizer = VRPOptimizer(config)
    optimizer.formulate_milp(customers_opt, vehicles, predictor)
    opt_metrics = optimizer.solve()

    solution_routes = None
    if opt_metrics['solver_status'] == 'Optimal':
        # This extraction logic is complex and duplicated from the optimization_model.py test block.
        # It should be refactored into the VRPOptimizer class for cleaner code.
        # For this demonstration, we'll keep it simple and assume it's available.
        # In a real scenario, optimizer.extract_solution() would be called here.
        pass # Not extracting routes here to keep demo simple

    # --- Generate Dashboard ---
    dashboard = ValidationDashboard(
        config=config,
        ml_metrics=ml_metrics,
        opt_metrics=opt_metrics,
        ml_predictor=predictor,
        customers=customers_opt,
        solution_routes=solution_routes # Pass None if no solution
    )
    dashboard.generate_full_report()
    print("Dashboard generation complete.")