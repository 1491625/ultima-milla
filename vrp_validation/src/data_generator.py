import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """
    Generates a realistic synthetic dataset for VRP validation.
    It creates customers, vehicles, and historical delivery data.
    """

    def __init__(self, config):
        """
        Initializes the generator with configuration parameters.

        Args:
            config (dict): A dictionary with configuration details.
        """
        self.config = config
        np.random.seed(42)

    def generate_customers(self):
        """Generates the customers.csv file."""
        n_customers = self.config['dataset']['n_customers']
        grid_size = self.config['dataset']['grid_size_km']

        customers_data = {
            'customer_id': range(1, n_customers + 1),
            'lat': np.random.uniform(0, grid_size, n_customers),
            'lon': np.random.uniform(0, grid_size, n_customers),
            'demand_kg': np.random.uniform(*self.config['dataset']['demand_range_kg'], n_customers),
            'service_time_min': np.random.uniform(*self.config['dataset']['service_time_range_min'], n_customers),
            'availability_base_prob': np.random.uniform(*self.config['dataset']['availability_base_prob_range'], n_customers),
            'customer_type': np.random.choice(['residential', 'business', 'priority'], n_customers, p=[0.6, 0.3, 0.1]),
            'access_difficulty': np.random.uniform(1.0, 1.5, n_customers)
        }

        # Generate time windows
        start_hours = np.random.randint(8, 18, n_customers)
        window_durations = np.random.uniform(*self.config['dataset']['time_window_range_hours'], n_customers)
        customers_data['time_window_start'] = start_hours
        customers_data['time_window_end'] = np.round(start_hours + window_durations).astype(int)
        # Ensure end is within working hours (e.g., before 20:00)
        customers_data['time_window_end'] = np.minimum(customers_data['time_window_end'], 20)

        customers_df = pd.DataFrame(customers_data)
        return customers_df

    def generate_vehicles(self):
        """Generates the vehicles.csv file."""
        n_vehicles = self.config['dataset']['n_vehicles']
        grid_size = self.config['dataset']['grid_size_km']

        vehicles_data = {
            'vehicle_id': range(1, n_vehicles + 1),
            'capacity_kg': np.random.uniform(200, 500, n_vehicles),
            'cost_per_km': np.random.uniform(0.8, 1.2, n_vehicles),
            'max_working_hours': np.random.choice([8, 10], n_vehicles),
            'depot_lat': np.full(n_vehicles, grid_size / 2),
            'depot_lon': np.full(n_vehicles, grid_size / 2)
        }
        vehicles_df = pd.DataFrame(vehicles_data)
        return vehicles_df

    def generate_historical_deliveries(self, customers_df):
        """
        Generates the historical_deliveries.csv file for ML training.

        Args:
            customers_df (pd.DataFrame): The dataframe of generated customers.
        """
        n_days = self.config['dataset']['n_days'] * 6 # More data for training
        n_customers = self.config['dataset']['n_customers']

        records = []
        delivery_id = 0
        start_date = datetime(2023, 1, 1)

        for day in range(n_days):
            date = start_date + timedelta(days=day)
            day_of_week = date.weekday()

            # Simulate a number of deliveries for the day
            n_deliveries_today = int(n_customers * np.random.uniform(0.7, 1.3))

            for _ in range(n_deliveries_today):
                customer = customers_df.sample(1).iloc[0]
                hour_of_day = np.random.randint(8, 19)

                # Simulate weather and traffic
                weather = np.random.choice(['sunny', 'rain', 'cloudy'], p=[0.7, 0.1, 0.2])
                traffic_level = 1.0 + (0.5 * (hour_of_day > 16 or hour_of_day < 10)) + np.random.uniform(-0.2, 0.2)
                if weather == 'rain':
                    traffic_level *= 1.3

                # Simulate delivery details
                distance_km = np.sqrt(customer['lat']**2 + customer['lon']**2) * np.random.uniform(0.8, 1.2) # Simplified
                base_travel_time = distance_km * 2.5 # 2.5 min/km avg
                travel_time_min = base_travel_time * traffic_level * np.random.uniform(0.8, 1.2) # with noise

                planned_arrival = date.replace(hour=hour_of_day, minute=np.random.randint(0,60))
                previous_stop_delay = np.random.uniform(-10, 30) if np.random.rand() > 0.3 else 0
                actual_arrival = planned_arrival + timedelta(minutes=int(travel_time_min - base_travel_time + previous_stop_delay))

                # Simulate customer availability
                prob = customer['availability_base_prob']
                if hour_of_day < customer['time_window_start'] or hour_of_day > customer['time_window_end']:
                    prob *= 0.5
                customer_available = np.random.rand() < prob

                records.append({
                    'delivery_id': delivery_id,
                    'customer_id': customer['customer_id'],
                    'date': date.strftime('%Y-%m-%d'),
                    'planned_arrival': planned_arrival.strftime('%Y-%m-%d %H:%M:%S'),
                    'actual_arrival': actual_arrival.strftime('%Y-%m-%d %H:%M:%S'),
                    'travel_time_min': travel_time_min,
                    'customer_available': customer_available,
                    'weather': weather,
                    'traffic_level': round(traffic_level, 2),
                    'day_of_week': day_of_week,
                    'hour_of_day': hour_of_day,
                    'distance_km': round(distance_km, 2),
                    'previous_stop_delay': round(previous_stop_delay, 2)
                })
                delivery_id += 1

        deliveries_df = pd.DataFrame(records)
        return deliveries_df

    def generate_all(self, output_dir):
        """
        Generates all synthetic data files and saves them to disk.

        Args:
            output_dir (str): The directory to save the CSV files.
        """
        customers = self.generate_customers()
        vehicles = self.generate_vehicles()
        historical_deliveries = self.generate_historical_deliveries(customers)

        # Save to CSV
        customers.to_csv(f"{output_dir}/customers.csv", index=False)
        vehicles.to_csv(f"{output_dir}/vehicles.csv", index=False)
        historical_deliveries.to_csv(f"{output_dir}/historical_deliveries.csv", index=False)

        print(f"Generated {len(customers)} customers.")
        print(f"Generated {len(vehicles)} vehicles.")
        print(f"Generated {len(historical_deliveries)} historical deliveries.")


if __name__ == '__main__':
    import yaml
    import os

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'validation_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Define output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    generator = SyntheticDataGenerator(config)
    generator.generate_all(output_dir)
    print(f"\nSynthetic data saved to {output_dir}")