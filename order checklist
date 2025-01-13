import pandas as pd
import datetime

# Mount Google Drive (for Colab)
from google.colab import drive
drive.mount('/content/drive')

# Load data from CSV files with proper error handling
def load_csv(file_path, file_name):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_name}")
        return df
    except FileNotFoundError:
        print(f"Error: {file_name} not found at {file_path}. Please check the path and try again.")
        return None

# File paths
orders_path = '/content/Orders.csv'
inventory_path = '/content/inventory.csv'
transport_path = '/content/transport.csv'
production_path = '/content/production.csv'
compliance_path = '/content/compliance.csv'

# Load files
orders_df = load_csv(orders_path, "Orders.csv")
inventory_df = load_csv(inventory_path, "inventory.csv")
transport_df = load_csv(transport_path, "transport.csv")
production_df = load_csv(production_path, "production.csv")
compliance_df = load_csv(compliance_path, "compliance.csv")



# Proceed only if all files are loaded successfully and are not empty
def validate_dataframes(dfs, df_names):
    for df, name in zip(dfs, df_names):
        if df is None:
            print(f"Error: {name} is None. Please check the file.")
            return False
        if df.empty:
            print(f"Error: {name} is empty. Please check the file content.")
            return False
    return True

# Validate DataFrames
dfs = [orders_df, inventory_df, transport_df, production_df, compliance_df]
df_names = ["Orders.csv", "inventory.csv", "transport.csv", "production.csv", "compliance.csv"]

if validate_dataframes(dfs, df_names):
    # Function to check stock availability
    def check_inventory(order):
        product = order['product']
        required_quantity = order['quantity']
        available_stock = inventory_df[inventory_df['product'] == product]['stock_level'].values[0]
        if available_stock >= required_quantity:
            return f"Stock Available: {available_stock}, meets the required quantity of {required_quantity}"
        else:
            return f"Stock Available: {available_stock}, insufficient stock for required quantity of {required_quantity}"

    # Function to evaluate transportation needs
    def evaluate_transport(order):
        product = order['product']
        quantity = order['quantity']
        available_transport = transport_df[transport_df['availability'] == 'Available']
        suitable_vehicle = available_transport[available_transport['capacity'] >= quantity]
        if not suitable_vehicle.empty:
            vehicle = suitable_vehicle.iloc[0]  # Take the first suitable vehicle
            return f"Available Vehicle: {vehicle['vehicle']} with capacity: {vehicle['capacity']}"
        else:
            return "No suitable transport available"

    # Function to align production with order
    def align_production(order):
        product = order['product']
        deadline = pd.to_datetime(order['deadline'])
        production_schedule = production_df[production_df['product'] == product]
        if production_schedule.empty:
            return "No production schedule available"
        finish_time = pd.to_datetime(production_schedule['finish_time'].values[0])
        if finish_time <= deadline:
            return f"Production scheduled to finish by {finish_time.strftime('%Y-%m-%d')}, on time"
        else:
            return f"Production scheduled to finish by {finish_time.strftime('%Y-%m-%d')}, past deadline"

    # Function to check compliance
    def check_compliance(order):
        shipment_id = order['shipment_id']
        compliance_status = compliance_df[compliance_df['shipment'] == shipment_id]['compliance_status'].values[0]
        return f"Compliance Status: {compliance_status}"

    # Processing orders with intermediate steps
    def process_orders(orders_df):
        print("Checking inventory availability...")
        orders_df['inventory_status'] = orders_df.apply(check_inventory, axis=1)
        print(orders_df[['order_id', 'inventory_status']])

        print("\nEvaluating transport needs...")
        orders_df['transport_status'] = orders_df.apply(evaluate_transport, axis=1)
        print(orders_df[['order_id', 'transport_status']])

        print("\nAligning production schedules...")
        orders_df['production_status'] = orders_df.apply(align_production, axis=1)
        print(orders_df[['order_id', 'production_status']])

        print("\nChecking compliance...")
        orders_df['compliance_status'] = orders_df.apply(check_compliance, axis=1)
        print(orders_df[['order_id', 'compliance_status']])

        print("\nCompiling checklist status...")
        orders_df['checklist_status'] = orders_df.apply(
            lambda row: all([
                isinstance(row['inventory_status'], str) and "insufficient" not in row['inventory_status'],
                isinstance(row['transport_status'], str) and "No suitable" not in row['transport_status'],
                isinstance(row['production_status'], str) and "past deadline" not in row['production_status'],
                isinstance(row['compliance_status'], str) and "Non-Compliant" not in row['compliance_status']
            ]),
            axis=1
        )
        print(orders_df[['order_id', 'checklist_status']])

        return orders_df

    # Run the checklist
    processed_orders = process_orders(orders_df)

    # Save the results
    processed_orders.to_csv('/content/processed_orders.csv', index=False)
    print("\nOrder checklist processing complete. Results saved to 'processed_orders.csv'.")
else:
    print("One or more DataFrames are invalid. Please address the errors and try again.")
