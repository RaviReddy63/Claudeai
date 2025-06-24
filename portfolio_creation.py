import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import List, Dict, Tuple, Optional
import itertools

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the haversine distance between two points on earth in miles.
    """
    try:
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in miles
        r = 3956
        return c * r
    except Exception as e:
        print(f"Error in haversine_distance: {e}")
        return float('inf')

def validate_customer_data(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate customer DataFrame and prepare for processing.
    Expected columns: ECN, LAT_NUM, LON_NUM, BILLINGSTREET, BILLINGCITY, BILLINGSTATE
    """
    print(f"Validating customer data...")
    print(f"Customer DataFrame shape: {customers_df.shape}")
    print(f"Customer columns: {list(customers_df.columns)}")
    
    customers = customers_df.copy()
    required_cols = ['ECN', 'LAT_NUM', 'LON_NUM', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE']
    
    # Validate required columns exist
    missing_cols = [col for col in required_cols if col not in customers.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing coordinates before dropping
    print(f"Customers with missing LAT_NUM: {customers['LAT_NUM'].isna().sum()}")
    print(f"Customers with missing LON_NUM: {customers['LON_NUM'].isna().sum()}")
    
    # Remove customers with missing coordinates
    customers = customers.dropna(subset=['LAT_NUM', 'LON_NUM'])
    print(f"Customers after removing missing coordinates: {len(customers)}")
    
    customers['assigned'] = False
    
    return customers

def validate_branch_data(branches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate branch DataFrame and prepare for processing.
    Expected columns: BRANCH_AU, BRANCH_LAT_NUM, BRANCH_LON_NUM
    """
    print(f"Validating branch data...")
    print(f"Branch DataFrame shape: {branches_df.shape}")
    print(f"Branch columns: {list(branches_df.columns)}")
    
    branches = branches_df.copy()
    required_cols = ['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
    
    # Validate required columns exist
    missing_cols = [col for col in required_cols if col not in branches.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing coordinates before dropping
    print(f"Branches with missing BRANCH_LAT_NUM: {branches['BRANCH_LAT_NUM'].isna().sum()}")
    print(f"Branches with missing BRANCH_LON_NUM: {branches['BRANCH_LON_NUM'].isna().sum()}")
    
    # Remove branches with missing coordinates
    branches = branches.dropna(subset=['BRANCH_LAT_NUM', 'BRANCH_LON_NUM'])
    print(f"Branches after removing missing coordinates: {len(branches)}")
    
    return branches

def find_customers_near_branch(customers: pd.DataFrame, branch_lat: float, branch_lon: float, 
                              radius_miles: float = 20) -> List[int]:
    """
    Find customers within specified radius of a branch.
    Returns list of customer indices.
    """
    nearby_customers = []
    
    try:
        # Debug: Check if customers DataFrame is empty
        if customers.empty:
            print("Warning: customers DataFrame is empty")
            return nearby_customers
            
        # Debug: Check if 'assigned' column exists
        if 'assigned' not in customers.columns:
            print("Warning: 'assigned' column not found in customers DataFrame")
            return nearby_customers
        
        for idx, customer in customers.iterrows():
            if not customer['assigned']:  # Only consider unassigned customers
                try:
                    distance = haversine_distance(
                        customer['LAT_NUM'], customer['LON_NUM'],
                        branch_lat, branch_lon
                    )
                    if distance <= radius_miles:
                        nearby_customers.append(idx)
                except Exception as e:
                    print(f"Error processing customer {idx}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error in find_customers_near_branch: {e}")
    
    return nearby_customers

def create_in_market_portfolios(customers: pd.DataFrame, branches: pd.DataFrame, 
                               min_portfolio_size: int = 220, max_portfolio_size: int = 250,
                               radius_miles: float = 20) -> Dict:
    """
    Create in-market portfolios for branches that have sufficient customers nearby.
    """
    print(f"Creating in-market portfolios...")
    portfolios = {}
    customers_copy = customers.copy()
    
    # Debug: Check DataFrame states
    print(f"Customers copy shape: {customers_copy.shape}")
    print(f"Branches shape: {branches.shape}")
    
    # Calculate potential portfolio size for each branch
    branch_potential = []
    
    try:
        for idx, branch in branches.iterrows():
            print(f"Processing branch {idx}: {branch['BRANCH_AU']}")
            nearby_customers = find_customers_near_branch(
                customers_copy, branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM'], radius_miles
            )
            branch_potential.append({
                'branch_idx': idx,
                'branch_id': branch['BRANCH_AU'],
                'potential_customers': len(nearby_customers),
                'customer_indices': nearby_customers
            })
            print(f"Branch {branch['BRANCH_AU']} has {len(nearby_customers)} potential customers")
    
    except Exception as e:
        print(f"Error calculating branch potential: {e}")
        return portfolios, customers_copy
    
    # Sort branches by potential customer count (descending) to prioritize high-density areas
    branch_potential.sort(key=lambda x: x['potential_customers'], reverse=True)
    
    portfolio_id = 1
    
    try:
        for branch_info in branch_potential:
            if branch_info['potential_customers'] >= min_portfolio_size:
                # Safely get branch row
                try:
                    branch_row = branches.loc[branch_info['branch_idx']]
                except KeyError:
                    print(f"Warning: Branch index {branch_info['branch_idx']} not found in branches DataFrame")
                    continue
                
                # Get current nearby customers (some may have been assigned already)
                nearby_customers = find_customers_near_branch(
                    customers_copy, branch_row['BRANCH_LAT_NUM'], branch_row['BRANCH_LON_NUM'], radius_miles
                )
                
                if len(nearby_customers) >= min_portfolio_size:
                    # Take up to max_portfolio_size customers
                    selected_customers = nearby_customers[:max_portfolio_size]
                    
                    # Debug: Check if selected customers exist in DataFrame
                    valid_customers = [idx for idx in selected_customers if idx in customers_copy.index]
                    if len(valid_customers) != len(selected_customers):
                        print(f"Warning: Some selected customers not found in DataFrame")
                        selected_customers = valid_customers
                    
                    if not selected_customers:
                        continue
                    
                    # Mark customers as assigned
                    customers_copy.loc[selected_customers, 'assigned'] = True
                    
                    # Create portfolio
                    portfolios[f'IN_MARKET_{portfolio_id}'] = {
                        'type': 'IN_MARKET',
                        'branch_id': branch_info['branch_id'],
                        'branch_lat': branch_row['BRANCH_LAT_NUM'],
                        'branch_lon': branch_row['BRANCH_LON_NUM'],
                        'customer_count': len(selected_customers),
                        'customer_indices': selected_customers,
                        'customers': customers_copy.loc[selected_customers].copy()
                    }
                    
                    print(f"Created IN_MARKET_{portfolio_id} with {len(selected_customers)} customers")
                    portfolio_id += 1
    
    except Exception as e:
        print(f"Error creating in-market portfolios: {e}")
    
    return portfolios, customers_copy

def create_centralized_portfolios(unassigned_customers: pd.DataFrame, 
                                 min_portfolio_size: int = 220, max_portfolio_size: int = 250) -> Dict:
    """
    Create centralized portfolios from remaining unassigned customers.
    """
    print(f"Creating centralized portfolios...")
    portfolios = {}
    
    try:
        # Get unassigned customers
        remaining = unassigned_customers[~unassigned_customers['assigned']].copy()
        remaining_indices = remaining.index.tolist()
        
        print(f"Remaining unassigned customers: {len(remaining_indices)}")
        
        portfolio_id = 1
        
        # Create portfolios of optimal size
        while len(remaining_indices) >= min_portfolio_size:
            # Take up to max_portfolio_size customers
            portfolio_size = min(max_portfolio_size, len(remaining_indices))
            selected_indices = remaining_indices[:portfolio_size]
            
            # Validate selected indices exist in DataFrame
            valid_indices = [idx for idx in selected_indices if idx in remaining.index]
            if len(valid_indices) != len(selected_indices):
                print(f"Warning: Some selected indices not found in remaining DataFrame")
                selected_indices = valid_indices
            
            if not selected_indices:
                break
            
            portfolios[f'CENTRALIZED_{portfolio_id}'] = {
                'type': 'CENTRALIZED',
                'branch_id': None,
                'branch_lat': None,
                'branch_lon': None,
                'customer_count': len(selected_indices),
                'customer_indices': selected_indices,
                'customers': remaining.loc[selected_indices].copy()
            }
            
            print(f"Created CENTRALIZED_{portfolio_id} with {len(selected_indices)} customers")
            
            # Remove assigned customers from remaining list
            remaining_indices = remaining_indices[portfolio_size:]
            portfolio_id += 1
        
        # Handle remaining customers (less than min_portfolio_size)
        if remaining_indices:
            print(f"Handling {len(remaining_indices)} remaining customers")
            # Try to distribute among existing centralized portfolios
            centralized_portfolios = [k for k in portfolios.keys() if k.startswith('CENTRALIZED')]
            
            if centralized_portfolios:
                for customer_idx in remaining_indices:
                    if customer_idx not in remaining.index:
                        continue
                        
                    # Find a centralized portfolio with space
                    for portfolio_name in centralized_portfolios:
                        if portfolios[portfolio_name]['customer_count'] < max_portfolio_size:
                            portfolios[portfolio_name]['customer_indices'].append(customer_idx)
                            portfolios[portfolio_name]['customer_count'] += 1
                            new_customer = remaining.loc[[customer_idx]].copy()
                            portfolios[portfolio_name]['customers'] = pd.concat([
                                portfolios[portfolio_name]['customers'], new_customer
                            ])
                            break
            else:
                # Create a small centralized portfolio if no other option
                valid_remaining = [idx for idx in remaining_indices if idx in remaining.index]
                if valid_remaining:
                    portfolios[f'CENTRALIZED_{portfolio_id}'] = {
                        'type': 'CENTRALIZED',
                        'branch_id': None,
                        'branch_lat': None,
                        'branch_lon': None,
                        'customer_count': len(valid_remaining),
                        'customer_indices': valid_remaining,
                        'customers': remaining.loc[valid_remaining].copy()
                    }
                    print(f"Created small CENTRALIZED_{portfolio_id} with {len(valid_remaining)} customers")
    
    except Exception as e:
        print(f"Error creating centralized portfolios: {e}")
    
    return portfolios

def optimize_portfolios_from_dataframes(customers_df: pd.DataFrame, branches_df: pd.DataFrame, 
                                       min_portfolio_size: int = 220, max_portfolio_size: int = 250,
                                       radius_miles: float = 20) -> Dict:
    """
    Main function to optimize customer portfolios from pandas DataFrames.
    
    Args:
        customers_df: DataFrame with customer data
        branches_df: DataFrame with branch data
        min_portfolio_size: Minimum customers per portfolio
        max_portfolio_size: Maximum customers per portfolio
        radius_miles: Radius for in-market portfolios
    
    Returns:
        Dictionary containing all portfolios
    """
    try:
        # Validate data
        customers = validate_customer_data(customers_df)
        branches = validate_branch_data(branches_df)
        
        print(f"Loaded {len(customers)} customers and {len(branches)} branches")
        
        # Create in-market portfolios
        in_market_portfolios, customers_with_assignments = create_in_market_portfolios(
            customers, branches, min_portfolio_size, max_portfolio_size, radius_miles
        )
        
        print(f"Created {len(in_market_portfolios)} in-market portfolios")
        
        # Create centralized portfolios for remaining customers
        centralized_portfolios = create_centralized_portfolios(
            customers_with_assignments, min_portfolio_size, max_portfolio_size
        )
        
        print(f"Created {len(centralized_portfolios)} centralized portfolios")
        
        # Combine all portfolios
        all_portfolios = {**in_market_portfolios, **centralized_portfolios}
        
        return all_portfolios
        
    except Exception as e:
        print(f"Error in optimize_portfolios_from_dataframes: {e}")
        import traceback
        traceback.print_exc()
        return {}

def generate_portfolio_summary(portfolios: Dict) -> pd.DataFrame:
    """
    Generate a summary report of all portfolios.
    """
    summary_data = []
    
    for portfolio_name, portfolio_info in portfolios.items():
        summary_data.append({
            'Portfolio_ID': portfolio_name,
            'Type': portfolio_info['type'],
            'Branch_ID': portfolio_info['branch_id'],
            'Customer_Count': portfolio_info['customer_count'],
            'Branch_Lat': portfolio_info['branch_lat'],
            'Branch_Lon': portfolio_info['branch_lon']
        })
    
    return pd.DataFrame(summary_data)

def export_portfolios_to_csv(portfolios: Dict, output_dir: str = './'):
    """
    Export each portfolio's customer data to separate CSV files.
    """
    import os
    
    for portfolio_name, portfolio_info in portfolios.items():
        try:
            filename = f"{output_dir}/{portfolio_name}_customers.csv"
            portfolio_info['customers'].to_csv(filename, index=False)
            print(f"Exported {portfolio_name} to {filename}")
        except Exception as e:
            print(f"Error exporting {portfolio_name}: {e}")

# Debugging helper function
def debug_dataframes(customers_df, branches_df):
    """
    Debug function to check DataFrame structure and content.
    """
    print("=== DEBUGGING DataFrames ===")
    print(f"Customers DataFrame:")
    print(f"  Shape: {customers_df.shape}")
    print(f"  Columns: {list(customers_df.columns)}")
    print(f"  Index range: {customers_df.index.min()} to {customers_df.index.max()}")
    print(f"  Sample data:")
    print(customers_df.head())
    
    print(f"\nBranches DataFrame:")
    print(f"  Shape: {branches_df.shape}")
    print(f"  Columns: {list(branches_df.columns)}")
    print(f"  Index range: {branches_df.index.min()} to {branches_df.index.max()}")
    print(f"  Sample data:")
    print(branches_df.head())
    print("=== END DEBUG ===\n")

# Example usage with debugging
if __name__ == "__main__":
    # Example usage with DataFrames
    try:
        # Add debugging before running the optimization
        # debug_dataframes(customers_df, branches_df)
        
        # Optimize portfolios from DataFrames
        portfolios = optimize_portfolios_from_dataframes(
            customers_df=customers_df,  # Your customer DataFrame
            branches_df=branches_df,    # Your branch DataFrame
            min_portfolio_size=220,
            max_portfolio_size=250,
            radius_miles=20
        )
        
        if portfolios:
            # Generate summary
            summary = generate_portfolio_summary(portfolios)
            print("\nPortfolio Summary:")
            print(summary.to_string(index=False))
            
            # Export to CSV files (optional)
            export_portfolios_to_csv(portfolios)
            summary.to_csv('portfolio_summary.csv', index=False)
            
            # Print statistics
            total_customers = sum(p['customer_count'] for p in portfolios.values())
            in_market_count = len([p for p in portfolios.values() if p['type'] == 'IN_MARKET'])
            centralized_count = len([p for p in portfolios.values() if p['type'] == 'CENTRALIZED'])
            
            print(f"\nFinal Statistics:")
            print(f"Total Portfolios: {len(portfolios)}")
            print(f"In-Market Portfolios: {in_market_count}")
            print(f"Centralized Portfolios: {centralized_count}")
            print(f"Total Customers Assigned: {total_customers}")
        else:
            print("No portfolios were created due to errors.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure your DataFrames have the correct column names and format.")
