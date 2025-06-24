import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import List, Dict, Tuple, Optional
import itertools

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the haversine distance between two points on earth in miles.
    """
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

def validate_customer_data(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate customer DataFrame and prepare for processing.
    Expected columns: ECN, LAT_NUM, LON_NUM, BILLINGSTREET, BILLINGCITY, BILLINGSTATE
    """
    customers = customers_df.copy()
    required_cols = ['ECN', 'LAT_NUM', 'LON_NUM', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE']
    
    # Validate required columns exist
    missing_cols = [col for col in required_cols if col not in customers.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove customers with missing coordinates
    customers = customers.dropna(subset=['LAT_NUM', 'LON_NUM'])
    customers['assigned'] = False
    
    return customers

def validate_branch_data(branches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate branch DataFrame and prepare for processing.
    Expected columns: BRANCH_AU, BRANCH_LAT_NUM, BRANCH_LON_NUM
    """
    branches = branches_df.copy()
    required_cols = ['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
    
    # Validate required columns exist
    missing_cols = [col for col in required_cols if col not in branches.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove branches with missing coordinates
    branches = branches.dropna(subset=['BRANCH_LAT_NUM', 'BRANCH_LON_NUM'])
    
    return branches

def find_customers_near_branch(customers: pd.DataFrame, branch_lat: float, branch_lon: float, 
                              radius_miles: float = 20) -> List[int]:
    """
    Find customers within specified radius of a branch.
    Returns list of customer indices.
    """
    nearby_customers = []
    
    for idx, customer in customers.iterrows():
        if not customer['assigned']:  # Only consider unassigned customers
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                branch_lat, branch_lon
            )
            if distance <= radius_miles:
                nearby_customers.append(idx)
    
    return nearby_customers

def create_in_market_portfolios(customers: pd.DataFrame, branches: pd.DataFrame, 
                               min_portfolio_size: int = 220, max_portfolio_size: int = 250,
                               radius_miles: float = 20) -> Dict:
    """
    Create in-market portfolios for branches that have sufficient customers nearby.
    """
    portfolios = {}
    customers_copy = customers.copy()
    
    # Calculate potential portfolio size for each branch
    branch_potential = []
    for idx, branch in branches.iterrows():
        nearby_customers = find_customers_near_branch(
            customers_copy, branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM'], radius_miles
        )
        branch_potential.append({
            'branch_idx': idx,
            'branch_id': branch['BRANCH_AU'],
            'potential_customers': len(nearby_customers),
            'customer_indices': nearby_customers
        })
    
    # Sort branches by potential customer count (descending) to prioritize high-density areas
    branch_potential.sort(key=lambda x: x['potential_customers'], reverse=True)
    
    portfolio_id = 1
    
    for branch_info in branch_potential:
        if branch_info['potential_customers'] >= min_portfolio_size:
            # Get current nearby customers (some may have been assigned already)
            branch_row = branches.iloc[branch_info['branch_idx']]
            nearby_customers = find_customers_near_branch(
                customers_copy, branch_row['BRANCH_LAT_NUM'], branch_row['BRANCH_LON_NUM'], radius_miles
            )
            
            if len(nearby_customers) >= min_portfolio_size:
                # Take up to max_portfolio_size customers
                selected_customers = nearby_customers[:max_portfolio_size]
                
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
                
                portfolio_id += 1
    
    return portfolios, customers_copy

def create_centralized_portfolios(unassigned_customers: pd.DataFrame, 
                                 min_portfolio_size: int = 220, max_portfolio_size: int = 250) -> Dict:
    """
    Create centralized portfolios from remaining unassigned customers.
    """
    portfolios = {}
    
    # Get unassigned customers
    remaining = unassigned_customers[~unassigned_customers['assigned']].copy()
    remaining_indices = remaining.index.tolist()
    
    portfolio_id = 1
    
    # Create portfolios of optimal size
    while len(remaining_indices) >= min_portfolio_size:
        # Take up to max_portfolio_size customers
        portfolio_size = min(max_portfolio_size, len(remaining_indices))
        selected_indices = remaining_indices[:portfolio_size]
        
        portfolios[f'CENTRALIZED_{portfolio_id}'] = {
            'type': 'CENTRALIZED',
            'branch_id': None,
            'branch_lat': None,
            'branch_lon': None,
            'customer_count': len(selected_indices),
            'customer_indices': selected_indices,
            'customers': remaining.loc[selected_indices].copy()
        }
        
        # Remove assigned customers from remaining list
        remaining_indices = remaining_indices[portfolio_size:]
        portfolio_id += 1
    
    # Handle remaining customers (less than min_portfolio_size)
    if remaining_indices:
        # Try to distribute among existing centralized portfolios
        centralized_portfolios = [k for k in portfolios.keys() if k.startswith('CENTRALIZED')]
        
        if centralized_portfolios:
            for customer_idx in remaining_indices:
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
            portfolios[f'CENTRALIZED_{portfolio_id}'] = {
                'type': 'CENTRALIZED',
                'branch_id': None,
                'branch_lat': None,
                'branch_lon': None,
                'customer_count': len(remaining_indices),
                'customer_indices': remaining_indices,
                'customers': remaining.loc[remaining_indices].copy()
            }
    
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
        filename = f"{output_dir}/{portfolio_name}_customers.csv"
        portfolio_info['customers'].to_csv(filename, index=False)
        print(f"Exported {portfolio_name} to {filename}")

# Example usage
if __name__ == "__main__":
    # Example usage with DataFrames
    try:
        # Assuming you have your DataFrames ready
        # customers_df = your_customer_dataframe
        # branches_df = your_branch_dataframe
        
        # Optimize portfolios from DataFrames
        portfolios = optimize_portfolios_from_dataframes(
            customers_df=customers_df,  # Your customer DataFrame
            branches_df=branches_df,    # Your branch DataFrame
            min_portfolio_size=220,
            max_portfolio_size=250,
            radius_miles=20
        )
        
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
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure your DataFrames have the correct column names and format.")
