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
    customers['nearest_branch'] = None
    customers['distance_to_nearest'] = float('inf')
    
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

def assign_customers_to_nearest_branches(customers: pd.DataFrame, branches: pd.DataFrame, 
                                       radius_miles: float = 20) -> pd.DataFrame:
    """
    Assign each customer to their nearest branch within the specified radius.
    """
    print(f"Assigning customers to nearest branches within {radius_miles} miles...")
    
    customers_copy = customers.copy()
    
    # For each customer, find the nearest branch within radius
    for cust_idx, customer in customers_copy.iterrows():
        min_distance = float('inf')
        nearest_branch = None
        
        for branch_idx, branch in branches.iterrows():
            try:
                distance = haversine_distance(
                    customer['LAT_NUM'], customer['LON_NUM'],
                    branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                )
                
                # Only consider branches within radius
                if distance <= radius_miles and distance < min_distance:
                    min_distance = distance
                    nearest_branch = branch['BRANCH_AU']
                    
            except Exception as e:
                print(f"Error calculating distance for customer {cust_idx} to branch {branch_idx}: {e}")
                continue
        
        # Assign customer to nearest branch if found
        if nearest_branch is not None:
            customers_copy.loc[cust_idx, 'nearest_branch'] = nearest_branch
            customers_copy.loc[cust_idx, 'distance_to_nearest'] = min_distance
    
    # Count customers per branch
    branch_customer_counts = customers_copy[customers_copy['nearest_branch'].notna()]['nearest_branch'].value_counts()
    print(f"Customer distribution by nearest branch:")
    for branch_id, count in branch_customer_counts.items():
        print(f"  {branch_id}: {count} customers")
    
    return customers_copy

def create_optimized_in_market_portfolios(customers: pd.DataFrame, branches: pd.DataFrame, 
                                        min_portfolio_size: int = 220, max_portfolio_size: int = 250) -> Tuple[Dict, pd.DataFrame]:
    """
    Create in-market portfolios ensuring customers are assigned to their nearest branch.
    """
    print(f"Creating optimized in-market portfolios...")
    portfolios = {}
    customers_copy = customers.copy()
    
    # Get unique branches that have customers assigned to them
    branches_with_customers = customers_copy[customers_copy['nearest_branch'].notna()]['nearest_branch'].unique()
    
    portfolio_id = 1
    
    for branch_id in branches_with_customers:
        # Get all customers assigned to this branch
        branch_customers = customers_copy[
            (customers_copy['nearest_branch'] == branch_id) & 
            (~customers_copy['assigned'])
        ]
        
        if len(branch_customers) >= min_portfolio_size:
            # Sort customers by distance to branch (closest first)
            branch_customers_sorted = branch_customers.sort_values('distance_to_nearest')
            
            # Create portfolios for this branch
            remaining_customers = branch_customers_sorted.index.tolist()
            branch_portfolio_id = 1
            
            while len(remaining_customers) >= min_portfolio_size:
                # Take up to max_portfolio_size customers (closest ones first)
                portfolio_size = min(max_portfolio_size, len(remaining_customers))
                selected_customers = remaining_customers[:portfolio_size]
                
                # Get branch information
                branch_info = branches[branches['BRANCH_AU'] == branch_id].iloc[0]
                
                # Mark customers as assigned
                customers_copy.loc[selected_customers, 'assigned'] = True
                
                # Create portfolio
                portfolio_name = f'IN_MARKET_{branch_id}_{branch_portfolio_id}'
                portfolios[portfolio_name] = {
                    'type': 'IN_MARKET',
                    'branch_id': branch_id,
                    'branch_lat': branch_info['BRANCH_LAT_NUM'],
                    'branch_lon': branch_info['BRANCH_LON_NUM'],
                    'customer_count': len(selected_customers),
                    'customer_indices': selected_customers,
                    'customers': customers_copy.loc[selected_customers].copy(),
                    'avg_distance_to_branch': customers_copy.loc[selected_customers, 'distance_to_nearest'].mean()
                }
                
                print(f"Created {portfolio_name} with {len(selected_customers)} customers (avg distance: {portfolios[portfolio_name]['avg_distance_to_branch']:.2f} miles)")
                
                # Remove assigned customers from remaining list
                remaining_customers = remaining_customers[portfolio_size:]
                branch_portfolio_id += 1
            
            # Handle remaining customers for this branch (less than min_portfolio_size)
            if remaining_customers:
                print(f"Branch {branch_id} has {len(remaining_customers)} remaining customers (below minimum portfolio size)")
                # These will be handled in centralized portfolios or distributed to existing portfolios
        
        else:
            print(f"Branch {branch_id} has only {len(branch_customers)} customers (below minimum portfolio size)")
    
    return portfolios, customers_copy

def create_centralized_portfolios(unassigned_customers: pd.DataFrame, 
                                 min_portfolio_size: int = 220, max_portfolio_size: int = 250) -> Dict:
    """
    Create centralized portfolios from remaining unassigned customers.
    This includes customers not near any branch and small groups from branches with insufficient customers.
    """
    print(f"Creating centralized portfolios...")
    portfolios = {}
    
    try:
        # Get unassigned customers
        remaining = unassigned_customers[~unassigned_customers['assigned']].copy()
        remaining_indices = remaining.index.tolist()
        
        print(f"Remaining unassigned customers: {len(remaining_indices)}")
        
        # Separate customers by whether they have a nearest branch or not
        customers_with_branch = remaining[remaining['nearest_branch'].notna()]
        customers_without_branch = remaining[remaining['nearest_branch'].isna()]
        
        print(f"  - Customers with nearest branch (but in small groups): {len(customers_with_branch)}")
        print(f"  - Customers without nearby branch: {len(customers_without_branch)}")
        
        # Combine all remaining customers for centralized portfolios
        all_remaining = remaining.index.tolist()
        
        portfolio_id = 1
        
        # Create portfolios of optimal size
        while len(all_remaining) >= min_portfolio_size:
            # Take up to max_portfolio_size customers
            portfolio_size = min(max_portfolio_size, len(all_remaining))
            selected_indices = all_remaining[:portfolio_size]
            
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
                'customers': remaining.loc[selected_indices].copy(),
                'avg_distance_to_branch': None
            }
            
            print(f"Created CENTRALIZED_{portfolio_id} with {len(selected_indices)} customers")
            
            # Remove assigned customers from remaining list
            all_remaining = all_remaining[portfolio_size:]
            portfolio_id += 1
        
        # Handle remaining customers (less than min_portfolio_size)
        if all_remaining:
            print(f"Handling {len(all_remaining)} remaining customers")
            # Try to distribute among existing centralized portfolios
            centralized_portfolios = [k for k in portfolios.keys() if k.startswith('CENTRALIZED')]
            
            if centralized_portfolios:
                remaining_to_distribute = [idx for idx in all_remaining if idx in remaining.index]
                
                for i, customer_idx in enumerate(remaining_to_distribute):
                    # Find a centralized portfolio with space (round-robin distribution)
                    portfolio_name = centralized_portfolios[i % len(centralized_portfolios)]
                    
                    if portfolios[portfolio_name]['customer_count'] < max_portfolio_size:
                        portfolios[portfolio_name]['customer_indices'].append(customer_idx)
                        portfolios[portfolio_name]['customer_count'] += 1
                
                # Rebuild customer DataFrames for all modified portfolios
                for portfolio_name in centralized_portfolios:
                    all_indices = portfolios[portfolio_name]['customer_indices']
                    valid_indices = [idx for idx in all_indices if idx in unassigned_customers.index]
                    portfolios[portfolio_name]['customers'] = unassigned_customers.loc[valid_indices].copy()
                    portfolios[portfolio_name]['customer_count'] = len(valid_indices)
                    
                print(f"Distributed {len(remaining_to_distribute)} remaining customers across existing portfolios")
                    
            else:
                # Create a small centralized portfolio if no other option
                valid_remaining = [idx for idx in all_remaining if idx in remaining.index]
                if valid_remaining:
                    portfolios[f'CENTRALIZED_{portfolio_id}'] = {
                        'type': 'CENTRALIZED',
                        'branch_id': None,
                        'branch_lat': None,
                        'branch_lon': None,
                        'customer_count': len(valid_remaining),
                        'customer_indices': valid_remaining,
                        'customers': remaining.loc[valid_remaining].copy(),
                        'avg_distance_to_branch': None
                    }
                    print(f"Created small CENTRALIZED_{portfolio_id} with {len(valid_remaining)} customers")
    
    except Exception as e:
        print(f"Error creating centralized portfolios: {e}")
        import traceback
        traceback.print_exc()
    
    return portfolios

def optimize_portfolios_from_dataframes(customers_df: pd.DataFrame, branches_df: pd.DataFrame, 
                                       min_portfolio_size: int = 220, max_portfolio_size: int = 250,
                                       radius_miles: float = 20) -> Dict:
    """
    Main function to optimize customer portfolios ensuring customers are assigned to nearest AU.
    
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
        
        # Step 1: Assign customers to their nearest branches
        customers_with_assignments = assign_customers_to_nearest_branches(
            customers, branches, radius_miles
        )
        
        # Step 2: Create in-market portfolios based on nearest branch assignments
        in_market_portfolios, customers_after_in_market = create_optimized_in_market_portfolios(
            customers_with_assignments, branches, min_portfolio_size, max_portfolio_size
        )
        
        print(f"Created {len(in_market_portfolios)} in-market portfolios")
        
        # Step 3: Create centralized portfolios for remaining customers
        centralized_portfolios = create_centralized_portfolios(
            customers_after_in_market, min_portfolio_size, max_portfolio_size
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
            'Branch_Lon': portfolio_info['branch_lon'],
            'Avg_Distance_to_Branch': portfolio_info.get('avg_distance_to_branch', None)
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

def validate_nearest_branch_assignment(portfolios: Dict) -> None:
    """
    Validate that customers in in-market portfolios are indeed assigned to their nearest branch.
    """
    print("\n=== VALIDATION: Nearest Branch Assignment ===")
    
    for portfolio_name, portfolio_info in portfolios.items():
        if portfolio_info['type'] == 'IN_MARKET':
            customers_df = portfolio_info['customers']
            branch_id = portfolio_info['branch_id']
            
            # Check if all customers in this portfolio have this branch as their nearest
            customers_with_different_nearest = customers_df[
                customers_df['nearest_branch'] != branch_id
            ]
            
            if len(customers_with_different_nearest) > 0:
                print(f"WARNING: {portfolio_name} has {len(customers_with_different_nearest)} customers not assigned to their nearest branch!")
                print(f"  Portfolio Branch: {branch_id}")
                print(f"  Customers' actual nearest branches: {customers_with_different_nearest['nearest_branch'].value_counts().to_dict()}")
            else:
                avg_distance = customers_df['distance_to_nearest'].mean()
                print(f"✓ {portfolio_name}: All {len(customers_df)} customers correctly assigned to nearest branch {branch_id} (avg distance: {avg_distance:.2f} miles)")

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
        
        # Optimize portfolios from DataFrames with nearest AU assignment
        portfolios = optimize_portfolios_from_dataframes(
            customers_df=customers_df,  # Your customer DataFrame
            branches_df=branches_df,    # Your branch DataFrame
            min_portfolio_size=220,
            max_portfolio_size=250,
            radius_miles=20
        )
        
        if portfolios:
            # Validate nearest branch assignments
            validate_nearest_branch_assignment(portfolios)
            
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
            
            # Calculate average distances for in-market portfolios
            in_market_portfolios = [p for p in portfolios.values() if p['type'] == 'IN_MARKET']
            if in_market_portfolios:
                avg_distances = [p['avg_distance_to_branch'] for p in in_market_portfolios if p['avg_distance_to_branch'] is not None]
                overall_avg_distance = sum(avg_distances) / len(avg_distances) if avg_distances else 0
                print(f"Average distance to branch for in-market portfolios: {overall_avg_distance:.2f} miles")
            
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
