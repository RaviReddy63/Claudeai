def tag_customers_with_nearest_au(portfolios: Dict, branches: pd.DataFrame) -> Dict:
    """
    Tag customers in In-Market portfolios with their nearest AU branch.
    This adds nearest_au and distance_to_nearest_au columns to customer data.
    """
    print(f"Tagging customers with nearest AU...")
    
    updated_portfolios = portfolios.copy()
    
    forimport pandas as pd
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

def optimize_customer_portfolio_assignments(portfolios: Dict, branches: pd.DataFrame) -> Dict:
    """
    Reassign customers between In-Market portfolios to ensure each customer is assigned 
    to the portfolio with the nearest base AU.
    """
    print(f"Optimizing customer assignments to nearest portfolio AUs...")
    
    # Get all In-Market portfolios
    in_market_portfolios = {k: v for k, v in portfolios.items() if v['type'] == 'IN_MARKET'}
    
    if not in_market_portfolios:
        print("No In-Market portfolios found to optimize")
        return portfolios
    
    # Create a mapping of portfolio to branch info
    portfolio_branches = {}
    for portfolio_name, portfolio_info in in_market_portfolios.items():
        portfolio_branches[portfolio_name] = {
            'branch_id': portfolio_info['branch_id'],
            'branch_lat': portfolio_info['branch_lat'],
            'branch_lon': portfolio_info['branch_lon']
        }
    
    # Collect all customers from In-Market portfolios
    all_customers = []
    customer_to_original_portfolio = {}
    
    for portfolio_name, portfolio_info in in_market_portfolios.items():
        for customer_idx in portfolio_info['customer_indices']:
            all_customers.append(customer_idx)
            customer_to_original_portfolio[customer_idx] = portfolio_name
    
    print(f"Total customers in In-Market portfolios to reassign: {len(all_customers)}")
    
    # For each customer, find the nearest portfolio AU
    customer_reassignments = {}
    
    for customer_idx in all_customers:
        # Get customer data from any portfolio (they should all have the same customer data)
        customer_data = None
        for portfolio_info in in_market_portfolios.values():
            if customer_idx in portfolio_info['customer_indices']:
                customer_row = portfolio_info['customers'][portfolio_info['customers'].index == customer_idx]
                if not customer_row.empty:
                    customer_data = customer_row.iloc[0]
                    break
        
        if customer_data is None:
            print(f"Warning: Could not find customer data for index {customer_idx}")
            continue
        
        # Find nearest portfolio AU
        min_distance = float('inf')
        nearest_portfolio = None
        
        for portfolio_name, branch_info in portfolio_branches.items():
            try:
                distance = haversine_distance(
                    customer_data['LAT_NUM'], customer_data['LON_NUM'],
                    branch_info['branch_lat'], branch_info['branch_lon']
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_portfolio = portfolio_name
                    
            except Exception as e:
                print(f"Error calculating distance for customer {customer_idx} to portfolio {portfolio_name}: {e}")
                continue
        
        if nearest_portfolio:
            customer_reassignments[customer_idx] = {
                'from_portfolio': customer_to_original_portfolio[customer_idx],
                'to_portfolio': nearest_portfolio,
                'distance': min_distance
            }
    
    # Count reassignments
    reassignment_count = 0
    for customer_idx, assignment in customer_reassignments.items():
        if assignment['from_portfolio'] != assignment['to_portfolio']:
            reassignment_count += 1
    
    print(f"Customers that need reassignment: {reassignment_count}")
    
    # Create new portfolio structure with reassigned customers
    optimized_portfolios = portfolios.copy()
    
    # Clear customer data from In-Market portfolios
    for portfolio_name in in_market_portfolios.keys():
        optimized_portfolios[portfolio_name]['customer_indices'] = []
        optimized_portfolios[portfolio_name]['customers'] = optimized_portfolios[portfolio_name]['customers'].iloc[0:0]  # Empty DataFrame with same structure
    
    # Reassign customers to their nearest portfolios
    for customer_idx, assignment in customer_reassignments.items():
        target_portfolio = assignment['to_portfolio']
        
        # Find the customer data from original portfolio
        customer_data = None
        for portfolio_info in in_market_portfolios.values():
            customer_row = portfolio_info['customers'][portfolio_info['customers'].index == customer_idx]
            if not customer_row.empty:
                customer_data = customer_row.iloc[0]
                break
        
        if customer_data is not None:
            # Add customer to target portfolio
            optimized_portfolios[target_portfolio]['customer_indices'].append(customer_idx)
            
            # Add customer data to target portfolio DataFrame
            if optimized_portfolios[target_portfolio]['customers'].empty:
                optimized_portfolios[target_portfolio]['customers'] = customer_data.to_frame().T
            else:
                optimized_portfolios[target_portfolio]['customers'] = pd.concat([
                    optimized_portfolios[target_portfolio]['customers'],
                    customer_data.to_frame().T
                ], ignore_index=False)
    
    # Update customer counts and clean up
    for portfolio_name in in_market_portfolios.keys():
        customer_count = len(optimized_portfolios[portfolio_name]['customer_indices'])
        optimized_portfolios[portfolio_name]['customer_count'] = customer_count
        
        # Calculate average distance to portfolio AU
        if customer_count > 0:
            customers_df = optimized_portfolios[portfolio_name]['customers']
            branch_lat = optimized_portfolios[portfolio_name]['branch_lat']
            branch_lon = optimized_portfolios[portfolio_name]['branch_lon']
            
            distances = []
            for idx, customer in customers_df.iterrows():
                try:
                    distance = haversine_distance(
                        customer['LAT_NUM'], customer['LON_NUM'],
                        branch_lat, branch_lon
                    )
                    distances.append(distance)
                except Exception as e:
                    print(f"Error calculating distance for customer {idx}: {e}")
            
            avg_distance = sum(distances) / len(distances) if distances else 0
            optimized_portfolios[portfolio_name]['avg_distance_to_au'] = avg_distance
            
            print(f"Portfolio {portfolio_name}: {customer_count} customers (avg distance to AU: {avg_distance:.2f} miles)")
        else:
            optimized_portfolios[portfolio_name]['avg_distance_to_au'] = 0
            print(f"Portfolio {portfolio_name}: 0 customers after optimization")
    
    return optimized_portfolios

def create_in_market_portfolios(customers: pd.DataFrame, branches: pd.DataFrame, 
                               min_portfolio_size: int = 220, max_portfolio_size: int = 250,
                               radius_miles: float = 20) -> Tuple[Dict, pd.DataFrame]:
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
        customers = validate_customer_data(customers_df)
        branches = validate_branch_data(branches_df)
        
        print(f"Loaded {len(customers)} customers and {len(branches)} branches")
        
        # Create in-market portfolios using original logic
        in_market_portfolios, customers_with_assignments = create_in_market_portfolios(
            customers, branches, min_portfolio_size, max_portfolio_size, radius_miles
        )
        
        print(f"Created {len(in_market_portfolios)} in-market portfolios")
        
        # Optimize customer assignments within In-Market portfolios
        optimized_in_market_portfolios = optimize_customer_portfolio_assignments(
            in_market_portfolios, branches
        )
        
        print(f"Optimized customer assignments within In-Market portfolios")
        
        # Create centralized portfolios for remaining customers
        centralized_portfolios = create_centralized_portfolios(
            customers_with_assignments, min_portfolio_size, max_portfolio_size
        )
        
        print(f"Created {len(centralized_portfolios)} centralized portfolios")
        
        # Combine optimized in-market and centralized portfolios
        all_portfolios = {**optimized_in_market_portfolios, **centralized_portfolios}
        
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
            'Avg_Distance_to_AU': portfolio_info.get('avg_distance_to_au', None)
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

def validate_portfolio_au_assignments(portfolios: Dict) -> None:
    """
    Validate that customers in each In-Market portfolio are closer to their assigned AU 
    than to any other In-Market portfolio's AU.
    """
    print("\n=== VALIDATION: Portfolio AU Assignment Optimization ===")
    
    in_market_portfolios = {k: v for k, v in portfolios.items() if v['type'] == 'IN_MARKET'}
    
    if len(in_market_portfolios) < 2:
        print("Less than 2 In-Market portfolios found, skipping validation")
        return
    
    # Create list of all portfolio AUs for comparison
    portfolio_aus = {}
    for portfolio_name, portfolio_info in in_market_portfolios.items():
        portfolio_aus[portfolio_name] = {
            'branch_lat': portfolio_info['branch_lat'],
            'branch_lon': portfolio_info['branch_lon'],
            'branch_id': portfolio_info['branch_id']
        }
    
    total_misassigned = 0
    
    for portfolio_name, portfolio_info in in_market_portfolios.items():
        customers_df = portfolio_info['customers']
        portfolio_au = portfolio_aus[portfolio_name]
        misassigned_customers = 0
        
        for idx, customer in customers_df.iterrows():
            # Calculate distance to assigned portfolio AU
            assigned_distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                portfolio_au['branch_lat'], portfolio_au['branch_lon']
            )
            
            # Check if any other portfolio AU is closer
            closer_portfolio = None
            min_distance = assigned_distance
            
            for other_portfolio_name, other_au in portfolio_aus.items():
                if other_portfolio_name != portfolio_name:
                    other_distance = haversine_distance(
                        customer['LAT_NUM'], customer['LON_NUM'],
                        other_au['branch_lat'], other_au['branch_lon']
                    )
                    
                    if other_distance < min_distance:
                        min_distance = other_distance
                        closer_portfolio = other_portfolio_name
            
            if closer_portfolio:
                misassigned_customers += 1
        
        total_misassigned += misassigned_customers
        
        if misassigned_customers > 0:
            print(f"⚠️  {portfolio_name} (AU: {portfolio_au['branch_id']}): {misassigned_customers} customers are closer to other portfolio AUs")
        else:
            avg_distance = portfolio_info.get('avg_distance_to_au', 0)
            print(f"✅ {portfolio_name} (AU: {portfolio_au['branch_id']}): All {len(customers_df)} customers optimally assigned (avg: {avg_distance:.2f} miles)")
    
    if total_misassigned == 0:
        print(f"\n🎉 PERFECT! All customers in In-Market portfolios are optimally assigned to their nearest portfolio AU")
    else:
        print(f"\n⚠️  Total misassigned customers: {total_misassigned}")
        print("Consider running the optimization again or adjusting parameters.")

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
            # Validate portfolio AU assignments
            validate_portfolio_au_assignments(portfolios)
            
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
                avg_distances = [p['avg_distance_to_au'] for p in in_market_portfolios if p.get('avg_distance_to_au') is not None]
                overall_avg_distance = sum(avg_distances) / len(avg_distances) if avg_distances else 0
                print(f"Average distance to AU for in-market portfolios: {overall_avg_distance:.2f} miles")
            
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
