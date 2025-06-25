import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import KMeans
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in miles
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in miles
    return c * r

def find_customers_within_radius(customers_df, branch_lat, branch_lon, radius_miles=20):
    """
    Find all customers within specified radius of a branch
    """
    customers_in_range = []
    
    for idx, customer in customers_df.iterrows():
        distance = haversine_distance(
            customer['LAT_NUM'], customer['LON_NUM'],
            branch_lat, branch_lon
        )
        
        if distance <= radius_miles:
            customers_in_range.append({
                'ECN': customer['ECN'],
                'distance': distance,
                'customer_idx': idx
            })
    
    return customers_in_range

def create_in_market_portfolios(customers_df, branches_df, min_size=220, max_size=250):
    """
    Create in-market portfolios for branches that have sufficient customers within 20 miles
    """
    in_market_portfolios = []
    assigned_customers = set()
    
    # For each branch, check if it can support an in-market portfolio
    for idx, branch in branches_df.iterrows():
        # Find customers within 20 miles
        customers_in_range = find_customers_within_radius(
            customers_df, 
            branch['BRANCH_LAT_NUM'], 
            branch['BRANCH_LON_NUM']
        )
        
        # Filter out already assigned customers
        available_customers = [
            c for c in customers_in_range 
            if c['customer_idx'] not in assigned_customers
        ]
        
        # If we have enough customers for a portfolio
        if len(available_customers) >= min_size:
            # Sort by distance to get closest customers first
            available_customers.sort(key=lambda x: x['distance'])
            
            # Take up to max_size customers
            selected_customers = available_customers[:max_size]
            
            # Mark these customers as assigned
            for customer in selected_customers:
                assigned_customers.add(customer['customer_idx'])
            
            portfolio = {
                'portfolio_type': 'in_market',
                'branch_au': branch['BRANCH_AU'],
                'branch_lat': branch['BRANCH_LAT_NUM'],
                'branch_lon': branch['BRANCH_LON_NUM'],
                'customers': [c['ECN'] for c in selected_customers],
                'customer_count': len(selected_customers),
                'avg_distance': np.mean([c['distance'] for c in selected_customers])
            }
            
            in_market_portfolios.append(portfolio)
    
    return in_market_portfolios, assigned_customers

def cluster_remaining_customers(customers_df, assigned_customers, min_size=220, max_size=250):
    """
    Cluster remaining customers for centralized portfolios
    """
    # Get unassigned customers
    unassigned_mask = ~customers_df.index.isin(assigned_customers)
    unassigned_customers = customers_df[unassigned_mask].copy()
    
    if len(unassigned_customers) == 0:
        return []
    
    # Estimate number of clusters needed
    n_customers = len(unassigned_customers)
    n_clusters = max(1, n_customers // max_size)
    
    # If we don't have enough customers for even one portfolio, return empty
    if n_customers < min_size:
        print(f"Warning: Only {n_customers} unassigned customers remaining, less than minimum portfolio size")
        return []
    
    # Perform K-means clustering on customer locations
    coordinates = unassigned_customers[['LAT_NUM', 'LON_NUM']].values
    
    if n_clusters == 1:
        # Single cluster - assign all customers to one portfolio
        cluster_labels = np.zeros(len(unassigned_customers))
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates)
    
    # Create portfolios from clusters
    centralized_portfolios = []
    
    for cluster_id in range(n_clusters):
        cluster_customers = unassigned_customers[cluster_labels == cluster_id]
        
        # If cluster is too small, merge with largest cluster
        if len(cluster_customers) < min_size:
            # Find the largest cluster that can accommodate these customers
            for i, portfolio in enumerate(centralized_portfolios):
                if portfolio['customer_count'] + len(cluster_customers) <= max_size:
                    portfolio['customers'].extend(cluster_customers['ECN'].tolist())
                    portfolio['customer_count'] += len(cluster_customers)
                    break
            else:
                # If no existing portfolio can take them, create a new one anyway
                # (This handles edge cases where clustering doesn't work perfectly)
                if len(cluster_customers) > 0:
                    portfolio = {
                        'portfolio_type': 'centralized',
                        'cluster_id': f'centralized_{cluster_id}',
                        'customers': cluster_customers['ECN'].tolist(),
                        'customer_count': len(cluster_customers),
                        'center_lat': cluster_customers['LAT_NUM'].mean(),
                        'center_lon': cluster_customers['LON_NUM'].mean()
                    }
                    centralized_portfolios.append(portfolio)
        else:
            # Cluster is large enough, create portfolio
            portfolio = {
                'portfolio_type': 'centralized',
                'cluster_id': f'centralized_{cluster_id}',
                'customers': cluster_customers['ECN'].tolist(),
                'customer_count': len(cluster_customers),
                'center_lat': cluster_customers['LAT_NUM'].mean(),
                'center_lon': cluster_customers['LON_NUM'].mean()
            }
            centralized_portfolios.append(portfolio)
    
    return centralized_portfolios

def balance_portfolios(portfolios, min_size=220, max_size=250):
    """
    Balance portfolio sizes by redistributing customers
    """
    # Separate oversized and undersized portfolios
    oversized = [p for p in portfolios if p['customer_count'] > max_size]
    undersized = [p for p in portfolios if p['customer_count'] < min_size]
    balanced = [p for p in portfolios if min_size <= p['customer_count'] <= max_size]
    
    # Redistribute customers from oversized to undersized portfolios
    for over_portfolio in oversized:
        excess_customers = over_portfolio['customer_count'] - max_size
        customers_to_move = over_portfolio['customers'][-excess_customers:]
        over_portfolio['customers'] = over_portfolio['customers'][:-excess_customers]
        over_portfolio['customer_count'] = len(over_portfolio['customers'])
        
        # Distribute excess customers to undersized portfolios
        for under_portfolio in undersized:
            needed = min_size - under_portfolio['customer_count']
            if needed > 0 and customers_to_move:
                take = min(needed, len(customers_to_move))
                under_portfolio['customers'].extend(customers_to_move[:take])
                under_portfolio['customer_count'] += take
                customers_to_move = customers_to_move[take:]
    
    # Combine all portfolios
    all_portfolios = balanced + [p for p in oversized if p['customer_count'] >= min_size] + \
                    [p for p in undersized if p['customer_count'] >= min_size]
    
    return all_portfolios

def optimize_customer_portfolios(customers_df, branches_df, min_size=220, max_size=250):
    """
    Main function to optimize customer portfolios
    """
    print("Starting portfolio optimization...")
    print(f"Total customers: {len(customers_df)}")
    print(f"Total branches: {len(branches_df)}")
    
    # Step 1: Create in-market portfolios
    print("\nStep 1: Creating in-market portfolios...")
    in_market_portfolios, assigned_customers = create_in_market_portfolios(
        customers_df, branches_df, min_size, max_size
    )
    
    print(f"Created {len(in_market_portfolios)} in-market portfolios")
    print(f"Assigned {len(assigned_customers)} customers to in-market portfolios")
    
    # Step 2: Create centralized portfolios for remaining customers
    print("\nStep 2: Creating centralized portfolios...")
    centralized_portfolios = cluster_remaining_customers(
        customers_df, assigned_customers, min_size, max_size
    )
    
    print(f"Created {len(centralized_portfolios)} centralized portfolios")
    
    # Step 3: Balance all portfolios
    print("\nStep 3: Balancing portfolios...")
    all_portfolios = in_market_portfolios + centralized_portfolios
    balanced_portfolios = balance_portfolios(all_portfolios, min_size, max_size)
    
    print(f"Final result: {len(balanced_portfolios)} balanced portfolios")
    
    return balanced_portfolios

def generate_portfolio_summary(portfolios):
    """
    Generate a summary report of the portfolios
    """
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION SUMMARY")
    print("="*60)
    
    in_market_count = sum(1 for p in portfolios if p['portfolio_type'] == 'in_market')
    centralized_count = sum(1 for p in portfolios if p['portfolio_type'] == 'centralized')
    total_customers = sum(p['customer_count'] for p in portfolios)
    
    print(f"Total Portfolios: {len(portfolios)}")
    print(f"  - In-Market: {in_market_count}")
    print(f"  - Centralized: {centralized_count}")
    print(f"Total Customers Assigned: {total_customers}")
    
    print("\nPortfolio Size Distribution:")
    sizes = [p['customer_count'] for p in portfolios]
    print(f"  - Min Size: {min(sizes)}")
    print(f"  - Max Size: {max(sizes)}")
    print(f"  - Average Size: {np.mean(sizes):.1f}")
    
    print("\nDetailed Portfolio Information:")
    for i, portfolio in enumerate(portfolios, 1):
        if portfolio['portfolio_type'] == 'in_market':
            print(f"{i}. In-Market Portfolio (Branch: {portfolio['branch_au']})")
            print(f"   Customers: {portfolio['customer_count']}, Avg Distance: {portfolio['avg_distance']:.1f} miles")
        else:
            print(f"{i}. Centralized Portfolio (Cluster: {portfolio['cluster_id']})")
            print(f"   Customers: {portfolio['customer_count']}, Center: ({portfolio['center_lat']:.3f}, {portfolio['center_lon']:.3f})")
    
    return {
        'total_portfolios': len(portfolios),
        'in_market_count': in_market_count,
        'centralized_count': centralized_count,
        'total_customers': total_customers,
        'avg_portfolio_size': np.mean(sizes)
    }

# Example usage function
def run_portfolio_optimization(customers_csv_path, branches_csv_path):
    """
    Example function to run the optimization with CSV files
    """
    # Load data
    customers_df = pd.read_csv(customers_csv_path)
    branches_df = pd.read_csv(branches_csv_path)
    
    # Validate required columns
    required_customer_cols = ['ECN', 'LAT_NUM', 'LON_NUM', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE']
    required_branch_cols = ['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
    
    for col in required_customer_cols:
        if col not in customers_df.columns:
            raise ValueError(f"Missing required column in customers data: {col}")
    
    for col in required_branch_cols:
        if col not in branches_df.columns:
            raise ValueError(f"Missing required column in branches data: {col}")
    
    # Run optimization
    portfolios = optimize_customer_portfolios(customers_df, branches_df)
    
    # Generate summary
    summary = generate_portfolio_summary(portfolios)
    
    return portfolios, summary
