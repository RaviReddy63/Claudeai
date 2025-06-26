import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles between two points"""
    R = 3959  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_cluster_radius(customers_coords):
    """Calculate maximum distance from centroid to any point in cluster"""
    if len(customers_coords) <= 1:
        return 0
    
    coords_array = np.array(customers_coords)
    centroid = coords_array.mean(axis=0)
    max_distance = 0
    
    for coord in coords_array:
        distance = haversine_distance(centroid[0], centroid[1], coord[0], coord[1])
        if distance > max_distance:
            max_distance = distance
    
    return max_distance

def constrained_clustering(customer_df, min_size=200, max_size=280, max_radius=20):
    """
    Create initial clusters to identify potential AU locations
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    unassigned_customers = customers_clean.copy()
    cluster_id = 0
    potential_centers = []
    
    print(f"Starting with {len(unassigned_customers)} customers")
    
    while len(unassigned_customers) >= min_size:
        seed_idx = unassigned_customers.index[0]
        seed_customer = unassigned_customers.loc[seed_idx]
        
        current_cluster = [seed_idx]
        current_coords = [(seed_customer['LAT_NUM'], seed_customer['LON_NUM'])]
        
        candidates = []
        for idx, customer in unassigned_customers.iterrows():
            if idx == seed_idx:
                continue
                
            distance = haversine_distance(
                seed_customer['LAT_NUM'], seed_customer['LON_NUM'],
                customer['LAT_NUM'], customer['LON_NUM']
            )
            
            if distance <= max_radius:
                candidates.append((idx, customer, distance))
        
        candidates.sort(key=lambda x: x[2])
        
        for candidate_idx, candidate_customer, _ in candidates:
            if len(current_cluster) >= max_size:
                break
            
            test_coords = current_coords + [(candidate_customer['LAT_NUM'], candidate_customer['LON_NUM'])]
            test_radius = calculate_cluster_radius(np.array(test_coords))
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append((candidate_customer['LAT_NUM'], candidate_customer['LON_NUM']))
        
        if min_size <= len(current_cluster) <= max_size:
            cluster_radius = calculate_cluster_radius(np.array(current_coords))
            centroid_lat = np.mean([coord[0] for coord in current_coords])
            centroid_lon = np.mean([coord[1] for coord in current_coords])
            
            potential_centers.append({
                'center_lat': centroid_lat,
                'center_lon': centroid_lon,
                'estimated_customers': len(current_cluster)
            })
            
            unassigned_customers = unassigned_customers.drop(current_cluster)
            cluster_id += 1
            
            print(f"Potential center {cluster_id}: {len(current_cluster)} customers, radius: {cluster_radius:.2f} miles")
        else:
            unassigned_customers = unassigned_customers.drop([seed_idx])
    
    return pd.DataFrame(potential_centers)

def create_circular_portfolios_around_branches(customer_df, branch_df, max_radius=20, max_customers=280):
    """
    Create circular portfolios around each branch within radius and customer limits
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['assigned_branch'] = None
    customers_clean['distance_to_branch'] = float('inf')
    
    portfolio_results = []
    
    print(f"Creating circular portfolios around {len(branch_df)} branches")
    
    for _, branch in branch_df.iterrows():
        branch_au = branch['BRANCH_AU']
        branch_lat = branch['BRANCH_LAT_NUM']
        branch_lon = branch['BRANCH_LON_NUM']
        
        # Find all unassigned customers within radius
        eligible_customers = []
        
        for idx, customer in customers_clean.iterrows():
            # Skip if already assigned
            if customers_clean.loc[idx, 'assigned_branch'] is not None:
                continue
                
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                branch_lat, branch_lon
            )
            
            if distance <= max_radius:
                eligible_customers.append((idx, distance))
        
        # Sort by distance (closest first) and take up to max_customers
        eligible_customers.sort(key=lambda x: x[1])
        selected_customers = eligible_customers[:max_customers]
        
        if len(selected_customers) > 0:
            # Assign these customers to this branch
            assigned_count = 0
            max_distance = 0
            
            for idx, distance in selected_customers:
                customers_clean.loc[idx, 'assigned_branch'] = branch_au
                customers_clean.loc[idx, 'distance_to_branch'] = distance
                assigned_count += 1
                max_distance = max(max_distance, distance)
            
            portfolio_results.append({
                'branch_au': branch_au,
                'branch_lat': branch_lat,
                'branch_lon': branch_lon,
                'customers_assigned': assigned_count,
                'max_distance_miles': max_distance,
                'portfolio_radius': max_distance  # Actual radius of the circular portfolio
            })
            
            print(f"Branch {branch_au}: {assigned_count} customers, max distance: {max_distance:.2f} miles")
        else:
            print(f"Branch {branch_au}: No eligible customers within {max_radius} miles")
    
    # Keep only assigned customers
    assigned_customers = customers_clean[customers_clean['assigned_branch'].notna()].copy()
    portfolio_summary = pd.DataFrame(portfolio_results)
    
    return assigned_customers, portfolio_summary

def create_customer_portfolios(customer_df, branch_df):
    """
    Main function: Create circular portfolios around branches
    """
    print("="*60)
    print("CREATING CIRCULAR PORTFOLIOS AROUND BRANCHES")
    print("="*60)
    
    # Create circular portfolios directly around existing branches
    assigned_customers, portfolio_summary = create_circular_portfolios_around_branches(
        customer_df, branch_df, max_radius=20, max_customers=280
    )
    
    if len(assigned_customers) == 0:
        print("No customers could be assigned to any branch portfolios")
        return pd.DataFrame(), pd.DataFrame()
    
    return assigned_customers, portfolio_summary

# Execute clustering
result_df, portfolio_summary = create_customer_portfolios(customer_df, branch_df)

# Display results
if len(result_df) > 0:
    print(f"\nTotal customers assigned: {len(result_df)}")
    print(f"Customers left unassigned: {len(customer_df) - len(result_df)}")
    print(f"Branches with portfolios: {len(portfolio_summary)}")
    
    print("\nPortfolio Summary:")
    print(portfolio_summary[['branch_au', 'customers_assigned', 'portfolio_radius']].round(2))
    
    print("\nCustomer assignments preview:")
    print(result_df[['ECN', 'assigned_branch', 'distance_to_branch', 'BILLINGCITY', 'BILLINGSTATE']].head(10))
    
    # Portfolio sizes by branch
    portfolio_sizes = result_df.groupby('assigned_branch').size()
    print(f"\nPortfolio sizes by branch:")
    print(portfolio_sizes)
    
    # CUSTOMER-AU MAPPING
    print("\n" + "="*50)
    print("CUSTOMER-AU MAPPING")
    print("="*50)
    
    # Simple mapping (ECN to AU)
    customer_au_mapping = result_df[['ECN', 'assigned_branch', 'distance_to_branch']].copy()
    customer_au_mapping.columns = ['Customer_ECN', 'Assigned_AU', 'Distance_Miles']
    customer_au_mapping['Distance_Miles'] = customer_au_mapping['Distance_Miles'].round(2)
    print("\nCustomer-AU Mapping with distances:")
    print(customer_au_mapping.head(10))
    
    # Detailed mapping with location info
    detailed_mapping = result_df[['ECN', 'assigned_branch', 'distance_to_branch', 'BILLINGCITY', 'BILLINGSTATE', 'LAT_NUM', 'LON_NUM']].copy()
    detailed_mapping.columns = ['Customer_ECN', 'Assigned_AU', 'Distance_Miles', 'Customer_City', 'Customer_State', 'Customer_Lat', 'Customer_Lon']
    detailed_mapping['Distance_Miles'] = detailed_mapping['Distance_Miles'].round(2)
    
    # Save to CSV
    customer_au_mapping.to_csv('customer_au_mapping.csv', index=False)
    detailed_mapping.to_csv('detailed_customer_au_mapping.csv', index=False)
    portfolio_summary.to_csv('portfolio_summary.csv', index=False)
    
    print(f"\nFiles saved:")
    print(f"- customer_au_mapping.csv ({len(customer_au_mapping)} records)")
    print(f"- detailed_customer_au_mapping.csv ({len(detailed_mapping)} records)")
    print(f"- portfolio_summary.csv ({len(portfolio_summary)} branches)")
    
    # Statistics
    print(f"\nPortfolio Statistics:")
    print(f"- Average portfolio size: {portfolio_sizes.mean():.1f} customers")
    print(f"- Largest portfolio: {portfolio_sizes.max()} customers")
    print(f"- Smallest portfolio: {portfolio_sizes.min()} customers")
    print(f"- Average portfolio radius: {portfolio_summary['portfolio_radius'].mean():.2f} miles")
    
else:
    print("No customer portfolios could be created with the given constraints")
