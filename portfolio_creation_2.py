import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from scipy.spatial.distance import cdist

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles between two points"""
    R = 3959  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# OPTION 1: Customer-to-customer clustering (current approach)
def create_customer_portfolios_v1(customer_df, branch_df):
    """
    Cluster customers based on customer-to-customer proximity, then assign to nearest branch
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    coords = np.radians(customers_clean[['LAT_NUM', 'LON_NUM']].values)
    
    eps_miles = 20
    eps_radians = eps_miles / 3959
    min_samples = 200
    
    dbscan = DBSCAN(eps=eps_radians, min_samples=min_samples, metric='haversine')
    cluster_labels = dbscan.fit_predict(coords)
    customers_clean['cluster'] = cluster_labels
    
    cluster_sizes = customers_clean['cluster'].value_counts()
    valid_clusters = cluster_sizes[(cluster_sizes >= 200) & (cluster_sizes <= 280)].index
    valid_clusters = valid_clusters[valid_clusters != -1]
    
    clustered_customers = customers_clean[customers_clean['cluster'].isin(valid_clusters)].copy()
    
    if len(clustered_customers) == 0:
        print("No valid clusters found with the given constraints")
        return pd.DataFrame(), pd.DataFrame()
    
    cluster_centroids = clustered_customers.groupby('cluster')[['LAT_NUM', 'LON_NUM']].mean()
    
    cluster_assignments = []
    for cluster_id, centroid in cluster_centroids.iterrows():
        min_distance = float('inf')
        assigned_branch = None
        
        for _, branch in branch_df.iterrows():
            distance = haversine_distance(
                centroid['LAT_NUM'], centroid['LON_NUM'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            if distance < min_distance:
                min_distance = distance
                assigned_branch = branch['BRANCH_AU']
        
        cluster_assignments.append({
            'cluster': cluster_id,
            'assigned_branch': assigned_branch,
            'distance_to_branch': min_distance,
            'centroid_lat': centroid['LAT_NUM'],
            'centroid_lon': centroid['LON_NUM']
        })
    
    cluster_branch_map = pd.DataFrame(cluster_assignments)
    final_assignments = clustered_customers.merge(
        cluster_branch_map[['cluster', 'assigned_branch']], on='cluster', how='left'
    )
    
    return final_assignments, cluster_branch_map

# OPTION 2: Branch-centered clustering (alternative approach)
def create_customer_portfolios_v2(customer_df, branch_df):
    """
    Create clusters around each branch, ensuring customers are within 20 miles of branch
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['assigned_branch'] = None
    customers_clean['distance_to_branch'] = float('inf')
    
    branch_portfolios = []
    
    for _, branch in branch_df.iterrows():
        # Find all customers within 20 miles of this branch
        customers_near_branch = []
        
        for idx, customer in customers_clean.iterrows():
            if customers_clean.loc[idx, 'assigned_branch'] is not None:
                continue  # Already assigned
                
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            
            if distance <= 20:
                customers_near_branch.append((idx, distance))
        
        # Sort by distance and take only 200-280 closest customers
        customers_near_branch.sort(key=lambda x: x[1])
        
        if 200 <= len(customers_near_branch) <= 280:
            # Assign these customers to this branch
            for idx, distance in customers_near_branch:
                customers_clean.loc[idx, 'assigned_branch'] = branch['BRANCH_AU']
                customers_clean.loc[idx, 'distance_to_branch'] = distance
            
            branch_portfolios.append({
                'branch': branch['BRANCH_AU'],
                'customer_count': len(customers_near_branch),
                'max_distance': max([d for _, d in customers_near_branch])
            })
    
    # Keep only assigned customers
    assigned_customers = customers_clean[customers_clean['assigned_branch'].notna()].copy()
    portfolio_summary = pd.DataFrame(branch_portfolios)
    
    return assigned_customers, portfolio_summary

# Use the desired approach
def create_customer_portfolios(customer_df, branch_df):
    """Default to customer-to-customer clustering approach"""
    return create_customer_portfolios_v1(customer_df, branch_df)

# Execute clustering
result_df, cluster_summary = create_customer_portfolios(customer_df, branch_df)

# Display results
print(f"Total customers clustered: {len(result_df)}")
print(f"Number of clusters: {len(cluster_summary)}")
print("\nCluster Summary:")
print(cluster_summary)

print("\nCustomer assignments preview:")
print(result_df[['ECN', 'cluster', 'assigned_branch', 'BILLINGCITY', 'BILLINGSTATE']].head(10))

# Portfolio sizes by branch
portfolio_sizes = result_df.groupby('assigned_branch').size()
print("\nPortfolio sizes by branch:")
print(portfolio_sizes)
