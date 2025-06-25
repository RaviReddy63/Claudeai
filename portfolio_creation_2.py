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

def create_customer_portfolios(customer_df, branch_df):
    """
    Cluster customers and assign to nearest branches
    
    Parameters:
    customer_df: DataFrame with columns ECN, LAT_NUM, LON_NUM, BILLINGSTREET, BILLINGCITY, BILLINGSTATE
    branch_df: DataFrame with columns BRANCH_AU, BRANCH_LAT_NUM, BRANCH_LON_NUM
    """
    
    # Remove customers with missing coordinates
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    
    # Convert coordinates to radians for DBSCAN
    coords = np.radians(customers_clean[['LAT_NUM', 'LON_NUM']].values)
    
    # DBSCAN parameters
    # eps in radians (20 miles ≈ 0.00508 radians)
    eps_miles = 20
    eps_radians = eps_miles / 3959  # Earth radius in miles
    min_samples = 200  # Minimum cluster size
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps_radians, min_samples=min_samples, metric='haversine')
    cluster_labels = dbscan.fit_predict(coords)
    
    customers_clean['cluster'] = cluster_labels
    
    # Filter clusters by size (200-280 customers)
    cluster_sizes = customers_clean['cluster'].value_counts()
    valid_clusters = cluster_sizes[(cluster_sizes >= 200) & (cluster_sizes <= 280)].index
    valid_clusters = valid_clusters[valid_clusters != -1]  # Remove noise points
    
    # Keep only customers in valid clusters
    clustered_customers = customers_clean[customers_clean['cluster'].isin(valid_clusters)].copy()
    
    if len(clustered_customers) == 0:
        print("No valid clusters found with the given constraints")
        return pd.DataFrame()
    
    # Calculate cluster centroids
    cluster_centroids = clustered_customers.groupby('cluster')[['LAT_NUM', 'LON_NUM']].mean()
    
    # Assign each cluster to nearest branch
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
    
    # Create assignment mapping
    cluster_branch_map = pd.DataFrame(cluster_assignments)
    
    # Merge with customer data
    final_assignments = clustered_customers.merge(
        cluster_branch_map[['cluster', 'assigned_branch']], 
        on='cluster', 
        how='left'
    )
    
    return final_assignments, cluster_branch_map

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
