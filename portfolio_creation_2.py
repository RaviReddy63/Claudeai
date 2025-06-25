import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from collections import defaultdict

def cluster_customers_to_branches(customer_df, branch_df, min_cluster_size=200, max_cluster_size=280, max_radius_miles=20):
    """
    Cluster customers and assign to nearest branches
    
    Parameters:
    customer_df: DataFrame with columns ['ECN', 'LAT_NUM', 'LON_NUM', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE']
    branch_df: DataFrame with columns ['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
    """
    
    # Convert miles to kilometers for distance calculations
    max_radius_km = max_radius_miles * 1.60934
    
    # Initial clustering using DBSCAN
    customer_coords = customer_df[['LAT_NUM', 'LON_NUM']].values
    
    # DBSCAN with approximate parameters (will be refined)
    eps_km = max_radius_km / 111.0  # Rough conversion to degrees
    clustering = DBSCAN(eps=eps_km, min_samples=min_cluster_size).fit(customer_coords)
    
    customer_df = customer_df.copy()
    customer_df['cluster'] = clustering.labels_
    
    converged = False
    iteration = 0
    max_iterations = 50
    
    while not converged and iteration < max_iterations:
        iteration += 1
        old_assignments = customer_df['cluster'].copy()
        
        # Get current clusters (excluding noise points with label -1)
        valid_clusters = customer_df[customer_df['cluster'] >= 0]['cluster'].unique()
        
        # Process each cluster
        for cluster_id in valid_clusters:
            cluster_customers = customer_df[customer_df['cluster'] == cluster_id]
            
            # Check cluster size and radius constraints
            if len(cluster_customers) < min_cluster_size or len(cluster_customers) > max_cluster_size:
                # Mark as noise if size constraints not met
                customer_df.loc[customer_df['cluster'] == cluster_id, 'cluster'] = -1
                continue
            
            # Calculate cluster centroid
            centroid_lat = cluster_customers['LAT_NUM'].mean()
            centroid_lon = cluster_customers['LON_NUM'].mean()
            
            # Check radius constraint
            max_distance = 0
            for _, customer in cluster_customers.iterrows():
                distance = geodesic((centroid_lat, centroid_lon), 
                                  (customer['LAT_NUM'], customer['LON_NUM'])).miles
                max_distance = max(max_distance, distance)
            
            if max_distance > max_radius_miles:
                # Mark as noise if radius constraint not met
                customer_df.loc[customer_df['cluster'] == cluster_id, 'cluster'] = -1
                continue
            
            # Find nearest branch to centroid
            min_distance = float('inf')
            nearest_branch = None
            
            for _, branch in branch_df.iterrows():
                distance = geodesic((centroid_lat, centroid_lon), 
                                  (branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM'])).miles
                if distance < min_distance:
                    min_distance = distance
                    nearest_branch = branch['BRANCH_AU']
            
            # Update centroid to nearest branch location
            branch_info = branch_df[branch_df['BRANCH_AU'] == nearest_branch].iloc[0]
            new_centroid_lat = branch_info['BRANCH_LAT_NUM']
            new_centroid_lon = branch_info['BRANCH_LON_NUM']
            
            # Reassign customers based on new centroid
            valid_customers = []
            for idx, customer in cluster_customers.iterrows():
                distance = geodesic((new_centroid_lat, new_centroid_lon), 
                                  (customer['LAT_NUM'], customer['LON_NUM'])).miles
                if distance <= max_radius_miles:
                    valid_customers.append(idx)
                else:
                    customer_df.loc[idx, 'cluster'] = -1
            
            # Check if cluster still meets size constraints after reassignment
            if len(valid_customers) < min_cluster_size or len(valid_customers) > max_cluster_size:
                customer_df.loc[customer_df.index.isin(valid_customers), 'cluster'] = -1
        
        # Re-cluster any noise points that might form new valid clusters
        noise_customers = customer_df[customer_df['cluster'] == -1]
        if len(noise_customers) >= min_cluster_size:
            noise_coords = noise_customers[['LAT_NUM', 'LON_NUM']].values
            noise_clustering = DBSCAN(eps=eps_km, min_samples=min_cluster_size).fit(noise_coords)
            
            # Assign new cluster IDs
            max_cluster_id = customer_df['cluster'].max()
            if max_cluster_id < 0:
                max_cluster_id = -1
            
            for i, label in enumerate(noise_clustering.labels_):
                if label >= 0:
                    customer_df.loc[noise_customers.index[i], 'cluster'] = max_cluster_id + 1 + label
        
        # Check for convergence
        converged = (old_assignments == customer_df['cluster']).all()
    
    # Create final mapping
    au_customer_mapping = defaultdict(list)
    
    # Process valid clusters only
    valid_clusters = customer_df[customer_df['cluster'] >= 0]['cluster'].unique()
    
    for cluster_id in valid_clusters:
        cluster_customers = customer_df[customer_df['cluster'] == cluster_id]
        
        if len(cluster_customers) >= min_cluster_size and len(cluster_customers) <= max_cluster_size:
            # Find centroid
            centroid_lat = cluster_customers['LAT_NUM'].mean()
            centroid_lon = cluster_customers['LON_NUM'].mean()
            
            # Find nearest branch
            min_distance = float('inf')
            nearest_branch = None
            
            for _, branch in branch_df.iterrows():
                distance = geodesic((centroid_lat, centroid_lon), 
                                  (branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM'])).miles
                if distance < min_distance:
                    min_distance = distance
                    nearest_branch = branch['BRANCH_AU']
            
            # Add customers to mapping
            customer_list = cluster_customers['ECN'].tolist()
            au_customer_mapping[nearest_branch].extend(customer_list)
    
    return dict(au_customer_mapping)

# Usage example:
# customer_data = pd.read_csv('customer_data.csv')
# branch_data = pd.read_csv('branch_data.csv')
# result = cluster_customers_to_branches(customer_data, branch_data)
# 
# Print results:
# for au, customers in result.items():
#     print(f"Branch AU {au}: {len(customers)} customers")
#     print(f"Customer ECNs: {customers}")
#     print("-" * 50)
