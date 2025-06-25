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
    
    centroid = customers_coords.mean(axis=0)
    max_distance = 0
    
    for coord in customers_coords:
        distance = haversine_distance(centroid[0], centroid[1], coord[0], coord[1])
        max_distance = max(max_distance, distance)
    
    return max_distance

def split_large_cluster(cluster_indices, cluster_coords, customers_df, min_size, max_size, max_radius):
    """
    Split a large cluster into smaller valid clusters using k-means approach
    """
    from sklearn.cluster import KMeans
    
    # Estimate number of sub-clusters needed
    total_customers = len(cluster_indices)
    n_subclusters = max(2, (total_customers + max_size - 1) // max_size)  # Ceiling division
    
    # Try different numbers of subclusters to find best split
    best_subclusters = []
    
    for n in range(2, min(6, total_customers // min_size + 1)):  # Try 2-5 subclusters
        try:
            # Apply k-means clustering
            coords_array = np.array(cluster_coords)
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            subcluster_labels = kmeans.fit_predict(coords_array)
            
            current_subclusters = []
            valid_split = True
            
            # Check each subcluster
            for sub_id in range(n):
                sub_mask = subcluster_labels == sub_id
                sub_indices = [cluster_indices[i] for i in range(len(cluster_indices)) if sub_mask[i]]
                sub_coords = [cluster_coords[i] for i in range(len(cluster_coords)) if sub_mask[i]]
                
                # Check constraints
                if len(sub_indices) < min_size or len(sub_indices) > max_size:
                    valid_split = False
                    break
                
                sub_radius = calculate_cluster_radius(np.array(sub_coords))
                if sub_radius > max_radius:
                    valid_split = False
                    break
                
                current_subclusters.append((sub_indices, sub_coords))
            
            # If this split is valid and better than previous, keep it
            if valid_split and len(current_subclusters) > len(best_subclusters):
                best_subclusters = current_subclusters
                
        except Exception as e:
            continue  # Try next number of clusters
    
    # If no valid split found, use distance-based splitting
    if not best_subclusters:
        best_subclusters = distance_based_split(cluster_indices, cluster_coords, min_size, max_size, max_radius)
    
    return best_subclusters

def distance_based_split(cluster_indices, cluster_coords, min_size, max_size, max_radius):
    """
    Fallback splitting method using distance-based approach
    """
    # Find two points that are farthest apart
    max_dist = 0
    seed1_idx, seed2_idx = 0, 1
    
    for i in range(len(cluster_coords)):
        for j in range(i + 1, len(cluster_coords)):
            dist = haversine_distance(
                cluster_coords[i][0], cluster_coords[i][1],
                cluster_coords[j][0], cluster_coords[j][1]
            )
            if dist > max_dist:
                max_dist = dist
                seed1_idx, seed2_idx = i, j
    
    # Assign each customer to nearest seed
    subcluster1_indices, subcluster1_coords = [], []
    subcluster2_indices, subcluster2_coords = [], []
    
    for i, (idx, coord) in enumerate(zip(cluster_indices, cluster_coords)):
        dist1 = haversine_distance(coord[0], coord[1], cluster_coords[seed1_idx][0], cluster_coords[seed1_idx][1])
        dist2 = haversine_distance(coord[0], coord[1], cluster_coords[seed2_idx][0], cluster_coords[seed2_idx][1])
        
        if dist1 <= dist2:
            subcluster1_indices.append(idx)
            subcluster1_coords.append(coord)
        else:
            subcluster2_indices.append(idx)
            subcluster2_coords.append(coord)
    
    # Return valid subclusters
    result = []
    for sub_indices, sub_coords in [(subcluster1_indices, subcluster1_coords), (subcluster2_indices, subcluster2_coords)]:
        if min_size <= len(sub_indices) <= max_size:
            sub_radius = calculate_cluster_radius(np.array(sub_coords))
            if sub_radius <= max_radius:
                result.append((sub_indices, sub_coords))
    
    return result
    """Calculate maximum distance from centroid to any point in cluster"""
    if len(customers_coords) <= 1:
        return 0
    
    centroid = customers_coords.mean(axis=0)
    max_distance = 0
    
    for coord in customers_coords:
        distance = haversine_distance(centroid[0], centroid[1], coord[0], coord[1])
        max_distance = max(max_distance, distance)
    
    return max_distance

def constrained_clustering(customer_df, min_size=200, max_size=280, max_radius=20):
    """
    Create clusters with size and radius constraints during cluster formation
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1  # Initialize as unassigned
    
    unassigned_customers = customers_clean.copy()
    cluster_id = 0
    final_clusters = []
    
    print(f"Starting with {len(unassigned_customers)} customers")
    
    while len(unassigned_customers) >= min_size:
        # Start with a random seed customer
        seed_idx = unassigned_customers.index[0]
        seed_customer = unassigned_customers.loc[seed_idx]
        
        current_cluster = [seed_idx]
        current_coords = [(seed_customer['LAT_NUM'], seed_customer['LON_NUM'])]
        
        # Find all customers within initial radius of seed
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
        
        # Sort candidates by distance from seed
        candidates.sort(key=lambda x: x[2])
        
        # Greedily add customers while maintaining constraints
        for candidate_idx, candidate_customer, _ in candidates:
            if len(current_cluster) >= max_size:
                break
            
            # Test adding this customer
            test_coords = current_coords + [(candidate_customer['LAT_NUM'], candidate_customer['LON_NUM'])]
            test_radius = calculate_cluster_radius(np.array(test_coords))
            
            # Add if it doesn't violate radius constraint
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append((candidate_customer['LAT_NUM'], candidate_customer['LON_NUM']))
        
        # Handle cluster based on size
        if len(current_cluster) < min_size:
            # Too small - reject and remove seed
            unassigned_customers = unassigned_customers.drop([seed_idx])
            print(f"Rejected cluster with {len(current_cluster)} customers (too small)")
            
        elif len(current_cluster) <= max_size:
            # Perfect size - accept as is
            cluster_radius = calculate_cluster_radius(np.array(current_coords))
            
            for idx in current_cluster:
                customers_clean.loc[idx, 'cluster'] = cluster_id
            
            final_clusters.append({
                'cluster_id': cluster_id,
                'size': len(current_cluster),
                'radius': cluster_radius,
                'centroid_lat': np.mean([coord[0] for coord in current_coords]),
                'centroid_lon': np.mean([coord[1] for coord in current_coords])
            })
            
            unassigned_customers = unassigned_customers.drop(current_cluster)
            cluster_id += 1
            print(f"Cluster {cluster_id-1}: {len(current_cluster)} customers, radius: {cluster_radius:.2f} miles")
            
        else:
            # Too large - split into smaller clusters
            print(f"Splitting large cluster of {len(current_cluster)} customers")
            split_clusters = split_large_cluster(current_cluster, current_coords, customers_clean, min_size, max_size, max_radius)
            
            for split_cluster_indices, split_coords in split_clusters:
                if len(split_cluster_indices) >= min_size:
                    cluster_radius = calculate_cluster_radius(np.array(split_coords))
                    
                    for idx in split_cluster_indices:
                        customers_clean.loc[idx, 'cluster'] = cluster_id
                    
                    final_clusters.append({
                        'cluster_id': cluster_id,
                        'size': len(split_cluster_indices),
                        'radius': cluster_radius,
                        'centroid_lat': np.mean([coord[0] for coord in split_coords]),
                        'centroid_lon': np.mean([coord[1] for coord in split_coords])
                    })
                    
                    cluster_id += 1
                    print(f"  Split cluster {cluster_id-1}: {len(split_cluster_indices)} customers, radius: {cluster_radius:.2f} miles")
            
            # Remove all customers from current cluster (assigned or not)
            unassigned_customers = unassigned_customers.drop(current_cluster)
    
    print(f"Final result: {cluster_id} valid clusters, {len(unassigned_customers)} unassigned customers")
    
    return customers_clean, pd.DataFrame(final_clusters)

def assign_clusters_to_branches(clustered_customers, cluster_info, branch_df):
    """Assign each cluster to the nearest branch"""
    cluster_assignments = []
    
    for _, cluster in cluster_info.iterrows():
        min_distance = float('inf')
        assigned_branch = None
        
        for _, branch in branch_df.iterrows():
            distance = haversine_distance(
                cluster['centroid_lat'], cluster['centroid_lon'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            
            if distance < min_distance:
                min_distance = distance
                assigned_branch = branch['BRANCH_AU']
        
        cluster_assignments.append({
            'cluster_id': cluster['cluster_id'],
            'assigned_branch': assigned_branch,
            'distance_to_branch': min_distance,
            'cluster_size': cluster['size'],
            'cluster_radius': cluster['radius']
        })
    
    return pd.DataFrame(cluster_assignments)

def create_customer_portfolios(customer_df, branch_df):
    """
    Create customer portfolios with exact size and radius constraints
    """
    # Step 1: Create constrained clusters
    clustered_customers, cluster_info = constrained_clustering(
        customer_df, min_size=200, max_size=280, max_radius=20
    )
    
    if len(cluster_info) == 0:
        print("No valid clusters found with the given constraints")
        return pd.DataFrame(), pd.DataFrame()
    
    # Step 2: Assign clusters to branches
    cluster_branch_map = assign_clusters_to_branches(clustered_customers, cluster_info, branch_df)
    
    # Step 3: Merge customer data with branch assignments
    # Only include customers that were assigned to clusters
    assigned_customers = clustered_customers[clustered_customers['cluster'] != -1].copy()
    
    final_assignments = assigned_customers.merge(
        cluster_branch_map[['cluster_id', 'assigned_branch']], 
        left_on='cluster', 
        right_on='cluster_id', 
        how='left'
    )
    
    return final_assignments, cluster_branch_map

# Execute clustering
result_df, cluster_summary = create_customer_portfolios(customer_df, branch_df)

# Display results
if len(result_df) > 0:
    print(f"\nTotal customers clustered: {len(result_df)}")
    print(f"Number of clusters: {len(cluster_summary)}")
    print(f"Customers left unassigned: {len(customer_df) - len(result_df)}")
    
    print("\nCluster Summary:")
    print(cluster_summary[['cluster_id', 'assigned_branch', 'cluster_size', 'cluster_radius', 'distance_to_branch']])
    
    print("\nCustomer assignments preview:")
    print(result_df[['ECN', 'cluster', 'assigned_branch', 'BILLINGCITY', 'BILLINGSTATE']].head(10))
    
    # Portfolio sizes by branch
    portfolio_sizes = result_df.groupby('assigned_branch').size()
    print("\nPortfolio sizes by branch:")
    print(portfolio_sizes)
else:
    print("No customer portfolios could be created with the given constraints")
