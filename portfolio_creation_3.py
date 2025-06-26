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
    
    print(f"Clustering complete: {cluster_id} valid clusters, {len(unassigned_customers)} unassigned customers")
    
    return customers_clean, pd.DataFrame(final_clusters)

def identify_nearest_branches(cluster_info, branch_df):
    """
    For each cluster, identify the nearest branch
    Returns unique branches that are nearest to at least one cluster
    """
    cluster_branch_mapping = []
    identified_branches = set()
    
    for _, cluster in cluster_info.iterrows():
        min_distance = float('inf')
        nearest_branch = None
        
        for _, branch in branch_df.iterrows():
            distance = haversine_distance(
                cluster['centroid_lat'], cluster['centroid_lon'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_branch = branch['BRANCH_AU']
        
        cluster_branch_mapping.append({
            'cluster_id': cluster['cluster_id'],
            'nearest_branch': nearest_branch,
            'distance_to_branch': min_distance
        })
        
        identified_branches.add(nearest_branch)
    
    print(f"Identified {len(identified_branches)} unique branches nearest to clusters:")
    print(list(identified_branches))
    
    return pd.DataFrame(cluster_branch_mapping), identified_branches

def assign_customers_to_identified_branches(customer_df, identified_branches, branch_df, max_distance=20):
    """
    For each identified branch, find all customers within max_distance miles
    and assign them directly to that branch
    """
    # Get clean customer data
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['assigned_branch'] = None
    customers_clean['distance_to_branch'] = None
    
    # Filter branches to only identified ones
    relevant_branches = branch_df[branch_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    assignment_summary = []
    
    for _, branch in relevant_branches.iterrows():
        branch_customers = []
        
        print(f"\nProcessing branch: {branch['BRANCH_AU']}")
        
        # Find all customers within max_distance of this branch
        for idx, customer in customers_clean.iterrows():
            # Skip if customer already assigned
            if pd.notna(customer['assigned_branch']):
                continue
                
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            
            if distance <= max_distance:
                # Assign customer to this branch
                customers_clean.loc[idx, 'assigned_branch'] = branch['BRANCH_AU']
                customers_clean.loc[idx, 'distance_to_branch'] = distance
                branch_customers.append(idx)
        
        assignment_summary.append({
            'branch': branch['BRANCH_AU'],
            'customers_assigned': len(branch_customers),
            'branch_lat': branch['BRANCH_LAT_NUM'],
            'branch_lon': branch['BRANCH_LON_NUM']
        })
        
        print(f"  Assigned {len(branch_customers)} customers to branch {branch['BRANCH_AU']}")
    
    return customers_clean, pd.DataFrame(assignment_summary)

def create_customer_portfolios(customer_df, branch_df):
    """
    Modified approach:
    1. Create constrained clusters
    2. Identify branches nearest to cluster centroids
    3. For identified branches only, assign customers within 20 miles
    """
    print("=== STEP 1: Creating constrained clusters ===")
    # Step 1: Create constrained clusters
    clustered_customers, cluster_info = constrained_clustering(
        customer_df, min_size=200, max_size=280, max_radius=20
    )
    
    if len(cluster_info) == 0:
        print("No valid clusters found with the given constraints")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"\n=== STEP 2: Identifying nearest branches to {len(cluster_info)} clusters ===")
    # Step 2: Identify which branches are nearest to the cluster centroids
    cluster_branch_mapping, identified_branches = identify_nearest_branches(cluster_info, branch_df)
    
    print(f"\n=== STEP 3: Assigning customers to {len(identified_branches)} identified branches ===")
    # Step 3: For identified branches only, assign customers within 20 miles
    final_assignments, branch_summary = assign_customers_to_identified_branches(
        customer_df, identified_branches, branch_df, max_distance=20
    )
    
    return final_assignments, cluster_branch_mapping, branch_summary

# Execute the modified clustering and assignment
result_df, cluster_branch_map, branch_portfolio_summary = create_customer_portfolios(customer_df, branch_df)

# Display results
if len(result_df) > 0:
    assigned_customers = result_df[pd.notna(result_df['assigned_branch'])]
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total customers processed: {len(result_df)}")
    print(f"Customers assigned to branches: {len(assigned_customers)}")
    print(f"Customers left unassigned: {len(result_df) - len(assigned_customers)}")
    
    print(f"\nNumber of clusters created: {len(cluster_branch_map)}")
    print(f"Number of branches identified: {len(branch_portfolio_summary)}")
    
    print("\nCluster to Branch Mapping:")
    print(cluster_branch_map)
    
    print("\nBranch Portfolio Summary:")
    print(branch_portfolio_summary)
    
    print("\nCustomer assignments preview:")
    print(assigned_customers[['ECN', 'assigned_branch', 'distance_to_branch', 'BILLINGCITY', 'BILLINGSTATE']].head(10))
    
    # Portfolio sizes by branch
    portfolio_sizes = assigned_customers.groupby('assigned_branch').size()
    print("\nFinal portfolio sizes by branch:")
    print(portfolio_sizes)
else:
    print("No customer portfolios could be created with the given constraints")
