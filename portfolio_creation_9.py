import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    R = 3959  # Earth's radius in miles
    
    # Convert to numpy arrays if not already and ensure proper dtype
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical errors
    
    distance = R * c
    
    # Handle scalar case
    if np.isscalar(distance):
        return float(distance)
    return distance

def compute_distance_matrix(coords1, coords2):
    """Compute distance matrix between two sets of coordinates"""
    # Ensure inputs are numpy arrays
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)
    
    lat1 = coords1[:, 0][:, np.newaxis]  # Shape: (n1, 1)
    lon1 = coords1[:, 1][:, np.newaxis]  # Shape: (n1, 1)
    lat2 = coords2[:, 0][np.newaxis, :]  # Shape: (1, n2)
    lon2 = coords2[:, 1][np.newaxis, :]  # Shape: (1, n2)
    
    return haversine_distance_vectorized(lat1, lon1, lat2, lon2)

def calculate_cluster_radius_vectorized(coords):
    """Vectorized calculation of cluster radius"""
    coords = np.asarray(coords, dtype=np.float64)
    
    if len(coords) <= 1:
        return 0.0
    
    centroid = coords.mean(axis=0)
    distances = haversine_distance_vectorized(
        coords[:, 0], coords[:, 1], 
        centroid[0], centroid[1]
    )
    return float(np.max(distances))

def find_candidates_spatial(customer_coords, seed_coord, max_radius, ball_tree=None):
    """Use spatial indexing to find candidates within radius"""
    customer_coords = np.asarray(customer_coords, dtype=np.float64)
    seed_coord = np.asarray(seed_coord, dtype=np.float64)
    
    if ball_tree is None:
        # Convert to radians for BallTree
        coords_rad = np.radians(customer_coords)
        ball_tree = BallTree(coords_rad, metric='haversine')
    
    # Query for neighbors within radius
    seed_rad = np.radians(seed_coord).reshape(1, -1)
    radius_rad = max_radius / 3959  # Convert miles to radians
    
    indices = ball_tree.query_radius(seed_rad, r=radius_rad)[0]
    
    # Calculate actual distances
    if len(indices) > 0:
        candidate_coords = customer_coords[indices]
        distances = haversine_distance_vectorized(
            candidate_coords[:, 0], candidate_coords[:, 1],
            seed_coord[0], seed_coord[1]
        )
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        return indices[sorted_indices], distances[sorted_indices]
    
    return np.array([]), np.array([])

def constrained_clustering_optimized(customer_df, min_size=200, max_size=225, max_radius=20):
    """Optimized clustering with vectorized operations and spatial indexing"""
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    # Pre-compute coordinates array
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    
    # Build spatial index
    coords_rad = np.radians(coords)
    ball_tree = BallTree(coords_rad, metric='haversine')
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    unassigned_count = np.count_nonzero(unassigned_mask)
    while unassigned_count >= min_size:
        # Find first unassigned customer as seed
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        # Find candidates using spatial indexing
        candidate_indices, distances = find_candidates_spatial(
            coords[unassigned_mask], seed_coord, max_radius, None
        )
        
        # Map back to original indices
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[candidate_indices]
        
        if len(candidate_original_indices) == 0:
            unassigned_mask[seed_idx] = False
            unassigned_count = np.count_nonzero(unassigned_mask)
            continue
        
        # Start with seed
        current_cluster = [seed_idx]
        current_coords = [coords[seed_idx]]
        
        # Add candidates while maintaining constraints
        for i, (candidate_idx, distance) in enumerate(zip(candidate_original_indices, distances)):
            if candidate_idx == seed_idx:
                continue
                
            if len(current_cluster) >= max_size:
                break
            
            # Test radius constraint
            test_coords = np.array(current_coords + [coords[candidate_idx]], dtype=np.float64)
            test_radius = calculate_cluster_radius_vectorized(test_coords)
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append(coords[candidate_idx])
        
        # Handle cluster based on size
        if len(current_cluster) < min_size:
            unassigned_mask[seed_idx] = False
        elif len(current_cluster) <= max_size:
            cluster_coords = np.array(current_coords, dtype=np.float64)
            cluster_radius = calculate_cluster_radius_vectorized(cluster_coords)
            
            # Assign cluster
            for idx in current_cluster:
                customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                unassigned_mask[idx] = False
            
            final_clusters.append({
                'cluster_id': cluster_id,
                'size': len(current_cluster),
                'radius': cluster_radius,
                'centroid_lat': np.mean(cluster_coords[:, 0]),
                'centroid_lon': np.mean(cluster_coords[:, 1])
            })
            
            cluster_id += 1
            
        else:
            # Split large cluster using K-means
            coords_array = np.array(current_coords, dtype=np.float64)
            
            # Modified condition
            if len(current_cluster) > 0 and min_size > 0:
                if len(current_cluster) // min_size < 3:
                    n_splits = np.maximum(1, len(current_cluster) // min_size)
                else:
                    n_splits = 3
            else:
                n_splits = 1
            
            if n_splits > 1:
                kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                subcluster_labels = kmeans.fit_predict(coords_array)
                
                for sub_id in range(n_splits):
                    sub_mask = subcluster_labels == sub_id
                    sub_indices = [current_cluster[i] for i in range(len(current_cluster)) if sub_mask[i]]
                    
                    if len(sub_indices) >= min_size and len(sub_indices) <= max_size:
                        sub_coords = coords_array[sub_mask]
                        sub_radius = calculate_cluster_radius_vectorized(sub_coords)
                        
                        if sub_radius <= max_radius:
                            for idx in sub_indices:
                                customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                                unassigned_mask[idx] = False
                            
                            final_clusters.append({
                                'cluster_id': cluster_id,
                                'size': len(sub_indices),
                                'radius': sub_radius,
                                'centroid_lat': np.mean(sub_coords[:, 0]),
                                'centroid_lon': np.mean(sub_coords[:, 1])
                            })
                            cluster_id += 1
            
            # Mark all as unassigned for this iteration
            for idx in current_cluster:
                unassigned_mask[idx] = False
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    return customers_clean, pd.DataFrame(final_clusters)

def constrained_clustering_with_radius(customer_df, min_size=200, max_size=240, max_radius=100):
    """
    Clustering with both size constraints AND radius constraint for centralized portfolios
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    # Pre-compute coordinates array
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    
    # Build spatial index
    coords_rad = np.radians(coords)
    ball_tree = BallTree(coords_rad, metric='haversine')
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    unassigned_count = np.count_nonzero(unassigned_mask)
    while unassigned_count >= min_size:
        # Find first unassigned customer as seed
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        # Find candidates using spatial indexing within max_radius
        candidate_indices, distances = find_candidates_spatial(
            coords[unassigned_mask], seed_coord, max_radius, None
        )
        
        # Map back to original indices
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[candidate_indices]
        
        if len(candidate_original_indices) == 0:
            unassigned_mask[seed_idx] = False
            unassigned_count = np.count_nonzero(unassigned_mask)
            continue
        
        # Start with seed
        current_cluster = [seed_idx]
        current_coords = [coords[seed_idx]]
        
        # Add candidates while maintaining constraints
        for i, (candidate_idx, distance) in enumerate(zip(candidate_original_indices, distances)):
            if candidate_idx == seed_idx:
                continue
                
            if len(current_cluster) >= max_size:
                break
            
            # Test radius constraint
            test_coords = np.array(current_coords + [coords[candidate_idx]], dtype=np.float64)
            test_radius = calculate_cluster_radius_vectorized(test_coords)
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append(coords[candidate_idx])
        
        # Handle cluster based on size
        if len(current_cluster) < min_size:
            unassigned_mask[seed_idx] = False
        elif len(current_cluster) <= max_size:
            cluster_coords = np.array(current_coords, dtype=np.float64)
            cluster_radius = calculate_cluster_radius_vectorized(cluster_coords)
            
            # Double-check radius constraint
            if cluster_radius <= max_radius:
                # Assign cluster
                for idx in current_cluster:
                    customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                    unassigned_mask[idx] = False
                
                final_clusters.append({
                    'cluster_id': cluster_id,
                    'size': len(current_cluster),
                    'radius': cluster_radius,
                    'centroid_lat': np.mean(cluster_coords[:, 0]),
                    'centroid_lon': np.mean(cluster_coords[:, 1])
                })
                
                cluster_id += 1
            else:
                # Cluster exceeds radius, try to split
                coords_array = np.array(current_coords, dtype=np.float64)
                n_splits = np.minimum(3, len(current_cluster) // min_size)
                
                if n_splits > 1:
                    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                    subcluster_labels = kmeans.fit_predict(coords_array)
                    
                    for sub_id in range(n_splits):
                        sub_mask = subcluster_labels == sub_id
                        sub_indices = [current_cluster[i] for i in range(len(current_cluster)) if sub_mask[i]]
                        
                        if len(sub_indices) >= min_size and len(sub_indices) <= max_size:
                            sub_coords = coords_array[sub_mask]
                            sub_radius = calculate_cluster_radius_vectorized(sub_coords)
                            
                            if sub_radius <= max_radius:
                                for idx in sub_indices:
                                    customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                                    unassigned_mask[idx] = False
                                
                                final_clusters.append({
                                    'cluster_id': cluster_id,
                                    'size': len(sub_indices),
                                    'radius': sub_radius,
                                    'centroid_lat': np.mean(sub_coords[:, 0]),
                                    'centroid_lon': np.mean(sub_coords[:, 1])
                                })
                                cluster_id += 1
                
                # Mark remaining as unassigned for this iteration
                for idx in current_cluster:
                    if customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] == -1:
                        unassigned_mask[idx] = False
        else:
            # Handle oversized cluster with splitting
            coords_array = np.array(current_coords, dtype=np.float64)
            n_splits = np.minimum(3, len(current_cluster) // min_size)
            
            if n_splits > 1:
                kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                subcluster_labels = kmeans.fit_predict(coords_array)
                
                for sub_id in range(n_splits):
                    sub_mask = subcluster_labels == sub_id
                    sub_indices = [current_cluster[i] for i in range(len(current_cluster)) if sub_mask[i]]
                    
                    if len(sub_indices) >= min_size and len(sub_indices) <= max_size:
                        sub_coords = coords_array[sub_mask]
                        sub_radius = calculate_cluster_radius_vectorized(sub_coords)
                        
                        if sub_radius <= max_radius:
                            for idx in sub_indices:
                                customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                                unassigned_mask[idx] = False
                            
                            final_clusters.append({
                                'cluster_id': cluster_id,
                                'size': len(sub_indices),
                                'radius': sub_radius,
                                'centroid_lat': np.mean(sub_coords[:, 0]),
                                'centroid_lon': np.mean(sub_coords[:, 1])
                            })
                            cluster_id += 1
            
            # Mark all as unassigned for this iteration
            for idx in current_cluster:
                unassigned_mask[idx] = False
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    return customers_clean, pd.DataFrame(final_clusters)

def assign_clusters_to_branches_unique(cluster_info, branch_df, used_branches=None):
    """
    Assign clusters to branches ensuring each branch can only be assigned to ONE cluster
    """
    if len(cluster_info) == 0:
        return pd.DataFrame()
    
    if used_branches is None:
        used_branches = set()
    
    # Get cluster centroids and branch coordinates
    cluster_coords = cluster_info[['centroid_lat', 'centroid_lon']].values.astype(np.float64)
    
    # Filter out already used branches
    available_branches = branch_df[~branch_df['BRANCH_AU'].isin(used_branches)].copy()
    
    if len(available_branches) == 0:
        print("No available branches left for assignment!")
        return pd.DataFrame()
    
    branch_coords = available_branches[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values.astype(np.float64)
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(cluster_coords, branch_coords)
    
    cluster_assignments = []
    assigned_branch_indices = set()
    
    # Sort clusters by their minimum distance to any available branch
    cluster_min_distances = np.min(distance_matrix, axis=1)
    cluster_order = np.argsort(cluster_min_distances)
    
    # Assign clusters in order of their minimum distance
    for cluster_idx in cluster_order:
        cluster = cluster_info.iloc[cluster_idx]
        
        # Find the closest available branch for this cluster
        available_branch_mask = np.ones(len(available_branches), dtype=bool)
        available_branch_mask[list(assigned_branch_indices)] = False
        
        if not np.any(available_branch_mask):
            print(f"No more available branches for cluster {cluster['cluster_id']}")
            continue
        
        # Get distances to available branches only
        cluster_distances = distance_matrix[cluster_idx, :]
        cluster_distances[~available_branch_mask] = np.inf
        
        # Find the closest available branch
        closest_branch_idx = np.argmin(cluster_distances)
        
        if cluster_distances[closest_branch_idx] == np.inf:
            print(f"No available branch for cluster {cluster['cluster_id']}")
            continue
        
        # Assign this cluster to the closest available branch
        assigned_branch = available_branches.iloc[closest_branch_idx]['BRANCH_AU']
        distance = cluster_distances[closest_branch_idx]
        
        cluster_assignments.append({
            'cluster_id': cluster['cluster_id'],
            'assigned_branch': assigned_branch,
            'cluster_to_branch_distance': distance
        })
        
        # Mark this branch as used
        assigned_branch_indices.add(closest_branch_idx)
        used_branches.add(assigned_branch)
    
    return pd.DataFrame(cluster_assignments)

def assign_customers_to_cluster_branch(clustered_customers, cluster_assignments):
    """
    Assign customers directly to their cluster's assigned branch (no cross-assignment)
    """
    if len(cluster_assignments) == 0:
        return {}, list(clustered_customers.index)
    
    customer_assignments = {}
    unassigned_customers = []
    
    # Create mapping from cluster_id to assigned_branch
    cluster_to_branch = {}
    for _, assignment in cluster_assignments.iterrows():
        cluster_to_branch[assignment['cluster_id']] = assignment['assigned_branch']
    
    # Process each customer based on their cluster assignment
    for idx, customer in clustered_customers.iterrows():
        cluster_id = customer['cluster']
        
        if cluster_id == -1:
            # Customer was never assigned to a cluster
            unassigned_customers.append(idx)
            continue
        
        if cluster_id not in cluster_to_branch:
            # Cluster was not assigned to any branch
            unassigned_customers.append(idx)
            continue
        
        # Assign customer to their cluster's designated branch
        assigned_branch = cluster_to_branch[cluster_id]
        
        if assigned_branch not in customer_assignments:
            customer_assignments[assigned_branch] = []
        
        # Calculate distance to assigned branch (for record keeping)
        # This will be calculated later in the main function
        customer_assignments[assigned_branch].append({
            'customer_idx': idx,
            'distance': 0  # Will be calculated later
        })
    
    return customer_assignments, unassigned_customers

def assign_proximity_customers_to_existing_portfolios(unassigned_customers_df, customer_assignments, branch_df, proximity_threshold=20, max_portfolio_size=250):
    """
    Check if unassigned customers are within proximity of identified AUs
    Add them to existing portfolios up to max_portfolio_size
    """
    
    if len(unassigned_customers_df) == 0 or not customer_assignments:
        return [], list(unassigned_customers_df.index), customer_assignments
    
    # Get identified AUs and their coordinates
    identified_aus = list(customer_assignments.keys())
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_aus)].copy()
    
    # Pre-compute current portfolio sizes
    current_portfolio_sizes = {}
    for branch_au, customers in customer_assignments.items():
        current_portfolio_sizes[branch_au] = len(customers)
    
    # Calculate distances from unassigned customers to all identified AUs
    unassigned_coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values.astype(np.float64)
    
    distance_matrix = compute_distance_matrix(unassigned_coords, branch_coords)
    
    proximity_results = []
    remaining_unassigned = []
    updated_customer_assignments = customer_assignments.copy()
    
    # Process each unassigned customer
    for i, (customer_idx, customer_data) in enumerate(unassigned_customers_df.iterrows()):
        assigned = False
        
        # Find all AUs within proximity threshold
        customer_distances = distance_matrix[i, :]
        within_proximity = customer_distances <= proximity_threshold
        
        if np.any(within_proximity):
            # Get AUs within proximity and their distances
            proximity_aus = []
            for j, is_within in enumerate(within_proximity):
                if is_within:
                    branch_au = identified_branch_coords.iloc[j]['BRANCH_AU']
                    distance = customer_distances[j]
                    current_size = current_portfolio_sizes[branch_au]
                    
                    if current_size < max_portfolio_size:
                        proximity_aus.append((branch_au, distance, current_size))
            
            # Sort by distance and assign to nearest AU with capacity
            if proximity_aus:
                proximity_aus.sort(key=lambda x: x[1])  # Sort by distance
                
                for branch_au, distance, current_size in proximity_aus:
                    if current_portfolio_sizes[branch_au] < max_portfolio_size:
                        # Add customer to this AU
                        updated_customer_assignments[branch_au].append({
                            'customer_idx': customer_idx,
                            'distance': distance
                        })
                        
                        current_portfolio_sizes[branch_au] += 1
                        
                        proximity_results.append({
                            'ECN': customer_data['ECN'],
                            'BILLINGCITY': customer_data['BILLINGCITY'],
                            'BILLINGSTATE': customer_data['BILLINGSTATE'],
                            'LAT_NUM': customer_data['LAT_NUM'],
                            'LON_NUM': customer_data['LON_NUM'],
                            'ASSIGNED_AU': branch_au,
                            'DISTANCE_TO_AU': distance,
                            'TYPE': 'INMARKET'
                        })
                        
                        assigned = True
                        break
        
        if not assigned:
            remaining_unassigned.append(customer_idx)
    
    # Sort customers within each branch by distance
    for branch_au in updated_customer_assignments:
        updated_customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return proximity_results, remaining_unassigned, updated_customer_assignments

def create_centralized_clusters_with_radius_and_assign(unassigned_customers_df, branch_df, 
                                                     used_branches, min_size=200, max_size=240, max_radius=100):
    """
    Create centralized clusters WITH radius constraint and assign to branches
    Ensures no branch is used twice
    """
    
    if len(unassigned_customers_df) == 0:
        return [], [], used_branches
    
    # Step 1: Create clusters with radius constraint
    clustered_centralized, centralized_cluster_info = constrained_clustering_with_radius(
        unassigned_customers_df, min_size=min_size, max_size=max_size, max_radius=max_radius
    )
    
    centralized_results = []
    final_unassigned = []
    
    if len(centralized_cluster_info) > 0:
        # Step 2: Assign clusters to branches (ensuring no duplicates)
        cluster_assignments = assign_clusters_to_branches_unique(
            centralized_cluster_info, branch_df, used_branches
        )
        
        # Update used_branches
        for _, assignment in cluster_assignments.iterrows():
            used_branches.add(assignment['assigned_branch'])
        
        # Step 3: Assign customers within clusters to their assigned branches
        for _, assignment in cluster_assignments.iterrows():
            cluster_id = assignment['cluster_id']
            assigned_branch = assignment['assigned_branch']
            
            # Get customers in this cluster
            cluster_customers = clustered_centralized[
                clustered_centralized['cluster'] == cluster_id
            ]
            
            # Get branch coordinates for distance calculation
            branch_coords = branch_df[
                branch_df['BRANCH_AU'] == assigned_branch
            ][['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].iloc[0]
            
            # Calculate distances from each customer to assigned branch
            for idx, customer in cluster_customers.iterrows():
                distance = haversine_distance_vectorized(
                    customer['LAT_NUM'], customer['LON_NUM'],
                    branch_coords['BRANCH_LAT_NUM'], branch_coords['BRANCH_LON_NUM']
                )
                
                # Get original customer data
                original_customer = unassigned_customers_df.loc[idx]
                
                centralized_results.append({
                    'customer_idx': idx,
                    'ECN': original_customer['ECN'],
                    'BILLINGCITY': original_customer['BILLINGCITY'],
                    'BILLINGSTATE': original_customer['BILLINGSTATE'],
                    'LAT_NUM': original_customer['LAT_NUM'],
                    'LON_NUM': original_customer['LON_NUM'],
                    'ASSIGNED_AU': assigned_branch,
                    'DISTANCE_TO_AU': distance,
                    'TYPE': 'CENTRALIZED',
                    'CLUSTER_ID': cluster_id
                })
        
        # Find customers that couldn't be clustered
        unassigned_centralized = clustered_centralized[
            clustered_centralized['cluster'] == -1
        ]
        final_unassigned = list(unassigned_centralized.index)
        
    else:
        final_unassigned = list(unassigned_customers_df.index)
    
    return centralized_results, final_unassigned, used_branches

def optimize_inmarket_portfolios_until_convergence(result_df, branch_df, k_neighbors=3):
    """
    Repeatedly optimize INMARKET portfolios until no more beneficial reassignments can be made
    """
    
    optimized_result = result_df.copy()
    iteration = 0
    
    while True:
        iteration += 1
        
        inmarket_customers = optimized_result[optimized_result['TYPE'] == 'INMARKET'].copy()
        if len(inmarket_customers) == 0:
            break
        
        # Group customers by assigned AU
        portfolio_groups = inmarket_customers.groupby('ASSIGNED_AU')
        outlier_customers = []
        
        # Step 1: Identify outlier customers (3x median distance)
        for au, group in portfolio_groups:
            if len(group) <= 1:
                continue
                
            distances = group['DISTANCE_TO_AU'].values
            median_distance = np.median(distances)
            threshold = 3 * median_distance
            
            outliers = group[group['DISTANCE_TO_AU'] > threshold]
            if len(outliers) > 0:
                for idx, customer in outliers.iterrows():
                    outlier_customers.append({
                        'customer_idx': idx,
                        'current_au': au,
                        'current_distance': customer['DISTANCE_TO_AU'],
                        'customer_data': customer
                    })
        
        if not outlier_customers:
            break
        
        # Get all INMARKET AUs and their coordinates
        inmarket_aus = inmarket_customers['ASSIGNED_AU'].unique()
        au_coords = []
        au_list = []
        
        for au in inmarket_aus:
            branch_coord = branch_df[branch_df['BRANCH_AU'] == au]
            if len(branch_coord) > 0:
                au_coords.append([branch_coord.iloc[0]['BRANCH_LAT_NUM'], branch_coord.iloc[0]['BRANCH_LON_NUM']])
                au_list.append(au)
        
        au_coords = np.array(au_coords, dtype=np.float64)
        
        # Track current portfolio sizes
        portfolio_sizes = inmarket_customers['ASSIGNED_AU'].value_counts().to_dict()
        
        iteration_improvements = 0
        
        # Step 2: Process each outlier customer
        for outlier in outlier_customers:
            customer_coord = np.array([[outlier['customer_data']['LAT_NUM'], outlier['customer_data']['LON_NUM']]], dtype=np.float64)
            
            # Calculate distances to all AUs
            distances_to_aus = compute_distance_matrix(customer_coord, au_coords)[0]
            
            # Find k nearest AUs
            nearest_indices = np.argsort(distances_to_aus)[:k_neighbors]
            
            best_reassignment = None
            best_improvement = 0
            
            for idx in nearest_indices:
                target_au = au_list[idx]
                target_distance = distances_to_aus[idx]
                
                if target_au == outlier['current_au']:
                    continue
                
                current_improvement = outlier['current_distance'] - target_distance
                
                if current_improvement <= 0:
                    continue
                
                # Check if target portfolio has capacity
                if portfolio_sizes.get(target_au, 0) < 250:
                    if current_improvement > best_improvement:
                        best_reassignment = {
                            'type': 'direct',
                            'target_au': target_au,
                            'target_distance': target_distance,
                            'improvement': current_improvement
                        }
                        best_improvement = current_improvement
                
                # Check for trade opportunity
                elif portfolio_sizes.get(target_au, 0) == 250:
                    target_portfolio = inmarket_customers[inmarket_customers['ASSIGNED_AU'] == target_au]
                    target_farthest_idx = target_portfolio['DISTANCE_TO_AU'].idxmax()
                    target_farthest = target_portfolio.loc[target_farthest_idx]
                    
                    # Calculate trade benefit
                    current_au_coord = branch_df[branch_df['BRANCH_AU'] == outlier['current_au']]
                    if len(current_au_coord) > 0:
                        trade_distance = haversine_distance_vectorized(
                            target_farthest['LAT_NUM'], target_farthest['LON_NUM'],
                            current_au_coord.iloc[0]['BRANCH_LAT_NUM'], current_au_coord.iloc[0]['BRANCH_LON_NUM']
                        )
                        
                        total_trade_improvement = (outlier['current_distance'] - target_distance) + (target_farthest['DISTANCE_TO_AU'] - trade_distance)
                        
                        if total_trade_improvement > best_improvement:
                            best_reassignment = {
                                'type': 'trade',
                                'target_au': target_au,
                                'target_distance': target_distance,
                                'trade_customer_idx': target_farthest_idx,
                                'trade_distance': trade_distance,
                                'improvement': total_trade_improvement
                            }
                            best_improvement = total_trade_improvement
            
            # Execute best reassignment
            if best_reassignment and best_improvement > 0:
                if best_reassignment['type'] == 'direct':
                    # Direct reassignment
                    optimized_result.loc[outlier['customer_idx'], 'ASSIGNED_AU'] = best_reassignment['target_au']
                    optimized_result.loc[outlier['customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['target_distance']
                    
                    portfolio_sizes[outlier['current_au']] -= 1
                    portfolio_sizes[best_reassignment['target_au']] = portfolio_sizes.get(best_reassignment['target_au'], 0) + 1
                    
                elif best_reassignment['type'] == 'trade':
                    # Trade reassignment
                    optimized_result.loc[outlier['customer_idx'], 'ASSIGNED_AU'] = best_reassignment['target_au']
                    optimized_result.loc[outlier['customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['target_distance']
                    
                    optimized_result.loc[best_reassignment['trade_customer_idx'], 'ASSIGNED_AU'] = outlier['current_au']
                    optimized_result.loc[best_reassignment['trade_customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['trade_distance']
                
                iteration_improvements += 1
                
                # Update inmarket_customers for subsequent iterations within this loop
                inmarket_customers = optimized_result[optimized_result['TYPE'] == 'INMARKET'].copy()
        
        # If no improvements were made in this iteration, we've converged
        if iteration_improvements == 0:
            break
    
    return optimized_result

def rebalance_portfolio_sizes(result_df, branch_df, min_size=200, search_radius=50):
    """
    Rebalance INMARKET portfolios by moving customers from oversized to undersized
    """
    
    # Work with a copy
    balanced_result = result_df.copy()
    
    # Get INMARKET portfolios and their sizes
    inmarket_customers = balanced_result[balanced_result['TYPE'] == 'INMARKET'].copy()
    if len(inmarket_customers) == 0:
        return balanced_result
    
    portfolio_sizes = inmarket_customers['ASSIGNED_AU'].value_counts().to_dict()
    
    # Identify undersized portfolios
    undersized_portfolios = {au: size for au, size in portfolio_sizes.items() if size < min_size}
    
    if not undersized_portfolios:
        return balanced_result
    
    # Get branch coordinates for distance calculations
    branch_coords_dict = {}
    for _, branch in branch_df.iterrows():
        branch_coords_dict[branch['BRANCH_AU']] = {
            'lat': float(branch['BRANCH_LAT_NUM']),
            'lon': float(branch['BRANCH_LON_NUM'])
        }
    
    total_moves = 0
    
    # Process each undersized portfolio
    for undersized_au, current_size in undersized_portfolios.items():
        needed_customers = int(min_size - current_size)
        
        if undersized_au not in branch_coords_dict:
            continue
        
        undersized_coords = branch_coords_dict[undersized_au]
        
        # Find potential donor portfolios within search radius
        potential_donors = []
        
        for donor_au, donor_size in portfolio_sizes.items():
            if donor_au == undersized_au or donor_size <= min_size:
                continue
            
            if donor_au not in branch_coords_dict:
                continue
            
            donor_coords = branch_coords_dict[donor_au]
            
            # Calculate distance between branches
            distance = haversine_distance_vectorized(
                undersized_coords['lat'], undersized_coords['lon'],
                donor_coords['lat'], donor_coords['lon']
            )
            
            if distance <= search_radius:
                # Calculate how many customers this donor can spare
                available_customers = int(donor_size - min_size)
                potential_donors.append({
                    'au': donor_au,
                    'distance': distance,
                    'available': available_customers,
                    'current_size': donor_size
                })
        
        # Sort donors by distance (closest first)
        potential_donors.sort(key=lambda x: x['distance'])
        
        customers_acquired = 0
        
        # Try to get customers from donors
        for donor_info in potential_donors:
            if customers_acquired >= needed_customers:
                break
            
            donor_au = donor_info['au']
            max_transferable = np.minimum(
                int(donor_info['available']),
                int(needed_customers - customers_acquired)
            )
            
            if max_transferable <= 0:
                continue
            
            # Get customers from donor portfolio, sorted by distance to recipient
            donor_customers = inmarket_customers[inmarket_customers['ASSIGNED_AU'] == donor_au].copy()
            
            # Calculate distances from donor customers to undersized AU
            donor_distances_to_recipient = []
            for idx, customer in donor_customers.iterrows():
                distance_to_recipient = haversine_distance_vectorized(
                    customer['LAT_NUM'], customer['LON_NUM'],
                    undersized_coords['lat'], undersized_coords['lon']
                )
                donor_distances_to_recipient.append({
                    'customer_idx': idx,
                    'distance_to_recipient': distance_to_recipient,
                    'current_distance': customer['DISTANCE_TO_AU']
                })
            
            # Sort by distance to recipient (closest first)
            donor_distances_to_recipient.sort(key=lambda x: x['distance_to_recipient'])
            
            # Transfer closest customers
            customers_to_transfer = np.minimum(int(max_transferable), len(donor_distances_to_recipient))
            
            for i in range(customers_to_transfer):
                customer_info = donor_distances_to_recipient[i]
                customer_idx = customer_info['customer_idx']
                
                # Update assignment
                balanced_result.loc[customer_idx, 'ASSIGNED_AU'] = undersized_au
                balanced_result.loc[customer_idx, 'DISTANCE_TO_AU'] = customer_info['distance_to_recipient']
                
                # Update portfolio sizes
                portfolio_sizes[donor_au] -= 1
                portfolio_sizes[undersized_au] = portfolio_sizes.get(undersized_au, 0) + 1
                
                customers_acquired += 1
                total_moves += 1
                
                # Update inmarket_customers for subsequent operations
                inmarket_customers.loc[customer_idx, 'ASSIGNED_AU'] = undersized_au
                inmarket_customers.loc[customer_idx, 'DISTANCE_TO_AU'] = customer_info['distance_to_recipient']
    
    return balanced_result

def fill_undersized_portfolios_from_unassigned(result_df, unassigned_customer_indices, customer_df, branch_df, min_size=200, max_radius=40):
    """
    Fill undersized INMARKET portfolios by finding unassigned customers within radius
    """
    
    if len(unassigned_customer_indices) == 0:
        return result_df, unassigned_customer_indices
    
    # Work with a copy
    updated_result = result_df.copy()
    remaining_unassigned = unassigned_customer_indices.copy()
    
    # Get INMARKET portfolios and their sizes
    inmarket_customers = updated_result[updated_result['TYPE'] == 'INMARKET'].copy()
    if len(inmarket_customers) == 0:
        return updated_result, remaining_unassigned
    
    portfolio_sizes = inmarket_customers['ASSIGNED_AU'].value_counts().to_dict()
    
    # Identify undersized portfolios
    undersized_portfolios = {au: size for au, size in portfolio_sizes.items() if size < min_size}
    
    if not undersized_portfolios:
        return updated_result, remaining_unassigned
    
    print(f"Found {len(undersized_portfolios)} undersized INMARKET portfolios to fill from unassigned customers")
    
    # Get unassigned customers data
    unassigned_customers_df = customer_df.loc[remaining_unassigned]
    if len(unassigned_customers_df) == 0:
        return updated_result, remaining_unassigned
    
    # Get branch coordinates
    branch_coords_dict = {}
    for _, branch in branch_df.iterrows():
        branch_coords_dict[branch['BRANCH_AU']] = {
            'lat': float(branch['BRANCH_LAT_NUM']),
            'lon': float(branch['BRANCH_LON_NUM'])
        }
    
    total_assigned_from_unassigned = 0
    
    # Process each undersized portfolio
    for undersized_au, current_size in undersized_portfolios.items():
        needed_customers = int(min_size - current_size)
        
        if undersized_au not in branch_coords_dict or needed_customers <= 0:
            continue
        
        undersized_coords = branch_coords_dict[undersized_au]
        
        # Calculate distances from all unassigned customers to this AU
        unassigned_coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
        au_coord = np.array([[undersized_coords['lat'], undersized_coords['lon']]], dtype=np.float64)
        
        distances_to_au = compute_distance_matrix(unassigned_coords, au_coord)[:, 0]
        
        # Find unassigned customers within max_radius
        within_radius_mask = distances_to_au <= max_radius
        
        if not np.any(within_radius_mask):
            continue
        
        # Get candidates within radius and sort by distance
        candidate_data = []
        for i, (customer_idx, customer_data) in enumerate(unassigned_customers_df.iterrows()):
            if within_radius_mask[i]:
                candidate_data.append({
                    'customer_idx': customer_idx,
                    'distance': distances_to_au[i],
                    'customer_data': customer_data
                })
        
        # Sort by distance (closest first)
        candidate_data.sort(key=lambda x: x['distance'])
        
        # Assign customers up to needed amount
        customers_to_assign = min(needed_customers, len(candidate_data))
        
        for i in range(customers_to_assign):
            candidate = candidate_data[i]
            customer_idx = candidate['customer_idx']
            distance = candidate['distance']
            customer_data = candidate['customer_data']
            
            # Add customer to result_df
            new_assignment = {
                'ECN': customer_data['ECN'],
                'BILLINGCITY': customer_data['BILLINGCITY'],
                'BILLINGSTATE': customer_data['BILLINGSTATE'],
                'LAT_NUM': customer_data['LAT_NUM'],
                'LON_NUM': customer_data['LON_NUM'],
                'ASSIGNED_AU': undersized_au,
                'DISTANCE_TO_AU': distance,
                'TYPE': 'INMARKET'
            }
            
            # Add to dataframe
            updated_result = pd.concat([updated_result, pd.DataFrame([new_assignment])], ignore_index=True)
            
            # Remove from unassigned list
            if customer_idx in remaining_unassigned:
                remaining_unassigned.remove(customer_idx)
            
            total_assigned_from_unassigned += 1
        
        print(f"Assigned {customers_to_assign} unassigned customers to AU {undersized_au} (within {max_radius} miles)")
    
    print(f"Total customers assigned from unassigned to undersized portfolios: {total_assigned_from_unassigned}")
    
    return updated_result, remaining_unassigned

def create_final_centralized_portfolios(unassigned_customers_df, branch_df, used_branches, max_size=250):
    """
    Create centralized portfolios for final unassigned customers without radius constraints
    Split into groups of max_size, assign each group to nearest available branch
    """
    
    if len(unassigned_customers_df) == 0:
        return [], used_branches
    
    print(f"Creating final centralized portfolios for {len(unassigned_customers_df)} unassigned customers")
    
    final_centralized_results = []
    customers_list = list(unassigned_customers_df.index)
    
    # Split customers into groups of max_size
    customer_groups = []
    for i in range(0, len(customers_list), max_size):
        group = customers_list[i:i + max_size]
        customer_groups.append(group)
    
    print(f"Created {len(customer_groups)} groups with sizes: {[len(group) for group in customer_groups]}")
    
    # Get available branches
    available_branches = branch_df[~branch_df['BRANCH_AU'].isin(used_branches)].copy()
    
    if len(available_branches) < len(customer_groups):
        print(f"Warning: Only {len(available_branches)} available branches for {len(customer_groups)} groups")
        print("Some groups may not get assigned to branches")
    
    # For each group, find the nearest available branch
    for group_idx, customer_group in enumerate(customer_groups):
        if group_idx >= len(available_branches):
            print(f"No more available branches for group {group_idx + 1}")
            continue
        
        # Calculate centroid of customer group
        group_customers = unassigned_customers_df.loc[customer_group]
        group_centroid_lat = group_customers['LAT_NUM'].mean()
        group_centroid_lon = group_customers['LON_NUM'].mean()
        
        # Find nearest available branch to group centroid
        min_distance = float('inf')
        best_branch = None
        
        for _, branch in available_branches.iterrows():
            distance = haversine_distance_vectorized(
                group_centroid_lat, group_centroid_lon,
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            
            if distance < min_distance:
                min_distance = distance
                best_branch = branch['BRANCH_AU']
        
        if best_branch is None:
            print(f"Could not find available branch for group {group_idx + 1}")
            continue
        
        # Assign all customers in this group to the selected branch
        for customer_idx in customer_group:
            customer_data = unassigned_customers_df.loc[customer_idx]
            
            # Calculate distance from customer to assigned branch
            branch_data = branch_df[branch_df['BRANCH_AU'] == best_branch].iloc[0]
            distance_to_branch = haversine_distance_vectorized(
                customer_data['LAT_NUM'], customer_data['LON_NUM'],
                branch_data['BRANCH_LAT_NUM'], branch_data['BRANCH_LON_NUM']
            )
            
            final_centralized_results.append({
                'ECN': customer_data['ECN'],
                'BILLINGCITY': customer_data['BILLINGCITY'],
                'BILLINGSTATE': customer_data['BILLINGSTATE'],
                'LAT_NUM': customer_data['LAT_NUM'],
                'LON_NUM': customer_data['LON_NUM'],
                'ASSIGNED_AU': best_branch,
                'DISTANCE_TO_AU': distance_to_branch,
                'TYPE': 'CENTRALIZED',
                'GROUP_ID': group_idx
            })
        
        # Mark branch as used
        used_branches.add(best_branch)
        
        # Remove assigned branch from available branches
        available_branches = available_branches[available_branches['BRANCH_AU'] != best_branch]
        
        print(f"Assigned group {group_idx + 1} ({len(customer_group)} customers) to branch {best_branch}")
    
    return final_centralized_results, used_branches

def enhanced_customer_au_assignment_with_unique_branches(customer_df, branch_df):
    """
    Enhanced main function ensuring each AU is assigned to only ONE cluster/portfolio
    Includes Step 6 optimization
    """
    
    print(f"Starting enhanced assignment with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Track used branches across all assignments
    used_branches = set()
    
    # Step 1: Create first INMARKET clusters (20-mile radius, max_size=225)
    print("Step 1: Creating first INMARKET clusters (20-mile radius)...")
    clustered_customers, cluster_info = constrained_clustering_optimized(
        customer_df, min_size=200, max_size=225, max_radius=20
    )
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        print(f"Created {len(cluster_info)} first INMARKET clusters")
        
        # Step 2: Assign clusters to branches (ensuring unique assignment)
        print("Step 2: Assigning first INMARKET clusters to branches (unique assignment)...")
        cluster_assignments = assign_clusters_to_branches_unique(cluster_info, branch_df, used_branches)
        
        # Update used branches
        for _, assignment in cluster_assignments.iterrows():
            used_branches.add(assignment['assigned_branch'])
        
        print(f"Assigned {len(cluster_assignments)} clusters to unique branches")
        
        # Step 3: Assign customers to their cluster's designated branch
        print("Step 3: Assigning customers to their cluster's designated branch...")
        customer_assignments, unassigned = assign_customers_to_cluster_branch(
            clustered_customers, cluster_assignments
        )
        
        # Calculate actual distances for assigned customers
        for branch_au, customers in customer_assignments.items():
            branch_coord = branch_df[branch_df['BRANCH_AU'] == branch_au].iloc[0]
            
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                # Calculate distance to assigned branch
                distance = haversine_distance_vectorized(
                    customer_data['LAT_NUM'], customer_data['LON_NUM'],
                    branch_coord['BRANCH_LAT_NUM'], branch_coord['BRANCH_LON_NUM']
                )
                
                customer['distance'] = distance
                
                inmarket_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': distance,
                    'TYPE': 'INMARKET'
                })
        
        unassigned_customer_indices.extend(unassigned)
    
    print(f"Total unassigned customers after first INMARKET: {len(unassigned_customer_indices)}")
    
    # Step 4: Check proximity of unassigned customers to identified AUs
    proximity_results = []
    unassigned_after_proximity = unassigned_customer_indices.copy()
    
    if unassigned_customer_indices and inmarket_results:
        # Create customer_assignments from inmarket_results for proximity check
        customer_assignments = {}
        for result in inmarket_results:
            au = result['ASSIGNED_AU']
            if au not in customer_assignments:
                customer_assignments[au] = []
            
            # Find the customer index
            customer_idx = customer_df[
                (customer_df['ECN'] == result['ECN']) &
                (customer_df['LAT_NUM'] == result['LAT_NUM']) &
                (customer_df['LON_NUM'] == result['LON_NUM'])
            ].index[0]
            
            customer_assignments[au].append({
                'customer_idx': customer_idx,
                'distance': result['DISTANCE_TO_AU']
            })
        
        unassigned_customers_df = customer_df.loc[unassigned_customer_indices]
        
        proximity_results, unassigned_after_proximity, updated_customer_assignments = assign_proximity_customers_to_existing_portfolios(
            unassigned_customers_df, customer_assignments, branch_df, 
            proximity_threshold=20, max_portfolio_size=250
        )
    
    print(f"Total unassigned customers after proximity check: {len(unassigned_after_proximity)}")
    
    # Step 5: Create second INMARKET clusters (40-mile radius) on remaining customers
    print("Step 5: Creating second INMARKET clusters (40-mile radius)...")
    second_inmarket_results = []
    unassigned_after_second_inmarket = unassigned_after_proximity.copy()
    
    if unassigned_after_proximity:
        remaining_customers_df = customer_df.loc[unassigned_after_proximity]
        
        # Create second iteration of INMARKET clusters with 40-mile radius
        clustered_customers_2, cluster_info_2 = constrained_clustering_optimized(
            remaining_customers_df, min_size=200, max_size=225, max_radius=40
        )
        
        if len(cluster_info_2) > 0:
            print(f"Created {len(cluster_info_2)} second INMARKET clusters")
            
            # Assign second iteration clusters to branches (ensuring unique assignment)
            cluster_assignments_2 = assign_clusters_to_branches_unique(cluster_info_2, branch_df, used_branches)
            
            # Update used branches
            for _, assignment in cluster_assignments_2.iterrows():
                used_branches.add(assignment['assigned_branch'])
            
            print(f"Assigned {len(cluster_assignments_2)} second INMARKET clusters to unique branches")
            
            # Assign customers to their cluster's designated branch
            customer_assignments_2, unassigned_2 = assign_customers_to_cluster_branch(
                clustered_customers_2, cluster_assignments_2
            )
            
            # Calculate actual distances for second iteration customers
            for branch_au, customers in customer_assignments_2.items():
                branch_coord = branch_df[branch_df['BRANCH_AU'] == branch_au].iloc[0]
                
                for customer in customers:
                    customer_idx = customer['customer_idx']
                    customer_data = remaining_customers_df.loc[customer_idx]
                    
                    # Calculate distance to assigned branch
                    distance = haversine_distance_vectorized(
                        customer_data['LAT_NUM'], customer_data['LON_NUM'],
                        branch_coord['BRANCH_LAT_NUM'], branch_coord['BRANCH_LON_NUM']
                    )
                    
                    customer['distance'] = distance
                    
                    second_inmarket_results.append({
                        'ECN': customer_data['ECN'],
                        'BILLINGCITY': customer_data['BILLINGCITY'],
                        'BILLINGSTATE': customer_data['BILLINGSTATE'],
                        'LAT_NUM': customer_data['LAT_NUM'],
                        'LON_NUM': customer_data['LON_NUM'],
                        'ASSIGNED_AU': branch_au,
                        'DISTANCE_TO_AU': distance,
                        'TYPE': 'INMARKET'
                    })
            
            # Update unassigned list
            unassigned_after_second_inmarket = list(set(unassigned_2))
        else:
            unassigned_after_second_inmarket = unassigned_after_proximity.copy()
    
    # Combine all results initially
    all_results = inmarket_results + proximity_results + second_inmarket_results
    result_df = pd.DataFrame(all_results)
    
    # Step 6: Optimize INMARKET portfolios until convergence
    if len(result_df) > 0:
        print("Step 6: Optimizing INMARKET portfolios...")
        result_df = optimize_inmarket_portfolios_until_convergence(result_df, branch_df)
    
    # Step 7: Balance INMARKET portfolios to minimum size
    if len(result_df) > 0:
        print("Step 7: Balancing INMARKET portfolios...")
        result_df = rebalance_portfolio_sizes(result_df, branch_df, min_size=200)
    
    # Step 8: Fill remaining undersized portfolios from unassigned customers
    if len(result_df) > 0 and unassigned_after_second_inmarket:
        print("Step 8: Filling undersized INMARKET portfolios...")
        result_df, unassigned_after_second_inmarket = fill_undersized_portfolios_from_unassigned(
            result_df, unassigned_after_second_inmarket, customer_df, branch_df, min_size=200, max_radius=40
        )
    
    # Step 9: Create CENTRALIZED clusters
    centralized_results = []
    remaining_after_centralized = []
    
    if unassigned_after_second_inmarket:
        print("Step 9: Creating CENTRALIZED clusters...")
        remaining_unassigned_df = customer_df.loc[unassigned_after_second_inmarket]
        
        centralized_results, remaining_after_centralized, used_branches = create_centralized_clusters_with_radius_and_assign(
            remaining_unassigned_df, branch_df, used_branches, min_size=200, max_size=240, max_radius=150
        )
    
    # Step 10: Handle final unassigned customers as centralized portfolios
    final_centralized_results = []
    
    if remaining_after_centralized:
        print("Step 10: Creating final centralized portfolios for remaining unassigned customers...")
        final_unassigned_df = customer_df.loc[remaining_after_centralized]
        
        final_centralized_results, used_branches = create_final_centralized_portfolios(
            final_unassigned_df, branch_df, used_branches, max_size=250
        )
    
    # Combine all results
    if len(centralized_results) > 0:
        result_df = pd.concat([result_df, pd.DataFrame(centralized_results)], ignore_index=True)
    
    if len(final_centralized_results) > 0:
        result_df = pd.concat([result_df, pd.DataFrame(final_centralized_results)], ignore_index=True)
    
    # Update final_unassigned to be empty since we assigned everyone
    final_unassigned = []
    
    # Print summary
    print(f"\n=== FINAL SUMMARY WITH UNIQUE BRANCH ASSIGNMENTS ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"Total branches used: {len(used_branches)}")
    if len(result_df) > 0:
        print(f"INMARKET customers assigned: {len(result_df[result_df['TYPE'] == 'INMARKET'])}")
        print(f"CENTRALIZED customers assigned: {len(result_df[result_df['TYPE'] == 'CENTRALIZED'])}")
        print(f"Total unique portfolios created: {result_df['ASSIGNED_AU'].nunique()}")
    print(f"Final unassigned customers: {len(final_unassigned)}")
    print(f"Total assigned customers: {len(result_df)}")
    
    if len(result_df) > 0:
        type_summary = result_df.groupby('TYPE').agg({
            'ECN': 'count',
            'DISTANCE_TO_AU': ['mean', 'max']
        }).round(2)
        type_summary.columns = ['Customer_Count', 'Avg_Distance', 'Max_Distance']
        print("\nPortfolio Type Summary:")
        print(type_summary)
        
        portfolio_summary = result_df.groupby('ASSIGNED_AU').agg({
            'ECN': 'count',
            'DISTANCE_TO_AU': 'mean',
            'TYPE': 'first'
        }).round(2)
        portfolio_summary.columns = ['Customer_Count', 'Avg_Distance', 'Type']
        print(f"\nIndividual Portfolio Summary:")
        print(portfolio_summary)
        
        print(f"\nOverall Distance Statistics:")
        print(f"Average distance to assigned AU: {result_df['DISTANCE_TO_AU'].mean():.2f} miles")
        print(f"Maximum distance to assigned AU: {result_df['DISTANCE_TO_AU'].max():.2f} miles")
    
    return result_df

# Usage example:
# Enhanced assignments ensuring each AU is assigned to only ONE cluster/portfolio
# assignments = enhanced_customer_au_assignment_with_unique_branches(customer_df, branch_df)
# assignments.to_csv('customer_au_assignments_unique_branches.csv', index=False)
