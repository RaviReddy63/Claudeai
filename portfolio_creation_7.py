def optimize_inmarket_portfolios_until_convergence(result_df, branch_df, k_neighbors=3):
    """
    Repeatedly optimize INMARKET portfolios until no more beneficial reassignments can be made
    """
    print(f"\nOptimizing INMARKET portfolios until convergence...")
    
    optimized_result = result_df.copy()
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n--- Optimization Iteration {iteration} ---")
        
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
            print("No outlier customers found - optimization converged")
            break
        
        print(f"Found {len(outlier_customers)} outlier customers to evaluate")
        
        # Get all INMARKET AUs and their coordinates
        inmarket_aus = inmarket_customers['ASSIGNED_AU'].unique()
        au_coords = []
        au_list = []
        
        for au in inmarket_aus:
            branch_coord = branch_df[branch_df['BRANCH_AU'] == au]
            if len(branch_coord) > 0:
                au_coords.append([branch_coord.iloc[0]['BRANCH_LAT_NUM'], branch_coord.iloc[0]['BRANCH_LON_NUM']])
                au_list.append(au)
        
        au_coords = np.array(au_coords)
        
        # Track current portfolio sizes
        portfolio_sizes = inmarket_customers['ASSIGNED_AU'].value_counts().to_dict()
        
        iteration_improvements = 0
        
        # Step 2: Process each outlier customer
        for outlier in outlier_customers:
            customer_coord = np.array([[outlier['customer_data']['LAT_NUM'], outlier['customer_data']['LON_NUM']]])
            
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
        
        print(f"Iteration {iteration}: {iteration_improvements} customers reassigned")
        
        # If no improvements were made in this iteration, we've converged
        if iteration_improvements == 0:
            print("No improvements found - optimization converged")
            break
    
    print(f"Optimization complete after {iteration} iterations")
    return optimized_resultimport pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    R = 3959  # Earth's radius in miles
    
    # Convert to numpy arrays if not already
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def compute_distance_matrix(coords1, coords2):
    """Compute distance matrix between two sets of coordinates"""
    lat1 = coords1[:, 0][:, np.newaxis]  # Shape: (n1, 1)
    lon1 = coords1[:, 1][:, np.newaxis]  # Shape: (n1, 1)
    lat2 = coords2[:, 0][np.newaxis, :]  # Shape: (1, n2)
    lon2 = coords2[:, 1][np.newaxis, :]  # Shape: (1, n2)
    
    return haversine_distance_vectorized(lat1, lon1, lat2, lon2)

def calculate_cluster_radius_vectorized(coords):
    """Vectorized calculation of cluster radius"""
    if len(coords) <= 1:
        return 0
    
    centroid = coords.mean(axis=0)
    distances = haversine_distance_vectorized(
        coords[:, 0], coords[:, 1], 
        centroid[0], centroid[1]
    )
    return np.max(distances)

def find_candidates_spatial(customer_coords, seed_coord, max_radius, ball_tree=None):
    """Use spatial indexing to find candidates within radius"""
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
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values
    
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
            test_coords = np.array(current_coords + [coords[candidate_idx]])
            test_radius = calculate_cluster_radius_vectorized(test_coords)
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append(coords[candidate_idx])
        
        # Handle cluster based on size
        if len(current_cluster) < min_size:
            unassigned_mask[seed_idx] = False
        elif len(current_cluster) <= max_size:
            cluster_coords = np.array(current_coords)
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
            coords_array = np.array(current_coords)
            
            # Modified condition
            if len(current_cluster) > 0 and min_size > 0:
                if len(current_cluster) // min_size < 3:
                    n_splits = len(current_cluster) // min_size
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

def constrained_clustering_no_radius(customer_df, min_size=200, max_size=240):
    """
    Clustering with size constraints but NO radius constraint
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    # Pre-compute coordinates array
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values
    
    # Build spatial index for efficient neighbor finding
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
        
        # Find ALL unassigned customers and sort by distance to seed
        unassigned_coords = coords[unassigned_mask]
        unassigned_positions = np.where(unassigned_mask)[0]
        
        # Calculate distances to seed
        distances = haversine_distance_vectorized(
            unassigned_coords[:, 0], unassigned_coords[:, 1],
            seed_coord[0], seed_coord[1]
        )
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        
        # Take up to max_size closest customers
        if max_size < len(sorted_indices):
            cluster_size = max_size
        else:
            cluster_size = len(sorted_indices)
        selected_local_indices = sorted_indices[:cluster_size]
        selected_original_indices = unassigned_positions[selected_local_indices]
        
        # Create cluster (no radius constraint)
        current_cluster = list(selected_original_indices)
        current_coords = coords[selected_original_indices]
        
        # Only create cluster if it meets minimum size
        if len(current_cluster) >= min_size:
            # Calculate cluster statistics (for reporting, no constraint)
            centroid = current_coords.mean(axis=0)
            cluster_radius = calculate_cluster_radius_vectorized(current_coords)
            
            # Assign cluster
            for idx in current_cluster:
                customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                unassigned_mask[idx] = False
            
            final_clusters.append({
                'cluster_id': cluster_id,
                'size': len(current_cluster),
                'radius': cluster_radius,  # For reporting only
                'centroid_lat': centroid[0],
                'centroid_lon': centroid[1]
            })
            
            cluster_id += 1
        else:
            # Not enough customers for minimum cluster, stop
            break
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    return customers_clean, pd.DataFrame(final_clusters)

def assign_clusters_to_branches_vectorized(cluster_info, branch_df):
    """Vectorized cluster to branch assignment"""
    if len(cluster_info) == 0:
        return pd.DataFrame()
    
    # Get cluster centroids and branch coordinates
    cluster_coords = cluster_info[['centroid_lat', 'centroid_lon']].values
    branch_coords = branch_df[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(cluster_coords, branch_coords)
    
    # Find nearest branch for each cluster
    nearest_branch_indices = np.argmin(distance_matrix, axis=1)
    min_distances = np.min(distance_matrix, axis=1)
    
    cluster_assignments = []
    for i, cluster in cluster_info.iterrows():
        branch_idx = nearest_branch_indices[i]
        assigned_branch = branch_df.iloc[branch_idx]['BRANCH_AU']
        distance = min_distances[i]
        
        cluster_assignments.append({
            'cluster_id': cluster['cluster_id'],
            'assigned_branch': assigned_branch,
            'cluster_to_branch_distance': distance
        })
    
    return pd.DataFrame(cluster_assignments)

def greedy_assign_customers_to_branches(clustered_customers, cluster_assignments, branch_df, max_distance=20, max_customers_per_branch=225):
    """
    Fast greedy assignment instead of Hungarian algorithm
    Assigns customers to nearest available branch with capacity
    """
    
    identified_branches = cluster_assignments['assigned_branch'].unique()
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    print(f"Identified branches: {list(identified_branches)}")
    print(f"Available branch coordinates: {len(identified_branch_coords)}")
    
    customers_to_assign = clustered_customers[clustered_customers['cluster'] != -1].copy()
    print(f"Customers to assign: {len(customers_to_assign)}")
    
    if len(customers_to_assign) == 0 or len(identified_branch_coords) == 0:
        return {}, list(customers_to_assign.index)
    
    # Pre-compute distance matrix
    customer_coords = customers_to_assign[['LAT_NUM', 'LON_NUM']].values
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    
    print(f"Computing distance matrix: {len(customer_coords)} x {len(branch_coords)}")
    distance_matrix = compute_distance_matrix(customer_coords, branch_coords)
    
    # Apply distance constraint
    distance_matrix[distance_matrix > max_distance] = np.inf
    
    customer_indices = list(customers_to_assign.index)
    branch_aus = list(identified_branch_coords['BRANCH_AU'])
    
    # Initialize branch capacities
    branch_capacity = {branch_au: max_customers_per_branch for branch_au in branch_aus}
    customer_assignments = {branch_au: [] for branch_au in branch_aus}
    unassigned_customers = []
    
    print("Applying greedy assignment...")
    
    # Create list of (customer_idx, branch_idx, distance) sorted by distance
    assignment_candidates = []
    for i, customer_idx in enumerate(customer_indices):
        for j, branch_au in enumerate(branch_aus):
            distance = distance_matrix[i, j]
            if distance < np.inf:  # Only consider valid assignments
                assignment_candidates.append((i, customer_idx, j, branch_au, distance))
    
    # Sort by distance (greedy: assign closest first)
    assignment_candidates.sort(key=lambda x: x[4])
    
    print(f"Processing {len(assignment_candidates)} possible assignments...")
    
    # Assign customers greedily
    assigned_customers = set()
    
    for customer_i, customer_idx, branch_j, branch_au, distance in assignment_candidates:
        # Skip if customer already assigned
        if customer_idx in assigned_customers:
            continue
        
        # Skip if branch is at capacity
        if branch_capacity[branch_au] <= 0:
            continue
        
        # Assign customer to branch
        customer_assignments[branch_au].append({
            'customer_idx': customer_idx,
            'distance': distance
        })
        
        assigned_customers.add(customer_idx)
        branch_capacity[branch_au] -= 1
    
    # Find unassigned customers
    for customer_idx in customer_indices:
        if customer_idx not in assigned_customers:
            unassigned_customers.append(customer_idx)
    
    # Remove empty branches
    customer_assignments = {k: v for k, v in customer_assignments.items() if v}
    
    # Sort customers within each branch by distance
    for branch_au in customer_assignments:
        customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    print(f"Greedy assignment complete:")
    print(f"  - Unassigned customers: {len(unassigned_customers)}")
    
    # Print assignment summary
    for branch_au, customers in customer_assignments.items():
        if customers:
            distances = [c['distance'] for c in customers]
            print(f"  - Branch {branch_au}: {len(customers)} customers, avg distance: {np.mean(distances):.2f} miles")
    
    return customer_assignments, unassigned_customers

def assign_proximity_customers_to_existing_portfolios(unassigned_customers_df, customer_assignments, branch_df, proximity_threshold=20, max_portfolio_size=250):
    """
    Check if unassigned customers are within proximity of identified AUs
    Add them to existing portfolios up to max_portfolio_size
    """
    print(f"\nChecking proximity for {len(unassigned_customers_df)} unassigned customers...")
    print(f"Proximity threshold: {proximity_threshold} miles, Max portfolio size: {max_portfolio_size}")
    
    if len(unassigned_customers_df) == 0 or not customer_assignments:
        return [], list(unassigned_customers_df.index), customer_assignments
    
    # Get identified AUs and their coordinates
    identified_aus = list(customer_assignments.keys())
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_aus)].copy()
    
    print(f"Checking proximity to {len(identified_aus)} identified AUs: {identified_aus}")
    
    # Pre-compute current portfolio sizes
    current_portfolio_sizes = {}
    for branch_au, customers in customer_assignments.items():
        current_portfolio_sizes[branch_au] = len(customers)
    
    print("Current portfolio sizes:")
    for branch_au, size in current_portfolio_sizes.items():
        print(f"  - {branch_au}: {size} customers")
    
    # Calculate distances from unassigned customers to all identified AUs
    unassigned_coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    
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
    
    print(f"\nProximity assignment results:")
    print(f"  - Customers assigned via proximity: {len(proximity_results)}")
    print(f"  - Customers still unassigned: {len(remaining_unassigned)}")
    
    # Sort customers within each branch by distance
    for branch_au in updated_customer_assignments:
        updated_customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    # Print updated portfolio sizes
    print("\nUpdated portfolio sizes after proximity assignment:")
    for branch_au, customers in updated_customer_assignments.items():
        print(f"  - {branch_au}: {len(customers)} customers")
    
    return proximity_results, remaining_unassigned, updated_customer_assignments

def create_centralized_clusters_and_assign(unassigned_customers_df, branch_df, 
                                         min_size=200, max_size=240):
    """
    Create centralized clusters (no radius constraint) and assign to branches
    """
    print(f"\nCreating centralized clusters for {len(unassigned_customers_df)} customers...")
    print(f"Cluster constraints: min_size={min_size}, max_size={max_size}, no radius limit")
    
    if len(unassigned_customers_df) == 0:
        return [], []
    
    # Step 1: Create clusters with no radius constraint
    clustered_centralized, centralized_cluster_info = constrained_clustering_no_radius(
        unassigned_customers_df, min_size=min_size, max_size=max_size
    )
    
    centralized_results = []
    final_unassigned = []
    
    if len(centralized_cluster_info) > 0:
        print(f"Created {len(centralized_cluster_info)} centralized clusters")
        
        # Print cluster statistics
        print("\nCentralized Cluster Statistics:")
        for _, cluster in centralized_cluster_info.iterrows():
            print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} customers, "
                  f"radius: {cluster['radius']:.1f} miles")
        
        # Step 2: Assign clusters to branches
        cluster_assignments = assign_clusters_to_branches_vectorized(
            centralized_cluster_info, branch_df
        )
        
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
        
        # Find customers that couldn't be clustered (too few remaining)
        unassigned_centralized = clustered_centralized[
            clustered_centralized['cluster'] == -1
        ]
        final_unassigned = list(unassigned_centralized.index)
        
    else:
        print("No centralized clusters could be created")
        final_unassigned = list(unassigned_customers_df.index)
    
    print(f"\nCentralized assignment results:")
    print(f"  - Customers in centralized clusters: {len(centralized_results)}")
    print(f"  - Remaining unassigned: {len(final_unassigned)}")
    
    return centralized_results, final_unassigned

def create_customer_au_dataframe_with_proximity_and_centralized_clusters(customer_df, branch_df):
    """
    Modified main function that includes proximity check before centralized clustering
    """
    
    print(f"Starting with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Step 1: Create INMARKET clusters (with radius constraint, max_size=225)
    print("Step 1: Creating INMARKET clusters...")
    clustered_customers, cluster_info = constrained_clustering_optimized(customer_df)
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        print(f"Created {len(cluster_info)} INMARKET clusters")
        
        # Step 2: Assign clusters to branches
        print("Step 2: Assigning INMARKET clusters to branches...")
        cluster_assignments = assign_clusters_to_branches_vectorized(cluster_info, branch_df)
        
        # Step 3: Use greedy assignment for customer-AU assignment
        print("Step 3: Using greedy assignment for INMARKET customer-AU assignment...")
        customer_assignments, unassigned = greedy_assign_customers_to_branches(
            clustered_customers, cluster_assignments, branch_df
        )
        
        # Create INMARKET results
        for branch_au, customers in customer_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                
                inmarket_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': distance_value,
                    'TYPE': 'INMARKET'
                })
        
        unassigned_customer_indices.extend(unassigned)
    
    # Add customers that were never assigned to any cluster
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    print(f"Total unassigned customers after INMARKET: {len(unassigned_customer_indices)}")
    
    # Step 4: NEW - Check proximity of unassigned customers to identified AUs
    proximity_results = []
    final_unassigned_after_proximity = unassigned_customer_indices.copy()
    
    if unassigned_customer_indices and customer_assignments:
        unassigned_customers_df = customer_df.loc[unassigned_customer_indices]
        
        proximity_results, final_unassigned_after_proximity, updated_customer_assignments = assign_proximity_customers_to_existing_portfolios(
            unassigned_customers_df, customer_assignments, branch_df, 
            proximity_threshold=20, max_portfolio_size=250
        )
    
    # Step 5: Create CENTRALIZED clusters (no radius constraint) for remaining unassigned
    centralized_results = []
    final_unassigned = []
    
    if final_unassigned_after_proximity:
        remaining_unassigned_df = customer_df.loc[final_unassigned_after_proximity]
        
        centralized_results, final_unassigned = create_centralized_clusters_and_assign(
            remaining_unassigned_df, branch_df, min_size=200, max_size=240
        )
    
    # Combine results
    all_results = inmarket_results + proximity_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Print summary
    print(f"\n=== FINAL COMPREHENSIVE SUMMARY ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"INMARKET customers assigned: {len(inmarket_results)}")
def optimize_inmarket_portfolios_with_knn(result_df, branch_df, k_neighbors=3):
    """
    Optimize INMARKET portfolios by reassigning outlier customers using KNN
    """
    print(f"\nOptimizing INMARKET portfolios...")
    
    inmarket_customers = result_df[result_df['TYPE'] == 'INMARKET'].copy()
    if len(inmarket_customers) == 0:
        return result_df
    
    print(f"Total INMARKET customers to optimize: {len(inmarket_customers)}")
    
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
            print(f"AU {au}: Found {len(outliers)} outliers (threshold: {threshold:.2f} miles)")
            for idx, customer in outliers.iterrows():
                outlier_customers.append({
                    'customer_idx': idx,
                    'current_au': au,
                    'current_distance': customer['DISTANCE_TO_AU'],
                    'customer_data': customer
                })
    
    if not outlier_customers:
        print("No outlier customers found")
        return result_df
    
    print(f"Total outlier customers identified: {len(outlier_customers)}")
    
    # Get all INMARKET AUs and their coordinates
    inmarket_aus = inmarket_customers['ASSIGNED_AU'].unique()
    au_coords = []
    au_list = []
    
    for au in inmarket_aus:
        branch_coord = branch_df[branch_df['BRANCH_AU'] == au]
        if len(branch_coord) > 0:
            au_coords.append([branch_coord.iloc[0]['BRANCH_LAT_NUM'], branch_coord.iloc[0]['BRANCH_LON_NUM']])
            au_list.append(au)
    
    au_coords = np.array(au_coords)
    
    # Track current portfolio sizes
    portfolio_sizes = inmarket_customers['ASSIGNED_AU'].value_counts().to_dict()
    
    optimized_result = result_df.copy()
    total_improvements = 0
    
    # Step 2: Process each outlier customer
    for outlier in outlier_customers:
        customer_coord = np.array([[outlier['customer_data']['LAT_NUM'], outlier['customer_data']['LON_NUM']]])
        
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
                trade_customer_to_current = haversine_distance_vectorized(
                    target_farthest['LAT_NUM'], target_farthest['LON_NUM'],
                    outlier['customer_data']['LAT_NUM'], outlier['customer_data']['LON_NUM']
                )
                
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
                
                print(f"Direct reassignment: Customer from {outlier['current_au']} to {best_reassignment['target_au']} (improvement: {best_improvement:.2f} miles)")
                
            elif best_reassignment['type'] == 'trade':
                # Trade reassignment
                optimized_result.loc[outlier['customer_idx'], 'ASSIGNED_AU'] = best_reassignment['target_au']
                optimized_result.loc[outlier['customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['target_distance']
                
                optimized_result.loc[best_reassignment['trade_customer_idx'], 'ASSIGNED_AU'] = outlier['current_au']
                optimized_result.loc[best_reassignment['trade_customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['trade_distance']
                
                print(f"Trade: Customer from {outlier['current_au']} to {best_reassignment['target_au']} (improvement: {best_improvement:.2f} miles)")
            
            total_improvements += 1
            
            # Update inmarket_customers for subsequent iterations
            inmarket_customers = optimized_result[optimized_result['TYPE'] == 'INMARKET'].copy()
    
    print(f"Optimization complete: {total_improvements} customers reassigned")
    return optimized_result

def create_customer_au_dataframe_with_proximity_and_centralized_clusters(customer_df, branch_df):
    """
    Modified main function that includes proximity check before centralized clustering
    """
    
    print(f"Starting with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Step 1: Create INMARKET clusters (with radius constraint, max_size=225)
    print("Step 1: Creating INMARKET clusters...")
    clustered_customers, cluster_info = constrained_clustering_optimized(customer_df)
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        print(f"Created {len(cluster_info)} INMARKET clusters")
        
        # Step 2: Assign clusters to branches
        print("Step 2: Assigning INMARKET clusters to branches...")
        cluster_assignments = assign_clusters_to_branches_vectorized(cluster_info, branch_df)
        
        # Step 3: Use greedy assignment for customer-AU assignment
        print("Step 3: Using greedy assignment for INMARKET customer-AU assignment...")
        customer_assignments, unassigned = greedy_assign_customers_to_branches(
            clustered_customers, cluster_assignments, branch_df
        )
        
        # Create INMARKET results
        for branch_au, customers in customer_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                
                inmarket_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': distance_value,
                    'TYPE': 'INMARKET'
                })
        
        unassigned_customer_indices.extend(unassigned)
    
    # Add customers that were never assigned to any cluster
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    print(f"Total unassigned customers after INMARKET: {len(unassigned_customer_indices)}")
    
    # Step 4: Check proximity of unassigned customers to identified AUs
    proximity_results = []
    final_unassigned_after_proximity = unassigned_customer_indices.copy()
    
    if unassigned_customer_indices and customer_assignments:
        unassigned_customers_df = customer_df.loc[unassigned_customer_indices]
        
        proximity_results, final_unassigned_after_proximity, updated_customer_assignments = assign_proximity_customers_to_existing_portfolios(
            unassigned_customers_df, customer_assignments, branch_df, 
            proximity_threshold=20, max_portfolio_size=250
        )
    
    # Step 5: Create CENTRALIZED clusters for remaining unassigned
    centralized_results = []
    final_unassigned = []
    
    if final_unassigned_after_proximity:
        remaining_unassigned_df = customer_df.loc[final_unassigned_after_proximity]
        
        centralized_results, final_unassigned = create_centralized_clusters_and_assign(
            remaining_unassigned_df, branch_df, min_size=200, max_size=240
        )
    
    # Combine results
    all_results = inmarket_results + proximity_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Step 6: Optimize INMARKET portfolios until convergence
    if len(result_df) > 0:
        result_df = optimize_inmarket_portfolios_until_convergence(result_df, branch_df)
    
    # Print summary
    print(f"\n=== FINAL COMPREHENSIVE SUMMARY ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"INMARKET customers assigned: {len(result_df[result_df['TYPE'] == 'INMARKET'])}")
    print(f"CENTRALIZED customers assigned: {len(result_df[result_df['TYPE'] == 'CENTRALIZED'])}")
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
        
        print("\nNote: INMARKET includes both clustered customers and individual customers added via proximity to existing portfolios")
        
        au_type_summary = result_df.groupby(['ASSIGNED_AU', 'TYPE']).agg({
            'ECN': 'count',
            'DISTANCE_TO_AU': ['mean', 'max']
        }).round(2)
        au_type_summary.columns = ['Customer_Count', 'Avg_Distance', 'Max_Distance']
        print("\nAU Assignment Summary by Type:")
        print(au_type_summary)
        
        print(f"\nOverall Distance Statistics:")
        print(f"Average distance to assigned AU: {result_df['DISTANCE_TO_AU'].mean():.2f} miles")
        print(f"Maximum distance to assigned AU: {result_df['DISTANCE_TO_AU'].max():.2f} miles")
        print(f"Number of unique AUs used: {result_df['ASSIGNED_AU'].nunique()}")
        
        # Show centralized cluster sizes
        if 'CLUSTER_ID' in result_df.columns:
            centralized_clusters = result_df[result_df['TYPE'] == 'CENTRALIZED']
            if len(centralized_clusters) > 0:
                cluster_sizes = centralized_clusters.groupby('CLUSTER_ID').size()
                min_cluster_size = cluster_sizes.min()
                max_cluster_size = cluster_sizes.max()
                avg_cluster_size = cluster_sizes.mean()
                print(f"\nCentralized Cluster Sizes:")
                print(f"  Min: {min_cluster_size}")
                print(f"  Max: {max_cluster_size}")
                print(f"  Average: {avg_cluster_size:.1f}")
    
    return result_df

# Usage:
# customer_au_assignments = create_customer_au_dataframe_with_proximity_and_centralized_clusters(customer_df, branch_df)
# print(customer_au_assignments.head())
# customer_au_assignments.to_csv('customer_au_assignments_optimized.csv', index=False)
