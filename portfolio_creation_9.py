import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    R = 3959  # Earth's radius in miles
    
    # Convert to numpy arrays and ensure proper dtype
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
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    distance = R * c
    
    if np.isscalar(distance):
        return float(distance)
    return distance

def compute_distance_matrix(coords1, coords2):
    """Compute distance matrix between two sets of coordinates"""
    coords1 = np.asarray(coords1, dtype=np.float64)
    coords2 = np.asarray(coords2, dtype=np.float64)
    
    lat1 = coords1[:, 0][:, np.newaxis]
    lon1 = coords1[:, 1][:, np.newaxis]
    lat2 = coords2[:, 0][np.newaxis, :]
    lon2 = coords2[:, 1][np.newaxis, :]
    
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

def find_candidates_spatial(customer_coords, seed_coord, max_radius):
    """Use spatial indexing to find candidates within radius"""
    customer_coords = np.asarray(customer_coords, dtype=np.float64)
    seed_coord = np.asarray(seed_coord, dtype=np.float64)
    
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
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    while np.count_nonzero(unassigned_mask) >= min_size:
        # Find first unassigned customer as seed
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        # Find candidates using spatial indexing
        candidate_indices, distances = find_candidates_spatial(
            coords[unassigned_mask], seed_coord, max_radius
        )
        
        # Map back to original indices
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[candidate_indices]
        
        if len(candidate_original_indices) == 0:
            unassigned_mask[seed_idx] = False
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
            n_splits = np.maximum(1, len(current_cluster) // min_size)
            
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
            
            # Mark remaining as unassigned
            for idx in current_cluster:
                if customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] == -1:
                    unassigned_mask[idx] = False
    
    return customers_clean, pd.DataFrame(final_clusters)

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
    
    print(f"Rebalancing completed: {total_moves} customers moved")
    return balanced_result

def constrained_clustering_no_radius(customer_df, min_size=200, max_size=250):
    """Create clusters with size constraints but no radius limit for centralized portfolios"""
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    # Pre-compute coordinates array
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    while np.count_nonzero(unassigned_mask) >= min_size:
        # Find first unassigned customer as seed
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        # Get all unassigned customers and their distances to seed
        unassigned_coords = coords[unassigned_mask]
        distances = haversine_distance_vectorized(
            unassigned_coords[:, 0], unassigned_coords[:, 1],
            seed_coord[0], seed_coord[1]
        )
        
        # Sort by distance to seed
        sorted_indices = np.argsort(distances)
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[sorted_indices]
        
        # Build cluster by taking closest customers up to max_size
        current_cluster = []
        current_coords = []
        
        for candidate_idx in candidate_original_indices:
            if len(current_cluster) >= max_size:
                break
            
            current_cluster.append(candidate_idx)
            current_coords.append(coords[candidate_idx])
        
        # Only create cluster if it meets minimum size
        if len(current_cluster) >= min_size:
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
            # If we can't make a cluster of min_size, break
            break
    
    return customers_clean, pd.DataFrame(final_clusters)

def assign_clusters_to_branches_vectorized(cluster_info, branch_df):
    """Vectorized cluster to branch assignment"""
    if len(cluster_info) == 0:
        return pd.DataFrame()
    
    # Get cluster centroids and branch coordinates
    cluster_coords = cluster_info[['centroid_lat', 'centroid_lon']].values.astype(np.float64)
    branch_coords = branch_df[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values.astype(np.float64)
    
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
    """Greedy assignment of customers to branches with capacity constraints"""
    
    identified_branches = cluster_assignments['assigned_branch'].unique()
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    customers_to_assign = clustered_customers[clustered_customers['cluster'] != -1].copy()
    
    if len(customers_to_assign) == 0 or len(identified_branch_coords) == 0:
        return {}, list(customers_to_assign.index)
    
    # Pre-compute distance matrix
    customer_coords = customers_to_assign[['LAT_NUM', 'LON_NUM']].values.astype(np.float64)
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values.astype(np.float64)
    
    distance_matrix = compute_distance_matrix(customer_coords, branch_coords)
    distance_matrix[distance_matrix > max_distance] = np.inf
    
    customer_indices = list(customers_to_assign.index)
    branch_aus = list(identified_branch_coords['BRANCH_AU'])
    
    # Initialize branch capacities
    branch_capacity = {branch_au: max_customers_per_branch for branch_au in branch_aus}
    customer_assignments = {branch_au: [] for branch_au in branch_aus}
    unassigned_customers = []
    
    # Create list of (customer_idx, branch_idx, distance) sorted by distance
    assignment_candidates = []
    for i, customer_idx in enumerate(customer_indices):
        for j, branch_au in enumerate(branch_aus):
            distance = distance_matrix[i, j]
            if distance < np.inf:
                assignment_candidates.append((i, customer_idx, j, branch_au, distance))
    
    # Sort by distance (greedy: assign closest first)
    assignment_candidates.sort(key=lambda x: x[4])
    
    # Assign customers greedily
    assigned_customers = set()
    
    for customer_i, customer_idx, branch_j, branch_au, distance in assignment_candidates:
        if customer_idx in assigned_customers or branch_capacity[branch_au] <= 0:
            continue
        
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
    
    # Remove empty branches and sort customers by distance
    customer_assignments = {k: v for k, v in customer_assignments.items() if v}
    for branch_au in customer_assignments:
        customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return customer_assignments, unassigned_customers

def assign_proximity_customers_to_existing_portfolios(unassigned_customers_df, customer_assignments, branch_df, proximity_threshold=20, max_portfolio_size=250):
    """Assign unassigned customers to existing portfolios if within proximity"""
    
    if len(unassigned_customers_df) == 0 or not customer_assignments:
        return [], list(unassigned_customers_df.index), customer_assignments
    
    # Get identified AUs and their coordinates
    identified_aus = list(customer_assignments.keys())
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_aus)].copy()
    
    # Pre-compute current portfolio sizes
    current_portfolio_sizes = {branch_au: len(customers) for branch_au, customers in customer_assignments.items()}
    
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
                proximity_aus.sort(key=lambda x: x[1])
                
                for branch_au, distance, current_size in proximity_aus:
                    if current_portfolio_sizes[branch_au] < max_portfolio_size:
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

def customer_au_assignment(customer_df, branch_df):
    """Main function for customer-AU assignment"""
    
    print(f"Starting assignment with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Step 1: Create INMARKET clusters (20-mile radius)
    print("Step 1: Creating INMARKET clusters (20-mile radius)...")
    clustered_customers, cluster_info = constrained_clustering_optimized(customer_df)
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        print(f"Created {len(cluster_info)} INMARKET clusters")
        
        # Step 2: Assign clusters to branches
        print("Step 2: Assigning clusters to branches...")
        cluster_assignments = assign_clusters_to_branches_vectorized(cluster_info, branch_df)
        
        # Step 3: Assign customers to branches
        print("Step 3: Assigning customers to branches...")
        customer_assignments, unassigned = greedy_assign_customers_to_branches(
            clustered_customers, cluster_assignments, branch_df
        )
        
        # Create INMARKET results
        for branch_au, customers in customer_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                inmarket_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': customer['distance'],
                    'TYPE': 'INMARKET'
                })
        
        unassigned_customer_indices.extend(unassigned)
    
    # Add customers that were never assigned to any cluster
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    print(f"Unassigned customers after first INMARKET pass: {len(unassigned_customer_indices)}")
    
    # Step 4: Check proximity of unassigned customers to identified AUs
    proximity_results = []
    if unassigned_customer_indices and inmarket_results:
        # Create customer_assignments from inmarket_results
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
        
        proximity_results, remaining_after_proximity, _ = assign_proximity_customers_to_existing_portfolios(
            unassigned_customers_df, customer_assignments, branch_df, 
            proximity_threshold=20, max_portfolio_size=250
        )
    else:
        remaining_after_proximity = unassigned_customer_indices
    
    print(f"Remaining customers after proximity assignment: {len(remaining_after_proximity)}")
    
    # Step 5: Create second iteration of INMARKET clusters with 40-mile radius
    extended_inmarket_results = []
    if remaining_after_proximity:
        print("Step 5: Creating extended INMARKET clusters (40-mile radius)...")
        
        remaining_customers_df = customer_df.loc[remaining_after_proximity]
        
        # Create clusters with 40-mile radius
        clustered_extended, cluster_info_extended = constrained_clustering_optimized(
            remaining_customers_df, min_size=200, max_size=225, max_radius=40
        )
        
        if len(cluster_info_extended) > 0:
            print(f"Created {len(cluster_info_extended)} extended INMARKET clusters")
            
            # Assign extended clusters to branches
            cluster_assignments_extended = assign_clusters_to_branches_vectorized(
                cluster_info_extended, branch_df
            )
            
            # Assign customers to branches with extended distance allowance
            customer_assignments_extended, unassigned_extended = greedy_assign_customers_to_branches(
                clustered_extended, cluster_assignments_extended, branch_df, 
                max_distance=40, max_customers_per_branch=225
            )
            
            # Create extended INMARKET results
            for branch_au, customers in customer_assignments_extended.items():
                for customer in customers:
                    customer_idx = customer['customer_idx']
                    customer_data = customer_df.loc[customer_idx]
                    
                    extended_inmarket_results.append({
                        'ECN': customer_data['ECN'],
                        'BILLINGCITY': customer_data['BILLINGCITY'],
                        'BILLINGSTATE': customer_data['BILLINGSTATE'],
                        'LAT_NUM': customer_data['LAT_NUM'],
                        'LON_NUM': customer_data['LON_NUM'],
                        'ASSIGNED_AU': branch_au,
                        'DISTANCE_TO_AU': customer['distance'],
                        'TYPE': 'INMARKET'
                    })
            
            # Update final unassigned list
            never_assigned_extended = clustered_extended[clustered_extended['cluster'] == -1].index.tolist()
            final_unassigned = list(set(unassigned_extended + never_assigned_extended))
        else:
            final_unassigned = remaining_after_proximity
    else:
        final_unassigned = []
    
    # Combine INMARKET and proximity results for rebalancing
    inmarket_and_proximity_results = inmarket_results + proximity_results + extended_inmarket_results
    
    # Step 6: Rebalance INMARKET portfolios to meet minimum size requirements
    if inmarket_and_proximity_results:
        print("Step 6: Rebalancing INMARKET portfolios to minimum size...")
        
        # Create DataFrame with proper indexing
        inmarket_df = pd.DataFrame(inmarket_and_proximity_results)
        
        # Add customer indices to the DataFrame for proper tracking
        customer_indices = []
        for result in inmarket_and_proximity_results:
            # Find customer index for this result
            customer_idx = customer_df[
                (customer_df['ECN'] == result['ECN']) &
                (customer_df['LAT_NUM'] == result['LAT_NUM']) &
                (customer_df['LON_NUM'] == result['LON_NUM'])
            ].index[0]
            customer_indices.append(customer_idx)
        
        # Set proper index on the DataFrame
        inmarket_df.index = customer_indices
        
        # Perform rebalancing
        inmarket_df = rebalance_portfolio_sizes(inmarket_df, branch_df, min_size=200)
        
        # Update the results with rebalanced assignments
        inmarket_and_proximity_results = inmarket_df.to_dict('records')
        
        # Update final_unassigned - remove customers that are now assigned through rebalancing
        assigned_indices = set(inmarket_df.index)
        final_unassigned = [idx for idx in final_unassigned if idx not in assigned_indices]
    else:
        inmarket_and_proximity_results = []
    
    # Step 7: Create centralized portfolios from remaining customers
    centralized_results = []
    if final_unassigned:
        print(f"Step 6: Creating centralized portfolios from {len(final_unassigned)} remaining customers...")
        
        remaining_customers_df = customer_df.loc[final_unassigned]
        
        # Create centralized clusters with no radius constraint
        centralized_clusters, centralized_cluster_info = constrained_clustering_no_radius(
            remaining_customers_df, min_size=200, max_size=250
        )
        
        if len(centralized_cluster_info) > 0:
            print(f"Created {len(centralized_cluster_info)} centralized clusters")
            
            # Assign centralized clusters to branches
            centralized_assignments = assign_clusters_to_branches_vectorized(
                centralized_cluster_info, branch_df
            )
            
            # Create centralized results
            for _, assignment in centralized_assignments.iterrows():
                cluster_id = assignment['cluster_id']
                assigned_branch = assignment['assigned_branch']
                
                # Get customers in this cluster
                cluster_customers = centralized_clusters[
                    centralized_clusters['cluster'] == cluster_id
                ]
                
                # Get branch coordinates for distance calculation
                branch_coords = branch_df[
                    branch_df['BRANCH_AU'] == assigned_branch
                ][['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].iloc[0]
                
                # Calculate distances and create results
                for idx, customer in cluster_customers.iterrows():
                    distance = haversine_distance_vectorized(
                        customer['LAT_NUM'], customer['LON_NUM'],
                        branch_coords['BRANCH_LAT_NUM'], branch_coords['BRANCH_LON_NUM']
                    )
                    
                    # Get original customer data
                    original_customer = customer_df.loc[idx]
                    
                    centralized_results.append({
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
            
            # Update final unassigned list
            final_unassigned = centralized_clusters[
                centralized_clusters['cluster'] == -1
            ].index.tolist()
        
        print(f"Centralized portfolios created: {len(centralized_cluster_info)}")
        print(f"Customers assigned to centralized: {len(centralized_results)}")
        print(f"Final unassigned customers: {len(final_unassigned)}")
    
    # Combine all results
    all_results = inmarket_results + proximity_results + extended_inmarket_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Step 7: Optimize portfolios (only INMARKET types)
    if len(result_df) > 0:
        print("Step 7: Optimizing INMARKET portfolios...")
        result_df = optimize_inmarket_portfolios(result_df, branch_df)
    
    # Print summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"Customers assigned: {len(result_df)}")
    print(f"Unassigned customers: {len(final_unassigned)}")
    
    if len(result_df) > 0:
        type_summary = result_df.groupby('TYPE').agg({
            'ECN': 'count',
            'DISTANCE_TO_AU': ['mean', 'max']
        }).round(2)
        type_summary.columns = ['Customer_Count', 'Avg_Distance', 'Max_Distance']
        print("\nPortfolio Type Summary:")
        print(type_summary)
        
        print(f"\nOverall Statistics:")
        print(f"Average distance to assigned AU: {result_df['DISTANCE_TO_AU'].mean():.2f} miles")
        print(f"Maximum distance to assigned AU: {result_df['DISTANCE_TO_AU'].max():.2f} miles")
        print(f"Number of unique AUs used: {result_df['ASSIGNED_AU'].nunique()}")
    
    return result_df

# Usage:
# assignments = customer_au_assignment(customer_df, branch_df)
# assignments.to_csv('customer_au_assignments.csv', index=False)
