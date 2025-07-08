import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    R = 3959  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def compute_distance_matrix(coords1, coords2):
    """Compute distance matrix between two sets of coordinates"""
    lat1 = coords1[:, 0][:, np.newaxis]
    lon1 = coords1[:, 1][:, np.newaxis]
    lat2 = coords2[:, 0][np.newaxis, :]
    lon2 = coords2[:, 1][np.newaxis, :]
    
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
        coords_rad = np.radians(customer_coords)
        ball_tree = BallTree(coords_rad, metric='haversine')
    
    seed_rad = np.radians(seed_coord).reshape(1, -1)
    radius_rad = max_radius / 3959
    
    indices = ball_tree.query_radius(seed_rad, r=radius_rad)[0]
    
    if len(indices) > 0:
        candidate_coords = customer_coords[indices]
        distances = haversine_distance_vectorized(
            candidate_coords[:, 0], candidate_coords[:, 1],
            seed_coord[0], seed_coord[1]
        )
        
        sorted_indices = np.argsort(distances)
        return indices[sorted_indices], distances[sorted_indices]
    
    return np.array([]), np.array([])

def constrained_clustering_optimized(customer_df, min_size=200, max_size=225, max_radius=20):
    """Optimized clustering with vectorized operations and spatial indexing"""
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values
    coords_rad = np.radians(coords)
    ball_tree = BallTree(coords_rad, metric='haversine')
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    unassigned_count = np.count_nonzero(unassigned_mask)
    while unassigned_count >= min_size:
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        candidate_indices, distances = find_candidates_spatial(
            coords[unassigned_mask], seed_coord, max_radius, None
        )
        
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[candidate_indices]
        
        if len(candidate_original_indices) == 0:
            unassigned_mask[seed_idx] = False
            unassigned_count = np.count_nonzero(unassigned_mask)
            continue
        
        current_cluster = [seed_idx]
        current_coords = [coords[seed_idx]]
        
        for i, (candidate_idx, distance) in enumerate(zip(candidate_original_indices, distances)):
            if candidate_idx == seed_idx:
                continue
                
            if len(current_cluster) >= max_size:
                break
            
            test_coords = np.array(current_coords + [coords[candidate_idx]])
            test_radius = calculate_cluster_radius_vectorized(test_coords)
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append(coords[candidate_idx])
        
        if len(current_cluster) < min_size:
            unassigned_mask[seed_idx] = False
        elif len(current_cluster) <= max_size:
            cluster_coords = np.array(current_coords)
            cluster_radius = calculate_cluster_radius_vectorized(cluster_coords)
            
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
            coords_array = np.array(current_coords)
            
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
            
            for idx in current_cluster:
                unassigned_mask[idx] = False
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    return customers_clean, pd.DataFrame(final_clusters)

def constrained_clustering_no_radius(customer_df, min_size=200, max_size=240, max_radius=100):
    """Clustering with size constraints and 100-mile radius constraint for centralized"""
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    coords = customers_clean[['LAT_NUM', 'LON_NUM']].values
    coords_rad = np.radians(coords)
    ball_tree = BallTree(coords_rad, metric='haversine')
    
    unassigned_mask = np.ones(len(customers_clean), dtype=bool)
    cluster_id = 0
    final_clusters = []
    
    unassigned_count = np.count_nonzero(unassigned_mask)
    while unassigned_count >= min_size:
        unassigned_indices = np.where(unassigned_mask)[0]
        seed_idx = unassigned_indices[0]
        seed_coord = coords[seed_idx]
        
        candidate_indices, distances = find_candidates_spatial(
            coords[unassigned_mask], seed_coord, max_radius, None
        )
        
        unassigned_positions = np.where(unassigned_mask)[0]
        candidate_original_indices = unassigned_positions[candidate_indices]
        
        if len(candidate_original_indices) == 0:
            unassigned_mask[seed_idx] = False
            unassigned_count = np.count_nonzero(unassigned_mask)
            continue
        
        if max_size < len(candidate_original_indices):
            cluster_size = max_size
        else:
            cluster_size = len(candidate_original_indices)
        
        current_cluster = list(candidate_original_indices[:cluster_size])
        current_coords = coords[current_cluster]
        
        if len(current_cluster) >= min_size:
            centroid = current_coords.mean(axis=0)
            cluster_radius = calculate_cluster_radius_vectorized(current_coords)
            
            for idx in current_cluster:
                customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] = cluster_id
                unassigned_mask[idx] = False
            
            final_clusters.append({
                'cluster_id': cluster_id,
                'size': len(current_cluster),
                'radius': cluster_radius,
                'centroid_lat': centroid[0],
                'centroid_lon': centroid[1]
            })
            
            cluster_id += 1
        else:
            break
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    return customers_clean, pd.DataFrame(final_clusters)

def assign_clusters_to_branches_vectorized(cluster_info, branch_df):
    """Vectorized cluster to branch assignment"""
    if len(cluster_info) == 0:
        return pd.DataFrame()
    
    cluster_coords = cluster_info[['centroid_lat', 'centroid_lon']].values
    branch_coords = branch_df[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    
    distance_matrix = compute_distance_matrix(cluster_coords, branch_coords)
    
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

def greedy_assign_customers_to_branches_with_managers(clustered_customers, cluster_assignments, branch_df, manager_df, max_distance=20, max_customers_per_manager=225):
    """Modified greedy assignment that assigns customers to managers"""
    
    identified_branches = cluster_assignments['assigned_branch'].unique()
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    customers_to_assign = clustered_customers[clustered_customers['cluster'] != -1].copy()
    
    if len(customers_to_assign) == 0 or len(identified_branch_coords) == 0:
        return {}, list(customers_to_assign.index), {}
    
    branch_managers = manager_df[manager_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    customer_coords = customers_to_assign[['LAT_NUM', 'LON_NUM']].values
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    
    distance_matrix = compute_distance_matrix(customer_coords, branch_coords)
    distance_matrix[distance_matrix > max_distance] = np.inf
    
    customer_indices = list(customers_to_assign.index)
    branch_aus = list(identified_branch_coords['BRANCH_AU'])
    
    manager_assignments = {}
    manager_capacities = {}
    branch_to_managers = {}
    
    for _, manager in branch_managers.iterrows():
        manager_id = manager['MANAGER_ID']
        branch_au = manager['BRANCH_AU']
        
        manager_assignments[manager_id] = []
        manager_capacities[manager_id] = max_customers_per_manager
        
        if branch_au not in branch_to_managers:
            branch_to_managers[branch_au] = []
        branch_to_managers[branch_au].append(manager_id)
    
    customer_assignments = {branch_au: [] for branch_au in branch_aus}
    unassigned_customers = []
    
    assignment_candidates = []
    for i, customer_idx in enumerate(customer_indices):
        for j, branch_au in enumerate(branch_aus):
            distance = distance_matrix[i, j]
            if distance < np.inf:
                assignment_candidates.append((i, customer_idx, j, branch_au, distance))
    
    assignment_candidates.sort(key=lambda x: x[4])
    
    assigned_customers = set()
    
    for customer_i, customer_idx, branch_j, branch_au, distance in assignment_candidates:
        if customer_idx in assigned_customers:
            continue
        
        available_managers = [mid for mid in branch_to_managers.get(branch_au, []) 
                             if manager_capacities[mid] > 0]
        
        if not available_managers:
            continue
        
        assigned_manager = available_managers[0]
        
        customer_assignments[branch_au].append({
            'customer_idx': customer_idx,
            'distance': distance,
            'assigned_manager': assigned_manager
        })
        
        manager_assignments[assigned_manager].append({
            'customer_idx': customer_idx,
            'distance': distance,
            'branch_au': branch_au
        })
        
        assigned_customers.add(customer_idx)
        manager_capacities[assigned_manager] -= 1
    
    for customer_idx in customer_indices:
        if customer_idx not in assigned_customers:
            unassigned_customers.append(customer_idx)
    
    customer_assignments = {k: v for k, v in customer_assignments.items() if v}
    
    for branch_au in customer_assignments:
        customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return customer_assignments, unassigned_customers, manager_assignments

def redistribute_small_portfolios(manager_assignments, customer_df, branch_df, manager_df, min_size=200):
    """Redistribute customers from portfolios smaller than 200 to nearby larger portfolios"""
    
    small_portfolios = []
    large_portfolios = []
    
    for manager_id, customers in manager_assignments.items():
        if len(customers) < min_size:
            small_portfolios.append(manager_id)
        else:
            large_portfolios.append(manager_id)
    
    if not small_portfolios or not large_portfolios:
        return manager_assignments
    
    # Calculate centroids for large portfolios
    large_portfolio_centroids = {}
    for manager_id in large_portfolios:
        customer_indices = [c['customer_idx'] for c in manager_assignments[manager_id]]
        customer_coords = customer_df.loc[customer_indices][['LAT_NUM', 'LON_NUM']].values
        
        if len(customer_coords) > 0:
            centroid_lat = np.mean(customer_coords[:, 0])
            centroid_lon = np.mean(customer_coords[:, 1])
            large_portfolio_centroids[manager_id] = (centroid_lat, centroid_lon)
    
    # Redistribute customers from small portfolios
    updated_manager_assignments = manager_assignments.copy()
    
    for small_manager_id in small_portfolios:
        customers_to_redistribute = updated_manager_assignments[small_manager_id].copy()
        
        for customer in customers_to_redistribute:
            customer_idx = customer['customer_idx']
            customer_data = customer_df.loc[customer_idx]
            customer_lat = customer_data['LAT_NUM']
            customer_lon = customer_data['LON_NUM']
            
            # Find nearest large portfolio
            min_distance = float('inf')
            nearest_manager = None
            
            for large_manager_id, (centroid_lat, centroid_lon) in large_portfolio_centroids.items():
                current_size = len(updated_manager_assignments[large_manager_id])
                
                distance = haversine_distance_vectorized(
                    customer_lat, customer_lon,
                    centroid_lat, centroid_lon
                )
                
                if distance < min_distance and current_size > min_size:
                    min_distance = distance
                    nearest_manager = large_manager_id
            
            # Transfer customer to nearest large portfolio
            if nearest_manager and len(updated_manager_assignments[nearest_manager]) > min_size:
                updated_manager_assignments[small_manager_id].remove(customer)
                
                updated_manager_assignments[nearest_manager].append({
                    'customer_idx': customer_idx,
                    'distance': min_distance,
                    'branch_au': customer['branch_au']
                })
                
                # Update centroid for future calculations
                customer_indices = [c['customer_idx'] for c in updated_manager_assignments[nearest_manager]]
                customer_coords = customer_df.loc[customer_indices][['LAT_NUM', 'LON_NUM']].values
                centroid_lat = np.mean(customer_coords[:, 0])
                centroid_lon = np.mean(customer_coords[:, 1])
                large_portfolio_centroids[nearest_manager] = (centroid_lat, centroid_lon)
    
    # Remove empty portfolios
    updated_manager_assignments = {k: v for k, v in updated_manager_assignments.items() if v}
    
    return updated_manager_assignments

def assign_proximity_customers_to_existing_portfolios(unassigned_customers_df, customer_assignments, branch_df, proximity_threshold=20, max_portfolio_size=250):
    """Check if unassigned customers are within proximity of identified AUs"""
    
    if len(unassigned_customers_df) == 0 or not customer_assignments:
        return [], list(unassigned_customers_df.index), customer_assignments
    
    identified_aus = list(customer_assignments.keys())
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_aus)].copy()
    
    current_portfolio_sizes = {}
    for branch_au, customers in customer_assignments.items():
        current_portfolio_sizes[branch_au] = len(customers)
    
    unassigned_coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values
    branch_coords = identified_branch_coords[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    
    distance_matrix = compute_distance_matrix(unassigned_coords, branch_coords)
    
    proximity_results = []
    remaining_unassigned = []
    updated_customer_assignments = customer_assignments.copy()
    
    for i, (customer_idx, customer_data) in enumerate(unassigned_customers_df.iterrows()):
        assigned = False
        
        customer_distances = distance_matrix[i, :]
        within_proximity = customer_distances <= proximity_threshold
        
        if np.any(within_proximity):
            proximity_aus = []
            for j, is_within in enumerate(within_proximity):
                if is_within:
                    branch_au = identified_branch_coords.iloc[j]['BRANCH_AU']
                    distance = customer_distances[j]
                    current_size = current_portfolio_sizes[branch_au]
                    
                    if current_size < max_portfolio_size:
                        proximity_aus.append((branch_au, distance, current_size))
            
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
    
    for branch_au in updated_customer_assignments:
        updated_customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return proximity_results, remaining_unassigned, updated_customer_assignments

def optimize_inmarket_portfolios_until_convergence(result_df, branch_df, k_neighbors=3):
    """Repeatedly optimize INMARKET portfolios until no more beneficial reassignments can be made"""
    
    optimized_result = result_df.copy()
    iteration = 0
    
    while True:
        iteration += 1
        
        inmarket_customers = optimized_result[optimized_result['TYPE'] == 'INMARKET'].copy()
        if len(inmarket_customers) == 0:
            break
        
        portfolio_groups = inmarket_customers.groupby('ASSIGNED_AU')
        outlier_customers = []
        
        # Identify outlier customers (3x median distance)
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
        
        au_coords = np.array(au_coords)
        
        portfolio_sizes = inmarket_customers['ASSIGNED_AU'].value_counts().to_dict()
        
        iteration_improvements = 0
        
        # Process each outlier customer
        for outlier in outlier_customers:
            customer_coord = np.array([[outlier['customer_data']['LAT_NUM'], outlier['customer_data']['LON_NUM']]])
            
            distances_to_aus = compute_distance_matrix(customer_coord, au_coords)[0]
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
                    optimized_result.loc[outlier['customer_idx'], 'ASSIGNED_AU'] = best_reassignment['target_au']
                    optimized_result.loc[outlier['customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['target_distance']
                    
                    portfolio_sizes[outlier['current_au']] -= 1
                    portfolio_sizes[best_reassignment['target_au']] = portfolio_sizes.get(best_reassignment['target_au'], 0) + 1
                    
                elif best_reassignment['type'] == 'trade':
                    optimized_result.loc[outlier['customer_idx'], 'ASSIGNED_AU'] = best_reassignment['target_au']
                    optimized_result.loc[outlier['customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['target_distance']
                    
                    optimized_result.loc[best_reassignment['trade_customer_idx'], 'ASSIGNED_AU'] = outlier['current_au']
                    optimized_result.loc[best_reassignment['trade_customer_idx'], 'DISTANCE_TO_AU'] = best_reassignment['trade_distance']
                
                iteration_improvements += 1
                
                inmarket_customers = optimized_result[optimized_result['TYPE'] == 'INMARKET'].copy()
        
        if iteration_improvements == 0:
            break
    
    return optimized_result

def create_customer_au_dataframe_with_proximity_and_centralized_clusters(customer_df, branch_df, manager_df=None):
    """Modified main function that maintains original output format while using manager logic internally"""
    
    # Use manager_df if provided, otherwise create dummy managers (one per branch)
    if manager_df is None:
        manager_df = pd.DataFrame({
            'MANAGER_ID': branch_df['BRANCH_AU'] + '_MGR',
            'BRANCH_AU': branch_df['BRANCH_AU']
        })
    
    # Step 1: Create INMARKET clusters
    clustered_customers, cluster_info = constrained_clustering_optimized(customer_df)
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        cluster_assignments = assign_clusters_to_branches_vectorized(cluster_info, branch_df)
        
        customer_assignments, unassigned, manager_assignments = greedy_assign_customers_to_branches_with_managers(
            clustered_customers, cluster_assignments, branch_df, manager_df
        )
        
        # Redistribute small portfolios
        manager_assignments = redistribute_small_portfolios(
            manager_assignments, customer_df, branch_df, manager_df, min_size=200
        )
        
        # Convert manager assignments back to branch assignments for output
        branch_customer_assignments = {}
        for manager_id, customers in manager_assignments.items():
            manager_branch = manager_df[manager_df['MANAGER_ID'] == manager_id]['BRANCH_AU'].iloc[0]
            
            if manager_branch not in branch_customer_assignments:
                branch_customer_assignments[manager_branch] = []
            
            for customer in customers:
                branch_customer_assignments[manager_branch].append({
                    'customer_idx': customer['customer_idx'],
                    'distance': customer.get('distance', 0)
                })
        
        # Create INMARKET results with original format
        for branch_au, customers in branch_customer_assignments.items():
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
        customer_assignments = branch_customer_assignments
    
    # Add customers that were never assigned to any cluster
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    # Step 2: Check proximity of unassigned customers to identified AUs
    proximity_results = []
    final_unassigned_after_proximity = unassigned_customer_indices.copy()
    
    if unassigned_customer_indices and customer_assignments:
        unassigned_customers_df = customer_df.loc[unassigned_customer_indices]
        
        proximity_results, final_unassigned_after_proximity, updated_customer_assignments = assign_proximity_customers_to_existing_portfolios(
            unassigned_customers_df, customer_assignments, branch_df, 
            proximity_threshold=20, max_portfolio_size=250
        )
    
    # Step 3: Create CENTRALIZED clusters with 100-mile radius
    centralized_results = []
    final_unassigned = []
    
    if final_unassigned_after_proximity:
        remaining_unassigned_df = customer_df.loc[final_unassigned_after_proximity]
        
        # Create clusters with 100-mile radius constraint
        clustered_centralized, centralized_cluster_info = constrained_clustering_no_radius(
            remaining_unassigned_df, min_size=200, max_size=240, max_radius=100
        )
        
        if len(centralized_cluster_info) > 0:
            cluster_assignments = assign_clusters_to_branches_vectorized(
                centralized_cluster_info, branch_df
            )
            
            for _, assignment in cluster_assignments.iterrows():
                cluster_id = assignment['cluster_id']
                assigned_branch = assignment['assigned_branch']
                
                cluster_customers = clustered_centralized[
                    clustered_centralized['cluster'] == cluster_id
                ]
                
                branch_coords = branch_df[
                    branch_df['BRANCH_AU'] == assigned_branch
                ][['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].iloc[0]
                
                for idx, customer in cluster_customers.iterrows():
                    distance = haversine_distance_vectorized(
                        customer['LAT_NUM'], customer['LON_NUM'],
                        branch_coords['BRANCH_LAT_NUM'], branch_coords['BRANCH_LON_NUM']
                    )
                    
                    original_customer = remaining_unassigned_df.loc[idx]
                    
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
            
            unassigned_centralized = clustered_centralized[
                clustered_centralized['cluster'] == -1
            ]
            final_unassigned = list(unassigned_centralized.index)
        else:
            final_unassigned = list(remaining_unassigned_df.index)
    
    # Combine results
    all_results = inmarket_results + proximity_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Step 4: Optimize INMARKET portfolios until convergence
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
                cluster_radii = []
                for cluster_id in cluster_sizes.index:
                    cluster_customers = centralized_clusters[centralized_clusters['CLUSTER_ID'] == cluster_id]
                    coords = cluster_customers[['LAT_NUM', 'LON_NUM']].values
                    radius = calculate_cluster_radius_vectorized(coords)
                    cluster_radii.append(radius)
                
                print(f"\nCentralized Cluster Statistics (100-mile radius constraint):")
                print(f"  Min cluster size: {cluster_sizes.min()}")
                print(f"  Max cluster size: {cluster_sizes.max()}")
                print(f"  Average cluster size: {cluster_sizes.mean():.1f}")
                if cluster_radii:
                    print(f"  Min cluster radius: {min(cluster_radii):.1f} miles")
                    print(f"  Max cluster radius: {max(cluster_radii):.1f} miles")
                    print(f"  Average cluster radius: {np.mean(cluster_radii):.1f} miles")
    
    return result_df

# Usage:
# customer_au_assignments = create_customer_au_dataframe_with_proximity_and_centralized_clusters(customer_df, branch_df)
# print(customer_au_assignments.head())
# customer_au_assignments.to_csv('customer_au_assignments_optimized.csv', index=False)
