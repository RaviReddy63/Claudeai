import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
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

def constrained_clustering_optimized(customer_df, min_size=200, max_size=280, max_radius=20):
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
    
    while np.sum(unassigned_mask) >= min_size:
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
                n_splits = min(3, len(current_cluster) // min_size)
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

def hungarian_assign_customers_to_branches_optimized(clustered_customers, cluster_assignments, branch_df, max_distance=20, max_customers_per_branch=280):
    """Optimized Hungarian algorithm with pre-computed distance matrices"""
    
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
    
    # Create expanded matrix for capacity constraints
    total_slots = len(branch_aus) * max_customers_per_branch
    expanded_distance_matrix = np.full((len(customer_indices), total_slots), np.inf)
    
    slot_to_branch = {}
    slot_idx = 0
    for j, branch_au in enumerate(branch_aus):
        for slot in range(max_customers_per_branch):
            expanded_distance_matrix[:, slot_idx] = distance_matrix[:, j]
            slot_to_branch[slot_idx] = branch_au
            slot_idx += 1
    
    # Make matrix square and apply Hungarian algorithm
    max_dim = max(len(customer_indices), total_slots)
    square_matrix = np.full((max_dim, max_dim), 1000000.0)  # High cost for dummy assignments
    square_matrix[:len(customer_indices), :total_slots] = expanded_distance_matrix
    
    print("Applying Hungarian algorithm...")
    customer_assignments_idx, slot_assignments_idx = linear_sum_assignment(square_matrix)
    
    # Process results
    customer_assignments = {}
    unassigned_customers = []
    
    for i, slot_idx in enumerate(slot_assignments_idx):
        if i < len(customer_indices):
            customer_idx = customer_indices[i]
            
            if slot_idx < total_slots:
                assignment_cost = square_matrix[i, slot_idx]
                
                if assignment_cost < np.inf and assignment_cost < 1000000:
                    branch_au = slot_to_branch[slot_idx]
                    distance = assignment_cost
                    
                    if branch_au not in customer_assignments:
                        customer_assignments[branch_au] = []
                    
                    customer_assignments[branch_au].append({
                        'customer_idx': customer_idx,
                        'distance': distance
                    })
                else:
                    unassigned_customers.append(customer_idx)
            else:
                unassigned_customers.append(customer_idx)
    
    # Sort customers within each branch by distance
    for branch_au in customer_assignments:
        customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    print(f"Hungarian assignment complete:")
    print(f"  - Unassigned customers: {len(unassigned_customers)}")
    
    # Print assignment summary
    for branch_au, customers in customer_assignments.items():
        if customers:
            distances = [c['distance'] for c in customers]
            print(f"  - Branch {branch_au}: {len(customers)} customers, avg distance: {np.mean(distances):.2f} miles")
    
    return customer_assignments, unassigned_customers

def create_centralized_portfolios_optimized(unassigned_customers_df, branch_df, max_size=280):
    """Optimized centralized portfolio creation"""
    print(f"\nCreating centralized portfolios from {len(unassigned_customers_df)} unassigned customers...")
    
    if len(unassigned_customers_df) == 0:
        return {}, []

    coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values
    customer_indices = unassigned_customers_df.index.tolist()
    
    n_customers = len(unassigned_customers_df)
    
    # Modified condition with safety check
    if max_size > 0:
        n_clusters = max(1, (n_customers + max_size - 1) // max_size)
    else:
        n_clusters = 1
    
    print(f"Creating {n_clusters} centralized clusters from {n_customers} customers")
    
    # Pre-compute customer-branch distance matrix
    branch_coords = branch_df[['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']].values
    distance_matrix = compute_distance_matrix(coords, branch_coords)
    
    centralized_assignments = {}
    remaining_customers = []
    
    if n_clusters == 1:
        # Single cluster
        total_distances = np.sum(distance_matrix, axis=0)
        best_branch_idx = np.argmin(total_distances)
        assigned_branch = branch_df.iloc[best_branch_idx]['BRANCH_AU']
        
        print(f"Assigned all customers to branch {assigned_branch}")
        
        centralized_assignments[assigned_branch] = []
        
        for i, customer_idx in enumerate(customer_indices):
            if i < max_size:
                distance = distance_matrix[i, best_branch_idx]
                centralized_assignments[assigned_branch].append({
                    'customer_idx': customer_idx,
                    'distance': distance
                })
            else:
                remaining_customers.append(customer_idx)
    else:
        # Multiple clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = [customer_indices[i] for i in range(len(customer_indices)) if cluster_mask[i]]
            
            if len(cluster_indices) == 0:
                continue
            
            # Take only max_size customers
            if len(cluster_indices) > max_size:
                remaining_customers.extend(cluster_indices[max_size:])
                cluster_indices = cluster_indices[:max_size]
            
            if len(cluster_indices) == 0:
                continue
            
            # Find best branch for this cluster
            cluster_distances = distance_matrix[cluster_mask][:len(cluster_indices)]
            total_distances = np.sum(cluster_distances, axis=0)
            best_branch_idx = np.argmin(total_distances)
            assigned_branch = branch_df.iloc[best_branch_idx]['BRANCH_AU']
            
            if assigned_branch not in centralized_assignments:
                centralized_assignments[assigned_branch] = []
            
            for i, customer_idx in enumerate(cluster_indices):
                distance = cluster_distances[i, best_branch_idx]
                centralized_assignments[assigned_branch].append({
                    'customer_idx': customer_idx,
                    'distance': distance
                })
    
    print(f"Created {len(centralized_assignments)} centralized portfolios")
    print(f"Remaining unassigned customers: {len(remaining_customers)}")
    
    return centralized_assignments, remaining_customers

def create_customer_au_dataframe_optimized(customer_df, branch_df):
    """Optimized main function with vectorized operations"""
    
    print(f"Starting with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Step 1: Create INMARKET clusters
    print("Step 1: Creating INMARKET clusters...")
    clustered_customers, cluster_info = constrained_clustering_optimized(customer_df)
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        print(f"Created {len(cluster_info)} INMARKET clusters")
        
        # Step 2: Assign clusters to branches
        print("Step 2: Assigning INMARKET clusters to branches...")
        cluster_assignments = assign_clusters_to_branches_vectorized(cluster_info, branch_df)
        
        # Step 3: Use Hungarian algorithm for optimal customer-AU assignment
        print("Step 3: Using Hungarian algorithm for optimal INMARKET customer-AU assignment...")
        customer_assignments, unassigned = hungarian_assign_customers_to_branches_optimized(
            clustered_customers, cluster_assignments, branch_df
        )
        
        # Create INMARKET results
        for branch_au, customers in customer_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                
                # Safe distance conversion with error handling
                try:
                    distance_rounded = round(float(distance_value), 2)
                except (ValueError, TypeError):
                    distance_rounded = 0
                
                inmarket_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': distance_rounded,
                    'TYPE': 'INMARKET'
                })
        
        unassigned_customer_indices.extend(unassigned)
    
    # Add customers that were never assigned to any cluster
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    print(f"Total unassigned customers for centralized processing: {len(unassigned_customer_indices)}")
    
    # Step 4: Create CENTRALIZED portfolios
    centralized_results = []
    final_unassigned = []
    
    if unassigned_customer_indices:
        unassigned_customers_df = customer_df.loc[unassigned_customer_indices]
        
        centralized_assignments, remaining_customers = create_centralized_portfolios_optimized(
            unassigned_customers_df, branch_df
        )
        
        for branch_au, customers in centralized_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                
                # Safe distance conversion with error handling
                try:
                    distance_rounded = round(float(distance_value), 2)
                except (ValueError, TypeError):
                    distance_rounded = 0
                
                centralized_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': distance_rounded,
                    'TYPE': 'CENTRALIZED'
                })
        
        final_unassigned = remaining_customers
    
    # Combine results
    all_results = inmarket_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Print summary
    print(f"\n=== FINAL COMPREHENSIVE SUMMARY ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"INMARKET customers assigned: {len(inmarket_results)}")
    print(f"CENTRALIZED customers assigned: {len(centralized_results)}")
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
    
    return result_df

# Usage:
# customer_au_assignments = create_customer_au_dataframe_optimized(customer_df, branch_df)
# print(customer_au_assignments.head())
# customer_au_assignments.to_csv('customer_au_assignments_optimized.csv', index=False)
