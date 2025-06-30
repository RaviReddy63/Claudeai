import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

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
        if distance > max_distance:
            max_distance = distance
    
    return max_distance

def constrained_clustering(customer_df, min_size=200, max_size=280, max_radius=20):
    """Create clusters with size and radius constraints"""
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1
    
    unassigned_customers = customers_clean.copy()
    cluster_id = 0
    final_clusters = []
    
    while len(unassigned_customers) >= min_size:
        seed_idx = unassigned_customers.index[0]
        seed_customer = unassigned_customers.loc[seed_idx]
        
        current_cluster = [seed_idx]
        current_coords = [(seed_customer['LAT_NUM'], seed_customer['LON_NUM'])]
        
        # Find candidates within radius
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
        
        candidates.sort(key=lambda x: x[2])
        
        # Add customers while maintaining constraints
        for candidate_idx, candidate_customer, _ in candidates:
            if len(current_cluster) >= max_size:
                break
            
            test_coords = current_coords + [(candidate_customer['LAT_NUM'], candidate_customer['LON_NUM'])]
            test_radius = calculate_cluster_radius(np.array(test_coords))
            
            if test_radius <= max_radius:
                current_cluster.append(candidate_idx)
                current_coords.append((candidate_customer['LAT_NUM'], candidate_customer['LON_NUM']))
        
        # Handle cluster based on size
        if len(current_cluster) < min_size:
            unassigned_customers = unassigned_customers.drop([seed_idx])
        elif len(current_cluster) <= max_size:
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
        else:
            # Split large cluster using K-means
            coords_array = np.array(current_coords)
            n_splits = min(3, len(current_cluster) // min_size)
            
            if n_splits > 1:
                kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                subcluster_labels = kmeans.fit_predict(coords_array)
                
                for sub_id in range(n_splits):
                    sub_mask = subcluster_labels == sub_id
                    sub_indices = [current_cluster[i] for i in range(len(current_cluster)) if sub_mask[i]]
                    sub_coords = [current_coords[i] for i in range(len(current_coords)) if sub_mask[i]]
                    
                    if len(sub_indices) >= min_size and len(sub_indices) <= max_size:
                        sub_radius = calculate_cluster_radius(np.array(sub_coords))
                        if sub_radius <= max_radius:
                            for idx in sub_indices:
                                customers_clean.loc[idx, 'cluster'] = cluster_id
                            
                            final_clusters.append({
                                'cluster_id': cluster_id,
                                'size': len(sub_indices),
                                'radius': sub_radius,
                                'centroid_lat': np.mean([coord[0] for coord in sub_coords]),
                                'centroid_lon': np.mean([coord[1] for coord in sub_coords])
                            })
                            cluster_id += 1
            
            unassigned_customers = unassigned_customers.drop(current_cluster)
    
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
            'cluster_to_branch_distance': min_distance
        })
    
    return pd.DataFrame(cluster_assignments)

def hungarian_assign_customers_to_branches(clustered_customers, cluster_assignments, branch_df, max_distance=20, max_customers_per_branch=280):
    """
    Use Hungarian algorithm for optimal customer-AU assignment with capacity constraints
    """
    # Get identified branches from cluster assignments
    identified_branches = cluster_assignments['assigned_branch'].unique()
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    print(f"Identified branches: {list(identified_branches)}")
    print(f"Available branch coordinates: {len(identified_branch_coords)}")
    
    customers_to_assign = clustered_customers[clustered_customers['cluster'] != -1].copy()
    print(f"Customers to assign: {len(customers_to_assign)}")
    
    if len(customers_to_assign) == 0 or len(identified_branch_coords) == 0:
        return {}, list(customers_to_assign.index)
    
    # Create distance matrix: customers x branches
    customer_indices = list(customers_to_assign.index)
    branch_aus = list(identified_branch_coords['BRANCH_AU'])
    
    print(f"Creating distance matrix: {len(customer_indices)} customers x {len(branch_aus)} branches")
    
    # Calculate all distances
    distance_matrix = np.full((len(customer_indices), len(branch_aus)), np.inf)
    
    for i, customer_idx in enumerate(customer_indices):
        customer = customers_to_assign.loc[customer_idx]
        for j, branch_au in enumerate(branch_aus):
            branch = identified_branch_coords[identified_branch_coords['BRANCH_AU'] == branch_au].iloc[0]
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            
            # Only allow assignment if within max_distance
            if distance <= max_distance:
                distance_matrix[i, j] = distance
    
    # Handle capacity constraints by creating virtual branches
    # Each branch gets max_customers_per_branch "slots"
    print(f"Creating virtual branches for capacity constraints...")
    
    # Create expanded matrix with virtual branches
    total_slots = len(branch_aus) * max_customers_per_branch
    expanded_distance_matrix = np.full((len(customer_indices), total_slots), np.inf)
    
    slot_to_branch = {}  # Maps slot index to branch AU
    
    slot_idx = 0
    for j, branch_au in enumerate(branch_aus):
        for slot in range(max_customers_per_branch):
            expanded_distance_matrix[:, slot_idx] = distance_matrix[:, j]
            slot_to_branch[slot_idx] = branch_au
            slot_idx += 1
    
    print(f"Expanded matrix size: {len(customer_indices)} x {total_slots}")
    
    # Apply Hungarian algorithm
    print("Applying Hungarian algorithm...")
    
    # Handle case where we have more customers than slots
    if len(customer_indices) > total_slots:
        print(f"Warning: More customers ({len(customer_indices)}) than available slots ({total_slots})")
        # We'll process in batches or handle unassigned customers separately
        
    # Make matrix square by padding if necessary
    if len(customer_indices) != total_slots:
        max_dim = max(len(customer_indices), total_slots)
        square_matrix = np.full((max_dim, max_dim), np.inf)
        square_matrix[:len(customer_indices), :total_slots] = expanded_distance_matrix
        
        # Add high cost for dummy assignments
        if len(customer_indices) < max_dim:
            square_matrix[len(customer_indices):, :] = 1000000  # High cost for dummy customers
        if total_slots < max_dim:
            square_matrix[:, total_slots:] = 1000000  # High cost for dummy slots
    else:
        square_matrix = expanded_distance_matrix
    
    # Apply Hungarian algorithm
    customer_assignments_idx, slot_assignments_idx = linear_sum_assignment(square_matrix)
    
    print("Hungarian algorithm completed. Processing results...")
    
    # Process results
    customer_assignments = {}
    unassigned_customers = []
    
    for i, slot_idx in enumerate(slot_assignments_idx):
        if i < len(customer_indices):  # Real customer (not dummy)
            customer_idx = customer_indices[i]
            
            if slot_idx < total_slots:  # Real slot (not dummy)
                assignment_cost = square_matrix[i, slot_idx]
                
                if assignment_cost < np.inf and assignment_cost < 1000000:  # Valid assignment
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
    print(f"  - Assigned customers: {sum(len(customers) for customers in customer_assignments.values())}")
    print(f"  - Unassigned customers: {len(unassigned_customers)}")
    
    # Print assignment summary
    for branch_au, customers in customer_assignments.items():
        if customers:
            distances = [c['distance'] for c in customers]
            print(f"  - Branch {branch_au}: {len(customers)} customers, avg distance: {np.mean(distances):.2f} miles")
    
    return customer_assignments, unassigned_customers

def create_centralized_portfolios(unassigned_customers_df, branch_df, max_size=280):
    """
    Create centralized portfolios from unassigned customers using Hungarian algorithm
    """
    print(f"\nCreating centralized portfolios from {len(unassigned_customers_df)} unassigned customers...")
    
    if len(unassigned_customers_df) == 0:
        return {}, []

    # Get coordinates for clustering
    coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values
    customer_indices = unassigned_customers_df.index.tolist()
    
    # Determine number of clusters needed
    n_customers = len(unassigned_customers_df)
    n_clusters = max(1, (n_customers + max_size - 1) // max_size)  # Ceiling division
    
    print(f"Creating {n_clusters} centralized clusters from {n_customers} customers")
    
    if n_clusters == 1:
        # Single cluster - assign all to nearest branch using Hungarian algorithm
        print("Single cluster - using Hungarian algorithm for branch assignment")
        
        # Create distance matrix: customers x branches
        distance_matrix = np.full((len(customer_indices), len(branch_df)), np.inf)
        
        for i, customer_idx in enumerate(customer_indices):
            customer = unassigned_customers_df.loc[customer_idx]
            for j, (_, branch) in enumerate(branch_df.iterrows()):
                distance = haversine_distance(
                    customer['LAT_NUM'], customer['LON_NUM'],
                    branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                )
                distance_matrix[i, j] = distance
        
        # Find the branch with minimum total distance
        total_distances = np.sum(distance_matrix, axis=0)
        best_branch_idx = np.argmin(total_distances)
        assigned_branch = branch_df.iloc[best_branch_idx]['BRANCH_AU']
        
        print(f"Assigned all customers to branch {assigned_branch}")
        
        centralized_assignments = {assigned_branch: []}
        remaining_customers = []
        
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
        # Multiple clusters needed
        print(f"Multiple clusters needed - using K-means + Hungarian algorithm")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        centralized_assignments = {}
        remaining_customers = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = [customer_indices[i] for i in range(len(customer_indices)) if cluster_mask[i]]
            
            if len(cluster_indices) == 0:
                continue
                
            print(f"Processing cluster {cluster_id} with {len(cluster_indices)} customers")
            
            # Take only max_size customers from this cluster
            if len(cluster_indices) > max_size:
                remaining_customers.extend(cluster_indices[max_size:])
                cluster_indices = cluster_indices[:max_size]
            
            if len(cluster_indices) == 0:
                continue
            
            # Create distance matrix for this cluster: customers x branches
            cluster_distance_matrix = np.full((len(cluster_indices), len(branch_df)), np.inf)
            
            for i, customer_idx in enumerate(cluster_indices):
                customer = unassigned_customers_df.loc[customer_idx]
                for j, (_, branch) in enumerate(branch_df.iterrows()):
                    distance = haversine_distance(
                        customer['LAT_NUM'], customer['LON_NUM'],
                        branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                    )
                    cluster_distance_matrix[i, j] = distance
            
            # Find best branch for this cluster (minimum total distance)
            total_distances = np.sum(cluster_distance_matrix, axis=0)
            best_branch_idx = np.argmin(total_distances)
            assigned_branch = branch_df.iloc[best_branch_idx]['BRANCH_AU']
            
            print(f"Cluster {cluster_id} assigned to branch {assigned_branch}")
            
            # Assign customers to this branch
            if assigned_branch not in centralized_assignments:
                centralized_assignments[assigned_branch] = []
            
            for i, customer_idx in enumerate(cluster_indices):
                distance = cluster_distance_matrix[i, best_branch_idx]
                centralized_assignments[assigned_branch].append({
                    'customer_idx': customer_idx,
                    'distance': distance
                })
    
    print(f"Created {len(centralized_assignments)} centralized portfolios")
    print(f"Remaining unassigned customers: {len(remaining_customers)}")
    
    return centralized_assignments, remaining_customers

def create_customer_au_dataframe(customer_df, branch_df):
    """
    Main function to create customer portfolio with AU assignments using Hungarian algorithm
    Returns: DataFrame with customer details, assigned AU, and portfolio type
    """
    
    print(f"Starting with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Step 1: Create INMARKET clusters
    print("Step 1: Creating INMARKET clusters...")
    clustered_customers, cluster_info = constrained_clustering(customer_df)
    
    inmarket_results = []
    unassigned_customer_indices = []
    
    if len(cluster_info) > 0:
        print(f"Created {len(cluster_info)} INMARKET clusters")
        
        # Step 2: Assign clusters to branches
        print("Step 2: Assigning INMARKET clusters to branches...")
        cluster_assignments = assign_clusters_to_branches(clustered_customers, cluster_info, branch_df)
        
        # Step 3: Use Hungarian algorithm for optimal customer-AU assignment
        print("Step 3: Using Hungarian algorithm for optimal INMARKET customer-AU assignment...")
        customer_assignments, unassigned = hungarian_assign_customers_to_branches(
            clustered_customers, cluster_assignments, branch_df
        )
        
        # Create INMARKET results
        for branch_au, customers in customer_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                if distance_value is None:
                    distance_value = 0
                
                inmarket_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': round(float(distance_value), 2),
                    'TYPE': 'INMARKET'
                })
        
        # Collect unassigned customers for centralized processing
        unassigned_customer_indices.extend(unassigned)
    
    # Add customers that were never assigned to any cluster
    never_assigned = clustered_customers[clustered_customers['cluster'] == -1].index.tolist()
    unassigned_customer_indices.extend(never_assigned)
    
    # Remove duplicates
    unassigned_customer_indices = list(set(unassigned_customer_indices))
    
    print(f"Total unassigned customers for centralized processing: {len(unassigned_customer_indices)}")
    
    # Step 4: Create CENTRALIZED portfolios from unassigned customers
    centralized_results = []
    final_unassigned = []
    
    if unassigned_customer_indices:
        unassigned_customers_df = customer_df.loc[unassigned_customer_indices]
        
        centralized_assignments, remaining_customers = create_centralized_portfolios(
            unassigned_customers_df, branch_df
        )
        
        # Create CENTRALIZED results
        for branch_au, customers in centralized_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customer_df.loc[customer_idx]
                
                distance_value = customer.get('distance', 0)
                if distance_value is None:
                    distance_value = 0
                
                centralized_results.append({
                    'ECN': customer_data['ECN'],
                    'BILLINGCITY': customer_data['BILLINGCITY'],
                    'BILLINGSTATE': customer_data['BILLINGSTATE'],
                    'LAT_NUM': customer_data['LAT_NUM'],
                    'LON_NUM': customer_data['LON_NUM'],
                    'ASSIGNED_AU': branch_au,
                    'DISTANCE_TO_AU': round(float(distance_value), 2),
                    'TYPE': 'CENTRALIZED'
                })
        
        final_unassigned = remaining_customers
    
    # Combine all results
    all_results = inmarket_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Print comprehensive summary
    print(f"\n=== FINAL COMPREHENSIVE SUMMARY ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"INMARKET customers assigned: {len(inmarket_results)}")
    print(f"CENTRALIZED customers assigned: {len(centralized_results)}")
    print(f"Final unassigned customers: {len(final_unassigned)}")
    print(f"Total assigned customers: {len(result_df)}")
    
    if len(result_df) > 0:
        # Portfolio type summary
        type_summary = result_df.groupby('TYPE').agg({
            'ECN': 'count',
            'DISTANCE_TO_AU': ['mean', 'max']
        }).round(2)
        type_summary.columns = ['Customer_Count', 'Avg_Distance', 'Max_Distance']
        print("\nPortfolio Type Summary:")
        print(type_summary)
        
        # AU assignment summary by type
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
        
        # Number of unique AUs used
        print(f"Number of unique AUs used: {result_df['ASSIGNED_AU'].nunique()}")
    
    return result_df

# Usage:
# customer_au_assignments = create_customer_au_dataframe(customer_df, branch_df)
# print(customer_au_assignments.head())
# customer_au_assignments.to_csv('customer_au_assignments_hungarian.csv', index=False)
