import pandas as pd
import numpy as np
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

def reassign_customers_by_distance(clustered_customers, cluster_assignments, branch_df, max_distance=20, max_customers_per_branch=280):
    """Reassign customers based on direct distance to branches with capacity constraints"""
    
    # Get identified branches from cluster assignments
    identified_branches = cluster_assignments['assigned_branch'].unique()
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    print(f"Identified branches: {list(identified_branches)}")
    print(f"Available branch coordinates: {len(identified_branch_coords)}")
    
    customers_to_reassign = clustered_customers[clustered_customers['cluster'] != -1].copy()
    print(f"Customers to reassign: {len(customers_to_reassign)}")
    
    # Calculate distances and assign to closest branch within range
    customer_assignments = {}
    unassigned_customers = []
    
    for _, customer in customers_to_reassign.iterrows():
        distances = []
        for _, branch in identified_branch_coords.iterrows():
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            distances.append((branch['BRANCH_AU'], distance))
        
        # Sort by distance and find closest within range
        distances.sort(key=lambda x: x[1])
        assigned = False
        
        for branch_au, distance in distances:
            if distance <= max_distance:
                if branch_au not in customer_assignments:
                    customer_assignments[branch_au] = []
                
                # Check capacity before adding
                if len(customer_assignments[branch_au]) < max_customers_per_branch:
                    customer_assignments[branch_au].append({
                        'customer_idx': customer.name,
                        'distance': distance
                    })
                    assigned = True
                    break
        
        if not assigned:
            unassigned_customers.append(customer.name)
    
    # Sort customers within each branch by distance (closest first)
    for branch_au in customer_assignments:
        customer_assignments[branch_au].sort(key=lambda x: x['distance'])
    
    return customer_assignments, unassigned_customers

def create_centralized_portfolios(unassigned_customers_df, branch_df, max_size=280):
    """
    Create centralized portfolios from unassigned customers
    No radius constraint, only size constraint
    """
    print(f"\nCreating centralized portfolios from {len(unassigned_customers_df)} unassigned customers...")
    
    if len(unassigned_customers_df) == 0:
        return {}, []
    
    # Get coordinates for clustering
    coords = unassigned_customers_df[['LAT_NUM', 'LON_NUM']].values
    customer_indices = unassigned_customers_df.index.tolist()
    
    # Determine number of clusters needed
    n_customers = len(unassigned_customers_df)
    n_clusters = max(1, n_customers // max_size)
    
    print(f"Creating {n_clusters} centralized clusters from {n_customers} customers")
    
    centralized_assignments = {}
    remaining_customers = []
    
    if n_clusters == 1:
        # If only one cluster possible, assign all to nearest branch
        if n_customers <= max_size:
            # Calculate centroid of all customers
            centroid_lat = np.mean(coords[:, 0])
            centroid_lon = np.mean(coords[:, 1])
            
            # Find nearest branch to centroid
            min_distance = float('inf')
            assigned_branch = None
            
            for _, branch in branch_df.iterrows():
                distance = haversine_distance(
                    centroid_lat, centroid_lon,
                    branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                )
                
                if distance < min_distance:
                    min_distance = distance
                    assigned_branch = branch['BRANCH_AU']
            
            # Assign all customers to this branch
            centralized_assignments[assigned_branch] = []
            for idx in customer_indices:
                customer_data = unassigned_customers_df.loc[idx]
                distance_to_branch = haversine_distance(
                    customer_data['LAT_NUM'], customer_data['LON_NUM'],
                    branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LAT_NUM'].iloc[0],
                    branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LON_NUM'].iloc[0]
                )
                
                centralized_assignments[assigned_branch].append({
                    'customer_idx': idx,
                    'distance': distance_to_branch
                })
        else:
            # Too many customers for one cluster, put excess as remaining
            remaining_customers = customer_indices[max_size:]
            
            # Assign first max_size customers
            selected_customers = customer_indices[:max_size]
            selected_coords = coords[:max_size]
            
            centroid_lat = np.mean(selected_coords[:, 0])
            centroid_lon = np.mean(selected_coords[:, 1])
            
            # Find nearest branch
            min_distance = float('inf')
            assigned_branch = None
            
            for _, branch in branch_df.iterrows():
                distance = haversine_distance(
                    centroid_lat, centroid_lon,
                    branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                )
                
                if distance < min_distance:
                    min_distance = distance
                    assigned_branch = branch['BRANCH_AU']
            
            centralized_assignments[assigned_branch] = []
            for idx in selected_customers:
                customer_data = unassigned_customers_df.loc[idx]
                distance_to_branch = haversine_distance(
                    customer_data['LAT_NUM'], customer_data['LON_NUM'],
                    branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LAT_NUM'].iloc[0],
                    branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LON_NUM'].iloc[0]
                )
                
                centralized_assignments[assigned_branch].append({
                    'customer_idx': idx,
                    'distance': distance_to_branch
                })
                
    else:
        # Multiple clusters needed
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = [customer_indices[i] for i in range(len(customer_indices)) if cluster_mask[i]]
            cluster_coords = coords[cluster_mask]
            
            if len(cluster_indices) <= max_size:
                # Calculate cluster centroid
                centroid_lat = np.mean(cluster_coords[:, 0])
                centroid_lon = np.mean(cluster_coords[:, 1])
                
                # Find nearest branch to centroid
                min_distance = float('inf')
                assigned_branch = None
                
                for _, branch in branch_df.iterrows():
                    distance = haversine_distance(
                        centroid_lat, centroid_lon,
                        branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        assigned_branch = branch['BRANCH_AU']
                
                # Assign customers to this branch
                if assigned_branch not in centralized_assignments:
                    centralized_assignments[assigned_branch] = []
                
                for idx in cluster_indices:
                    customer_data = unassigned_customers_df.loc[idx]
                    distance_to_branch = haversine_distance(
                        customer_data['LAT_NUM'], customer_data['LON_NUM'],
                        branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LAT_NUM'].iloc[0],
                        branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LON_NUM'].iloc[0]
                    )
                    
                    centralized_assignments[assigned_branch].append({
                        'customer_idx': idx,
                        'distance': distance_to_branch
                    })
            else:
                # Cluster too large, keep only max_size customers, rest go to remaining
                cluster_indices_trimmed = cluster_indices[:max_size]
                remaining_customers.extend(cluster_indices[max_size:])
                
                cluster_coords_trimmed = cluster_coords[:max_size]
                centroid_lat = np.mean(cluster_coords_trimmed[:, 0])
                centroid_lon = np.mean(cluster_coords_trimmed[:, 1])
                
                # Find nearest branch
                min_distance = float('inf')
                assigned_branch = None
                
                for _, branch in branch_df.iterrows():
                    distance = haversine_distance(
                        centroid_lat, centroid_lon,
                        branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        assigned_branch = branch['BRANCH_AU']
                
                if assigned_branch not in centralized_assignments:
                    centralized_assignments[assigned_branch] = []
                
                for idx in cluster_indices_trimmed:
                    customer_data = unassigned_customers_df.loc[idx]
                    distance_to_branch = haversine_distance(
                        customer_data['LAT_NUM'], customer_data['LON_NUM'],
                        branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LAT_NUM'].iloc[0],
                        branch_df[branch_df['BRANCH_AU'] == assigned_branch]['BRANCH_LON_NUM'].iloc[0]
                    )
                    
                    centralized_assignments[assigned_branch].append({
                        'customer_idx': idx,
                        'distance': distance_to_branch
                    })
    
    print(f"Created {len(centralized_assignments)} centralized portfolios")
    print(f"Remaining unassigned customers: {len(remaining_customers)}")
    
    return centralized_assignments, remaining_customers

def create_customer_au_dataframe(customer_df, branch_df):
    """
    Main function to create customer portfolio with AU assignments
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
        
        # Step 3: Reassign customers by distance for INMARKET
        print("Step 3: Reassigning INMARKET customers by distance...")
        customer_assignments, unassigned = reassign_customers_by_distance(
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
# customer_au_assignments.to_csv('customer_au_assignments_with_centralized.csv', index=False)
