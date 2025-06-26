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

def create_customer_au_dataframe(customer_df, branch_df):
    """
    Main function to create customer portfolio with AU assignments
    Returns: DataFrame with customer details and assigned AU
    """
    
    print(f"Starting with {len(customer_df)} customers and {len(branch_df)} branches")
    
    # Step 1: Create clusters
    print("Step 1: Creating clusters...")
    clustered_customers, cluster_info = constrained_clustering(customer_df)
    
    if len(cluster_info) == 0:
        print("No valid clusters found")
        return pd.DataFrame()
    
    print(f"Created {len(cluster_info)} clusters")
    
    # Step 2: Assign clusters to branches
    print("Step 2: Assigning clusters to branches...")
    cluster_assignments = assign_clusters_to_branches(clustered_customers, cluster_info, branch_df)
    print(f"Cluster assignments created: {len(cluster_assignments)}")
    
    # Step 3: Reassign customers by distance
    print("Step 3: Reassigning customers by distance...")
    customer_assignments, unassigned = reassign_customers_by_distance(
        clustered_customers, cluster_assignments, branch_df
    )
    
    # Step 4: Create final DataFrame with customer details and AU assignment
    print("Step 4: Creating final DataFrame...")
    final_results = []
    
    for branch_au, customers in customer_assignments.items():
        for customer in customers:
            customer_idx = customer['customer_idx']
            customer_data = customer_df.loc[customer_idx]
            
            # Handle potential None values and ensure proper data types
            distance_value = customer.get('distance', 0)
            if distance_value is None:
                distance_value = 0
            
            final_results.append({
                'ECN': customer_data['ECN'],
                'BILLINGCITY': customer_data['BILLINGCITY'],
                'BILLINGSTATE': customer_data['BILLINGSTATE'],
                'LAT_NUM': customer_data['LAT_NUM'],
                'LON_NUM': customer_data['LON_NUM'],
                'ASSIGNED_AU': branch_au,
                'DISTANCE_TO_AU': round(float(distance_value), 2)
            })
    
    result_df = pd.DataFrame(final_results)
    
    # Print summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total customers assigned: {len(result_df)}")
    print(f"Customers unassigned: {len(unassigned)}")
    print(f"Number of AUs used: {result_df['ASSIGNED_AU'].nunique() if len(result_df) > 0 else 0}")
    
    if len(result_df) > 0:
        # AU assignment summary
        au_summary = result_df.groupby('ASSIGNED_AU').agg({
            'ECN': 'count',
            'DISTANCE_TO_AU': ['mean', 'max']
        }).round(2)
        au_summary.columns = ['Customer_Count', 'Avg_Distance', 'Max_Distance']
        print("\nAU Assignment Summary:")
        print(au_summary)
        
        # Verify nearest branch assignment
        print(f"\nDistance verification:")
        print(f"Average distance to assigned AU: {result_df['DISTANCE_TO_AU'].mean():.2f} miles")
        print(f"Maximum distance to assigned AU: {result_df['DISTANCE_TO_AU'].max():.2f} miles")
    
    return result_df

# Usage:
# customer_au_assignments = create_customer_au_dataframe(customer_df, branch_df)
# print(customer_au_assignments.head())
# customer_au_assignments.to_csv('customer_au_assignments.csv', index=False)
