import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# ===================================================================
# STEP 1: HELPER FUNCTIONS FOR DISTANCE CALCULATIONS
# ===================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance in miles between two geographic points
    This accounts for Earth's curvature to give accurate distances
    """
    R = 3959  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_cluster_radius(customers_coords):
    """
    Calculate maximum distance from centroid to any point in cluster
    This ensures clusters don't spread too far geographically
    """
    if len(customers_coords) <= 1:
        return 0
    
    centroid = customers_coords.mean(axis=0)
    max_distance = 0
    
    for coord in customers_coords:
        distance = haversine_distance(centroid[0], centroid[1], coord[0], coord[1])
        if distance > max_distance:
            max_distance = distance
    
    return max_distance

# ===================================================================
# STEP 2: CLUSTER SPLITTING LOGIC (FOR OVERSIZED CLUSTERS)
# ===================================================================

def split_large_cluster(cluster_indices, cluster_coords, customers_df, min_size, max_size, max_radius):
    """
    Split a large cluster into smaller valid clusters using k-means approach
    This handles cases where initial clustering creates clusters that are too big
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
    Finds two farthest points and splits cluster around them
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

# ===================================================================
# STEP 3: INITIAL CONSTRAINED CLUSTERING
# ===================================================================

def constrained_clustering(customer_df, min_size=200, max_size=280, max_radius=20):
    """
    Create initial clusters with size and radius constraints during cluster formation
    This creates geographic clusters based on customer locations
    """
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['cluster'] = -1  # Initialize as unassigned
    
    unassigned_customers = customers_clean.copy()
    cluster_id = 0
    final_clusters = []
    
    print(f"Step 3: Starting initial clustering with {len(unassigned_customers)} customers")
    
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
            print(f"  Rejected cluster with {len(current_cluster)} customers (too small)")
            
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
            print(f"  Cluster {cluster_id-1}: {len(current_cluster)} customers, radius: {cluster_radius:.2f} miles")
            
        else:
            # Too large - split into smaller clusters
            print(f"  Splitting large cluster of {len(current_cluster)} customers")
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
                    print(f"    Split cluster {cluster_id-1}: {len(split_cluster_indices)} customers, radius: {cluster_radius:.2f} miles")
            
            # Remove all customers from current cluster (assigned or not)
            unassigned_customers = unassigned_customers.drop(current_cluster)
    
    print(f"Step 3 Complete: {cluster_id} initial clusters, {len(unassigned_customers)} unassigned customers")
    
    return customers_clean, pd.DataFrame(final_clusters)

# ===================================================================
# STEP 4: ASSIGN INITIAL CLUSTERS TO BRANCHES
# ===================================================================

def assign_clusters_to_branches(clustered_customers, cluster_info, branch_df):
    """
    Step 4: Assign each geographic cluster to the nearest branch
    This determines which branches will be involved in the final solution
    """
    print("\nStep 4: Assigning initial clusters to branches...")
    
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
        
        print(f"  Cluster {cluster['cluster_id']} → Branch {assigned_branch} (distance: {min_distance:.2f} miles)")
    
    active_branches = pd.DataFrame(cluster_assignments)['assigned_branch'].unique()
    print(f"Step 4 Complete: {len(active_branches)} active branches identified")
    
    return pd.DataFrame(cluster_assignments)

# ===================================================================
# STEP 5: REFORM CLUSTERS WITH BRANCH CENTROIDS
# ===================================================================

def reform_clusters_with_branch_centroids(customer_df, cluster_branch_map, branch_df, min_size=200, max_size=280, max_radius=20):
    """
    Step 5: Reform clusters using assigned branches as centroids
    This is the key step - instead of geographic centroids, use branch locations
    """
    print("\nStep 5: Reforming clusters with branch centroids...")
    
    # Get active branches from Step 4
    active_branches = cluster_branch_map['assigned_branch'].unique()
    active_branch_df = branch_df[branch_df['BRANCH_AU'].isin(active_branches)].copy()
    
    print(f"  Reforming clusters using {len(active_branches)} branch centroids")
    
    # Clean customer data
    customers_clean = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_clean['final_cluster'] = -1  # Initialize as unassigned
    customers_clean['assigned_branch'] = None
    
    new_clusters = []
    cluster_id = 0
    
    # For each active branch, create a cluster using branch as centroid
    for _, branch in active_branch_df.iterrows():
        branch_id = branch['BRANCH_AU']
        branch_lat = branch['BRANCH_LAT_NUM']
        branch_lon = branch['BRANCH_LON_NUM']
        
        print(f"\n  Processing Branch {branch_id} at ({branch_lat:.4f}, {branch_lon:.4f})")
        
        # Find all unassigned customers within max_radius of this branch
        candidates = []
        for idx, customer in customers_clean.iterrows():
            if customer['final_cluster'] != -1:  # Skip already assigned customers
                continue
                
            distance = haversine_distance(
                branch_lat, branch_lon,
                customer['LAT_NUM'], customer['LON_NUM']
            )
            
            if distance <= max_radius:
                candidates.append((idx, customer, distance))
        
        # Sort candidates by distance from branch (closest first)
        candidates.sort(key=lambda x: x[2])
        
        print(f"    Found {len(candidates)} candidates within {max_radius} miles")
        
        # Select customers for this cluster (up to max_size)
        cluster_customers = []
        cluster_coords = []
        
        for candidate_idx, candidate_customer, distance in candidates[:max_size]:
            cluster_customers.append(candidate_idx)
            cluster_coords.append((candidate_customer['LAT_NUM'], candidate_customer['LON_NUM']))
        
        # Check if cluster meets minimum size requirement
        if len(cluster_customers) >= min_size:
            # Calculate actual radius (max distance from branch to any customer in cluster)
            actual_radius = 0
            for coord in cluster_coords:
                distance = haversine_distance(branch_lat, branch_lon, coord[0], coord[1])
                if distance > actual_radius:
                    actual_radius = distance
            
            # Assign customers to this cluster
            for customer_idx in cluster_customers:
                customers_clean.loc[customer_idx, 'final_cluster'] = cluster_id
                customers_clean.loc[customer_idx, 'assigned_branch'] = branch_id
            
            new_clusters.append({
                'cluster_id': cluster_id,
                'assigned_branch': branch_id,
                'size': len(cluster_customers),
                'radius': actual_radius,
                'centroid_lat': branch_lat,  # Branch location as centroid
                'centroid_lon': branch_lon,
                'branch_distance': 0  # Distance to branch is 0 since branch IS the centroid
            })
            
            print(f"    ✓ Created cluster {cluster_id}: {len(cluster_customers)} customers, radius: {actual_radius:.2f} miles")
            cluster_id += 1
            
        else:
            print(f"    ✗ Insufficient customers ({len(cluster_customers)} < {min_size}) for branch {branch_id}")
    
    # Count unassigned customers
    unassigned_count = len(customers_clean[customers_clean['final_cluster'] == -1])
    
    print(f"\nStep 5 Complete:")
    print(f"  Total clusters created: {cluster_id}")
    print(f"  Total customers assigned: {len(customers_clean[customers_clean['final_cluster'] != -1])}")
    print(f"  Unassigned customers: {unassigned_count}")
    
    return customers_clean, pd.DataFrame(new_clusters)

# ===================================================================
# STEP 6: COMPLETE WORKFLOW ORCHESTRATION
# ===================================================================

def create_branch_centered_portfolios(customer_df, branch_df):
    """
    Complete workflow: Execute all steps to create branch-centered portfolios
    """
    print("="*70)
    print("CUSTOMER PORTFOLIO CREATION WITH BRANCH-CENTERED CLUSTERING")
    print("="*70)
    
    # Step 3: Initial clustering based on geographic proximity
    clustered_customers, cluster_info = constrained_clustering(
        customer_df, min_size=200, max_size=280, max_radius=20
    )
    
    if len(cluster_info) == 0:
        print("No valid clusters found with the given constraints")
        return pd.DataFrame(), pd.DataFrame()
    
    # Step 4: Assign initial clusters to nearest branches
    cluster_branch_map = assign_clusters_to_branches(clustered_customers, cluster_info, branch_df)
    
    # Step 5: Reform clusters with branch centroids
    final_customers, final_clusters = reform_clusters_with_branch_centroids(
        customer_df, cluster_branch_map, branch_df, min_size=200, max_size=280, max_radius=20
    )
    
    return final_customers, final_clusters

# ===================================================================
# STEP 7: EXECUTION AND RESULTS
# ===================================================================

def execute_clustering_and_display_results(customer_df, branch_df):
    """
    Execute the complete clustering process and display comprehensive results
    """
    print("Starting branch-centered portfolio creation...")
    
    # Execute the complete workflow
    final_result_df, final_cluster_summary = create_branch_centered_portfolios(customer_df, branch_df)
    
    # Display final results
    if len(final_result_df) > 0:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        assigned_customers = final_result_df[final_result_df['final_cluster'] != -1]
        
        print(f"\nOverall Summary:")
        print(f"  Total customers processed: {len(customer_df):,}")
        print(f"  Customers assigned to portfolios: {len(assigned_customers):,}")
        print(f"  Assignment rate: {len(assigned_customers)/len(customer_df)*100:.1f}%")
        print(f"  Number of branch portfolios: {len(final_cluster_summary)}")
        print(f"  Unassigned customers: {len(final_result_df) - len(assigned_customers):,}")
        
        print(f"\nDetailed Cluster Information:")
        print("-" * 60)
        for _, cluster in final_cluster_summary.iterrows():
            print(f"Branch {cluster['assigned_branch']}: {cluster['size']:,} customers, "
                  f"{cluster['radius']:.1f} mile radius")
        
        print(f"\nPortfolio Distribution by Branch:")
        print("-" * 40)
        portfolio_sizes = assigned_customers.groupby('assigned_branch').size().sort_values(ascending=False)
        for branch, size in portfolio_sizes.items():
            print(f"  Branch {branch}: {size:,} customers")
        
        print(f"\nConstraint Validation:")
        print("-" * 30)
        size_violations = final_cluster_summary[(final_cluster_summary['size'] < 200) | 
                                               (final_cluster_summary['size'] > 280)]
        radius_violations = final_cluster_summary[final_cluster_summary['radius'] > 20]
        
        print(f"  Size constraint violations: {len(size_violations)}")
        print(f"  Radius constraint violations: {len(radius_violations)}")
        print(f"  All constraints satisfied: {'✓' if len(size_violations) == 0 and len(radius_violations) == 0 else '✗'}")
        
        print(f"\nSample Customer Assignments:")
        print("-" * 40)
        if 'ECN' in assigned_customers.columns:
            sample_columns = ['ECN', 'final_cluster', 'assigned_branch', 'BILLINGCITY', 'BILLINGSTATE']
            available_columns = [col for col in sample_columns if col in assigned_customers.columns]
            sample_data = assigned_customers[available_columns].head(10)
            print(sample_data.to_string(index=False))
        else:
            print("Customer ID column not found - showing first few rows:")
            print(assigned_customers.head(10).to_string())
        
    else:
        print("No customer portfolios could be created with the given constraints")
    
    return final_result_df, final_cluster_summary

# ===================================================================
# MAIN EXECUTION
# ===================================================================

# Execute the complete clustering process
# Uncomment the line below to run (assumes customer_df and branch_df are defined)
# final_result_df, final_cluster_summary = execute_clustering_and_display_results(customer_df, branch_df)
