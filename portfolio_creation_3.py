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
    
    print(f"Final result: {cluster_id} valid clusters, {len(unassigned_customers)} unassigned customers")
    
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
            'distance_to_branch': min_distance,
            'cluster_size': cluster['size'],
            'cluster_radius': cluster['radius']
        })
    
    return pd.DataFrame(cluster_assignments)

def reassign_customers_by_distance(clustered_customers, cluster_info, branch_df, max_distance=20, max_customers_per_branch=280):
    """
    Step 5: Reassign customers based on direct distance to identified branches
    with capacity constraints and overflow handling
    """
    # Get unique branches that were assigned clusters
    identified_branches = cluster_info['assigned_branch'].unique()
    identified_branch_coords = branch_df[branch_df['BRANCH_AU'].isin(identified_branches)].copy()
    
    print(f"Identified branches: {list(identified_branches)}")
    
    # Only work with customers that were successfully clustered
    customers_to_reassign = clustered_customers[clustered_customers['cluster'] != -1].copy()
    
    # Calculate distance from each customer to each identified branch
    customer_branch_distances = []
    
    for _, customer in customers_to_reassign.iterrows():
        customer_distances = []
        
        for _, branch in identified_branch_coords.iterrows():
            distance = haversine_distance(
                customer['LAT_NUM'], customer['LON_NUM'],
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
            )
            customer_distances.append({
                'customer_id': customer.name,  # Using index as customer ID
                'ecn': customer['ECN'],
                'branch_au': branch['BRANCH_AU'],
                'distance': distance
            })
        
        customer_branch_distances.extend(customer_distances)
    
    # Convert to DataFrame for easier manipulation
    distance_df = pd.DataFrame(customer_branch_distances)
    
    # Step 5a: Initial assignment - assign each customer to closest branch within max_distance
    initial_assignments = {}
    unassigned_customers = []
    
    for customer_id in customers_to_reassign.index:
        customer_distances = distance_df[distance_df['customer_id'] == customer_id].sort_values('distance')
        
        # Find closest branch within max_distance
        closest_within_range = customer_distances[customer_distances['distance'] <= max_distance]
        
        if len(closest_within_range) > 0:
            closest_branch = closest_within_range.iloc[0]['branch_au']
            distance = closest_within_range.iloc[0]['distance']
            
            if closest_branch not in initial_assignments:
                initial_assignments[closest_branch] = []
            
            initial_assignments[closest_branch].append({
                'customer_id': customer_id,
                'ecn': closest_within_range.iloc[0]['ecn'],
                'distance': distance
            })
        else:
            unassigned_customers.append(customer_id)
    
    print(f"Initial assignment completed. {len(unassigned_customers)} customers remain unassigned (no branch within {max_distance} miles)")
    
    # Step 5b: Handle capacity constraints and rebalancing
    def rebalance_branch_assignments(assignments, max_customers):
        """Handle overflow by moving farthest customers to their next closest branch"""
        changes_made = True
        iteration = 0
        
        while changes_made and iteration < 100:  # Prevent infinite loops
            changes_made = False
            iteration += 1
            
            for branch_au, customers in list(assignments.items()):
                if len(customers) > max_customers:
                    # Sort customers by distance (farthest first)
                    customers_sorted = sorted(customers, key=lambda x: x['distance'], reverse=True)
                    
                    # Remove the farthest customer
                    removed_customer = customers_sorted[0]
                    assignments[branch_au] = customers_sorted[1:]
                    
                    print(f"Iteration {iteration}: Removing customer {removed_customer['ecn']} from branch {branch_au} (distance: {removed_customer['distance']:.2f} miles)")
                    
                    # Find next closest branch for this customer
                    customer_id = removed_customer['customer_id']
                    customer_distances = distance_df[distance_df['customer_id'] == customer_id].sort_values('distance')
                    
                    # Try to assign to next closest branch within range
                    assigned_to_new_branch = False
                    for _, row in customer_distances.iterrows():
                        candidate_branch = row['branch_au']
                        candidate_distance = row['distance']
                        
                        # Skip current branch and branches outside range
                        if candidate_branch == branch_au or candidate_distance > max_distance:
                            continue
                        
                        # Assign to this branch
                        if candidate_branch not in assignments:
                            assignments[candidate_branch] = []
                        
                        assignments[candidate_branch].append({
                            'customer_id': customer_id,
                            'ecn': removed_customer['ecn'],
                            'distance': candidate_distance
                        })
                        
                        print(f"  -> Reassigned to branch {candidate_branch} (distance: {candidate_distance:.2f} miles)")
                        assigned_to_new_branch = True
                        changes_made = True
                        break
                    
                    if not assigned_to_new_branch:
                        unassigned_customers.append(customer_id)
                        print(f"  -> Could not reassign customer {removed_customer['ecn']} - added to unassigned list")
        
        return assignments
    
    # Perform rebalancing
    print("\nStarting capacity rebalancing...")
    final_assignments = rebalance_branch_assignments(initial_assignments, max_customers_per_branch)
    
    # Step 5c: Create final results DataFrame
    final_customer_assignments = []
    
    for branch_au, customers in final_assignments.items():
        for customer in customers:
            customer_data = customers_to_reassign.loc[customer['customer_id']].copy()
            final_customer_assignments.append({
                'ECN': customer['ecn'],
                'customer_id': customer['customer_id'],
                'assigned_branch': branch_au,
                'distance_to_branch': customer['distance'],
                'LAT_NUM': customer_data['LAT_NUM'],
                'LON_NUM': customer_data['LON_NUM'],
                'BILLINGCITY': customer_data['BILLINGCITY'],
                'BILLINGSTATE': customer_data['BILLINGSTATE']
            })
    
    final_df = pd.DataFrame(final_customer_assignments)
    
    # Create summary statistics
    branch_summary = []
    for branch_au in identified_branches:
        branch_customers = [c for c in final_assignments.get(branch_au, [])]
        if branch_customers:
            avg_distance = np.mean([c['distance'] for c in branch_customers])
            max_distance_in_branch = max([c['distance'] for c in branch_customers])
        else:
            avg_distance = 0
            max_distance_in_branch = 0
        
        branch_summary.append({
            'branch_au': branch_au,
            'customer_count': len(branch_customers),
            'avg_distance': avg_distance,
            'max_distance': max_distance_in_branch
        })
    
    summary_df = pd.DataFrame(branch_summary)
    
    return final_df, summary_df, unassigned_customers

def create_customer_portfolios_enhanced(customer_df, branch_df):
    """
    Enhanced portfolio creation with distance-based reassignment
    """
    print("="*60)
    print("STEP 1-4: CONSTRAINED CLUSTERING AND BRANCH ASSIGNMENT")
    print("="*60)
    
    # Steps 1-4: Original clustering and branch assignment
    clustered_customers, cluster_info = constrained_clustering(
        customer_df, min_size=200, max_size=280, max_radius=20
    )
    
    if len(cluster_info) == 0:
        print("No valid clusters found with the given constraints")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    
    cluster_branch_map = assign_clusters_to_branches(clustered_customers, cluster_info, branch_df)
    
    print("\n" + "="*60)
    print("STEP 5: DISTANCE-BASED CUSTOMER REASSIGNMENT")
    print("="*60)
    
    # Step 5: Enhanced distance-based reassignment
    final_assignments, branch_summary, unassigned = reassign_customers_by_distance(
        clustered_customers, cluster_branch_map, branch_df, max_distance=20, max_customers_per_branch=280
    )
    
    return final_assignments, branch_summary, cluster_branch_map, unassigned

# Main execution function
def run_complete_analysis(customer_df, branch_df):
    """
    Run the complete customer portfolio clustering analysis
    """
    print("Starting Enhanced Customer Portfolio Creation...")
    print("="*80)
    
    # Execute the enhanced clustering
    result_df, branch_summary, cluster_summary, unassigned_customers = create_customer_portfolios_enhanced(customer_df, branch_df)
    
    # Display enhanced results
    if len(result_df) > 0:
        print(f"\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Total customers assigned: {len(result_df)}")
        print(f"Customers unassigned: {len(unassigned_customers)}")
        print(f"Number of branches used: {len(branch_summary)}")
        
        print("\nBranch Summary (after distance-based reassignment):")
        print(branch_summary.round(2))
        
        print("\nSample customer assignments:")
        print(result_df[['ECN', 'assigned_branch', 'distance_to_branch', 'BILLINGCITY', 'BILLINGSTATE']].head(10))
        
        print("\nBranch capacity check:")
        capacity_check = result_df.groupby('assigned_branch').size()
        print(capacity_check)
        
        over_capacity = capacity_check[capacity_check > 280]
        if len(over_capacity) > 0:
            print(f"\nWARNING: Branches over capacity (>280): {list(over_capacity.index)}")
        else:
            print("\n✓ All branches are within capacity limits (≤280 customers)")
            
        print(f"\nDistance compliance check:")
        distance_violations = result_df[result_df['distance_to_branch'] > 20]
        if len(distance_violations) > 0:
            print(f"WARNING: {len(distance_violations)} customers assigned beyond 20-mile limit")
        else:
            print("✓ All assigned customers are within 20-mile distance limit")
        
        # Additional statistics
        print(f"\nAdditional Statistics:")
        print(f"Average distance to branch: {result_df['distance_to_branch'].mean():.2f} miles")
        print(f"Maximum distance to branch: {result_df['distance_to_branch'].max():.2f} miles")
        print(f"Standard deviation of distances: {result_df['distance_to_branch'].std():.2f} miles")
        
        return {
            'customer_assignments': result_df,
            'branch_summary': branch_summary,
            'cluster_summary': cluster_summary,
            'unassigned_customers': unassigned_customers
        }
    else:
        print("No customer portfolios could be created with the given constraints")
        return None

# Usage example (uncomment and modify variable names as needed):
# results = run_complete_analysis(customer_df, branch_df)
# 
# if results:
#     # Access the results
#     customer_assignments = results['customer_assignments']
#     branch_summary = results['branch_summary']
#     cluster_summary = results['cluster_summary']
#     unassigned_customers = results['unassigned_customers']
#     
#     # Save results to files
#     customer_assignments.to_csv('customer_assignments.csv', index=False)
#     branch_summary.to_csv('branch_summary.csv', index=False)
#     cluster_summary.to_csv('cluster_summary.csv', index=False)
