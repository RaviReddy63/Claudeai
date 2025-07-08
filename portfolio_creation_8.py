def balance_inmarket_portfolios_to_minimum(result_df, branch_df, min_size=200, search_radius=50):
    """
    Balance INMARKET portfolios by moving customers from oversized portfolios to undersized ones
    Ensures all portfolios have at least min_size customers while maintaining donor portfolio above min_size
    """
    print(f"\nBalancing INMARKET portfolios to minimum size of {min_size}...")
    
    # Work with a copy
    balanced_result = result_df.copy()
    
    # Get INMARKET portfolios and their sizes
    inmarket_customers = balanced_result[balanced_result['TYPE'] == 'INMARKET'].copy()
    if len(inmarket_customers) == 0:
        print("No INMARKET customers found")
        return balanced_result
    
    portfolio_sizes = inmarket_customers['ASSIGNED_AU'].value_counts().to_dict()
    
    print("Current INMARKET portfolio sizes:")
    for au, size in sorted(portfolio_sizes.items()):
        print(f"  AU {au}: {size} customers")
    
    # Identify undersized portfolios
    undersized_portfolios = {au: size for au, size in portfolio_sizes.items() if size < min_size}
    oversized_portfolios = {au: size for au, size in portfolio_sizes.items() if size > min_size}
    
    if not undersized_portfolios:
        print("All INMARKET portfolios already meet minimum size requirement")
        return balanced_result
    
    print(f"\nUndersized portfolios: {len(undersized_portfolios)}")
    for au, size in undersized_portfolios.items():
        print(f"  AU {au}: {size} customers (needs {min_size - size} more)")
    
    print(f"\nOversized portfolios available for balancing: {len(oversized_portfolios)}")
    
    # Get branch coordinates for distance calculations
    branch_coords_dict = {}
    for _, branch in branch_df.iterrows():
        branch_coords_dict[branch['BRANCH_AU']] = {
            'lat': branch['BRANCH_LAT_NUM'],
            'lon': branch['BRANCH_LON_NUM']
        }
    
    total_moves = 0
    
    # Process each undersized portfolio
    for undersized_au, current_size in undersized_portfolios.items():
        needed_customers = min_size - current_size
        print(f"\nProcessing AU {undersized_au} (needs {needed_customers} customers)...")
        
        if undersized_au not in branch_coords_dict:
            print(f"  Warning: No coordinates found for AU {undersized_au}")
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
                available_customers = donor_size - min_size
                potential_donors.append({
                    'au': donor_au,
                    'distance': distance,
                    'available': available_customers,
                    'current_size': donor_size
                })
        
        # Sort donors by distance (closest first)
        potential_donors.sort(key=lambda x: x['distance'])
        
        print(f"  Found {len(potential_donors)} potential donor portfolios within {search_radius} miles")
        
        customers_acquired = 0
        
        # Try to get customers from donors
        for donor_info in potential_donors:
            if customers_acquired >= needed_customers:
                break
            
            donor_au = donor_info['au']
            max_transferable = min(
                donor_info['available'],
                needed_customers - customers_acquired
            )
            
            if max_transferable <= 0:
                continue
            
            print(f"    Checking donor AU {donor_au} (distance: {donor_info['distance']:.1f} miles, available: {max_transferable})")
            
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
            customers_to_transfer = min(max_transferable, len(donor_distances_to_recipient))
            
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
            
            print(f"      Transferred {customers_to_transfer} customers from AU {donor_au}")
        
        print(f"  AU {undersized_au}: Acquired {customers_acquired} customers (target was {needed_customers})")
    
    print(f"\nBalancing complete: {total_moves} customers moved")
    
    # Print final sizes
    final_portfolio_sizes = balanced_result[balanced_result['TYPE'] == 'INMARKET']['ASSIGNED_AU'].value_counts().to_dict()
    print("\nFinal INMARKET portfolio sizes:")
    for au, size in sorted(final_portfolio_sizes.items()):
        status = "✓" if size >= min_size else "⚠"
        print(f"  AU {au}: {size} customers {status}")
    
    return balanced_result


def constrained_clustering_with_radius(customer_df, min_size=200, max_size=240, max_radius=100):
    """
    Clustering with both size constraints AND radius constraint for centralized portfolios
    """
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
    
    print(f"Creating centralized clusters with max radius {max_radius} miles...")
    
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
                print(f"  Created cluster {cluster_id-1}: {len(current_cluster)} customers, radius: {cluster_radius:.1f} miles")
            else:
                # Cluster exceeds radius, try to split
                print(f"  Cluster exceeds radius ({cluster_radius:.1f} > {max_radius}), attempting to split...")
                
                # Try K-means splitting
                coords_array = np.array(current_coords)
                n_splits = min(3, len(current_cluster) // min_size)
                
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
                                print(f"    Split cluster {cluster_id-1}: {len(sub_indices)} customers, radius: {sub_radius:.1f} miles")
                
                # Mark remaining as unassigned for this iteration
                for idx in current_cluster:
                    if customers_clean.iloc[idx, customers_clean.columns.get_loc('cluster')] == -1:
                        unassigned_mask[idx] = False
        else:
            # Handle oversized cluster with splitting
            coords_array = np.array(current_coords)
            n_splits = min(3, len(current_cluster) // min_size)
            
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
                            print(f"  Split large cluster {cluster_id-1}: {len(sub_indices)} customers, radius: {sub_radius:.1f} miles")
            
            # Mark all as unassigned for this iteration
            for idx in current_cluster:
                unassigned_mask[idx] = False
        
        unassigned_count = np.count_nonzero(unassigned_mask)
    
    print(f"Centralized clustering complete: {len(final_clusters)} clusters created")
    
    return customers_clean, pd.DataFrame(final_clusters)


def create_centralized_clusters_with_radius_and_assign(unassigned_customers_df, branch_df, 
                                                     min_size=200, max_size=240, max_radius=100):
    """
    Create centralized clusters WITH radius constraint and assign to branches
    """
    print(f"\nCreating centralized clusters with radius constraint for {len(unassigned_customers_df)} customers...")
    print(f"Cluster constraints: min_size={min_size}, max_size={max_size}, max_radius={max_radius} miles")
    
    if len(unassigned_customers_df) == 0:
        return [], []
    
    # Step 1: Create clusters with radius constraint
    clustered_centralized, centralized_cluster_info = constrained_clustering_with_radius(
        unassigned_customers_df, min_size=min_size, max_size=max_size, max_radius=max_radius
    )
    
    centralized_results = []
    final_unassigned = []
    
    if len(centralized_cluster_info) > 0:
        print(f"Created {len(centralized_cluster_info)} centralized clusters with radius constraint")
        
        # Print cluster statistics
        print("\nCentralized Cluster Statistics (with radius constraint):")
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
        
        # Find customers that couldn't be clustered
        unassigned_centralized = clustered_centralized[
            clustered_centralized['cluster'] == -1
        ]
        final_unassigned = list(unassigned_centralized.index)
        
    else:
        print("No centralized clusters could be created with radius constraint")
        final_unassigned = list(unassigned_customers_df.index)
    
    print(f"\nCentralized assignment results (with radius constraint):")
    print(f"  - Customers in centralized clusters: {len(centralized_results)}")
    print(f"  - Remaining unassigned: {len(final_unassigned)}")
    
    return centralized_results, final_unassigned


def enhanced_customer_au_assignment_with_balancing(customer_df, branch_df):
    """
    Enhanced main function with portfolio balancing and centralized radius constraint
    """
    
    print(f"Starting enhanced assignment with {len(customer_df)} customers and {len(branch_df)} branches")
    
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
    
    # Step 5: Create CENTRALIZED clusters WITH radius constraint
    centralized_results = []
    final_unassigned = []
    
    if final_unassigned_after_proximity:
        remaining_unassigned_df = customer_df.loc[final_unassigned_after_proximity]
        
        centralized_results, final_unassigned = create_centralized_clusters_with_radius_and_assign(
            remaining_unassigned_df, branch_df, min_size=200, max_size=240, max_radius=100
        )
    
    # Combine results
    all_results = inmarket_results + proximity_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Step 6: Optimize INMARKET portfolios until convergence
    if len(result_df) > 0:
        result_df = optimize_inmarket_portfolios_until_convergence(result_df, branch_df)
    
    # Step 7: NEW - Balance INMARKET portfolios to minimum size
    if len(result_df) > 0:
        result_df = balance_inmarket_portfolios_to_minimum(result_df, branch_df, min_size=200)
    
    # Print summary
    print(f"\n=== FINAL ENHANCED SUMMARY ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"INMARKET customers assigned: {len(result_df[result_df['TYPE'] == 'INMARKET'])}")
    print(f"CENTRALIZED customers assigned: {len(result_df[result_df['TYPE'] == 'CENTRALIZED'])}")
    print(f"Final unassigned customers: {len(final_unassigned)}")
    print(f"Total assigned customers: {len(result_df)}")
    
    if len(result_df) > 0:
        # Portfolio size analysis
        inmarket_sizes = result_df[result_df['TYPE'] == 'INMARKET']['ASSIGNED_AU'].value_counts()
        print(f"\nINMARKET Portfolio Size Analysis:")
        print(f"  Portfolios meeting minimum (≥200): {sum(inmarket_sizes >= 200)}")
        print(f"  Portfolios below minimum (<200): {sum(inmarket_sizes < 200)}")
        if sum(inmarket_sizes < 200) > 0:
            print(f"  Smallest portfolio size: {inmarket_sizes.min()}")
        
        # Centralized cluster analysis
        centralized_df = result_df[result_df['TYPE'] == 'CENTRALIZED']
        if len(centralized_df) > 0 and 'CLUSTER_ID' in centralized_df.columns:
            cluster_radii = []
            for cluster_id in centralized_df['CLUSTER_ID'].unique():
                cluster_customers = centralized_df[centralized_df['CLUSTER_ID'] == cluster_id]
                coords = cluster_customers[['LAT_NUM', 'LON_NUM']].values
                radius = calculate_cluster_radius_vectorized(coords)
                cluster_radii.append(radius)
            
            print(f"\nCentralized Cluster Radius Analysis:")
            print(f"  Average radius: {np.mean(cluster_radii):.1f} miles")
            print(f"  Maximum radius: {np.max(cluster_radii):.1f} miles")
            print(f"  Clusters within 100-mile limit: {sum(np.array(cluster_radii) <= 100)}/{len(cluster_radii)}")
    
    return result_df


# Usage:
# enhanced_assignments = enhanced_customer_au_assignment_with_balancing(customer_df, branch_df)
# enhanced_assignments.to_csv('enhanced_customer_au_assignments.csv', index=False)
