import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

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

def geocode_address(street, city, state, max_retries=3):
    """
    Geocode an address to get latitude and longitude
    """
    geolocator = Nominatim(user_agent="customer_portfolio_assignment")
    
    # Construct address string
    address_parts = []
    if street and str(street).strip() and str(street).strip().lower() != 'nan':
        address_parts.append(str(street).strip())
    if city and str(city).strip() and str(city).strip().lower() != 'nan':
        address_parts.append(str(city).strip())
    if state and str(state).strip() and str(state).strip().lower() != 'nan':
        address_parts.append(str(state).strip())
    
    if not address_parts:
        return None, None
    
    address = ", ".join(address_parts)
    
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                return location.latitude, location.longitude
            else:
                # Try with just city and state if full address fails
                if len(address_parts) > 2:
                    city_state_address = ", ".join(address_parts[-2:])
                    location = geolocator.geocode(city_state_address, timeout=10)
                    if location:
                        return location.latitude, location.longitude
                return None, None
        except (GeocoderTimedOut, GeocoderUnavailable):
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            else:
                return None, None
    
    return None, None

def assign_geocoded_customers_to_portfolios(customers_without_coords, result_df, branch_df):
    """
    Geocode customers without coordinates and assign them to nearest existing portfolios
    With fallback logic for failed geocoding cases
    """
    print(f"\nProcessing {len(customers_without_coords)} customers without coordinates...")
    
    if len(customers_without_coords) == 0:
        return pd.DataFrame()
    
    geocoded_results = []
    failed_geocoding = []
    
    for idx, customer in customers_without_coords.iterrows():
        print(f"Geocoding customer {idx}...")
        
        # Extract address components
        street = customer.get('BILLINGSTREET', '')
        city = customer.get('BILLINGCITY', '')
        state = customer.get('BILLINGSTATE', '')
        
        # Attempt geocoding
        lat, lon = geocode_address(street, city, state)
        
        if lat is not None and lon is not None:
            print(f"  Successfully geocoded: {lat}, {lon}")
            
            # Find nearest portfolio (AU)
            min_distance = float('inf')
            nearest_au = None
            nearest_type = None
            
            # Check against all existing assignments
            for _, assigned_customer in result_df.iterrows():
                distance = haversine_distance(
                    lat, lon,
                    assigned_customer['LAT_NUM'], assigned_customer['LON_NUM']
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_au = assigned_customer['ASSIGNED_AU']
                    nearest_type = assigned_customer['TYPE']
            
            if nearest_au is not None:
                # Handle potential inf or None values for distance
                try:
                    if min_distance == float('inf') or min_distance is None:
                        distance_value = None
                    else:
                        distance_value = round(float(min_distance), 2)
                except (ValueError, TypeError):
                    distance_value = None
                
                geocoded_results.append({
                    'ECN': customer['ECN'],
                    'BILLINGCITY': customer['BILLINGCITY'],
                    'BILLINGSTATE': customer['BILLINGSTATE'],
                    'LAT_NUM': lat,
                    'LON_NUM': lon,
                    'ASSIGNED_AU': nearest_au,
                    'DISTANCE_TO_AU': distance_value,
                    'TYPE': nearest_type + '_GEOCODED'
                })
                print(f"  Assigned to AU {nearest_au} (distance: {min_distance:.2f} miles)")
            else:
                failed_geocoding.append(idx)
                print(f"  Could not find nearest AU")
        else:
            failed_geocoding.append(idx)
            print(f"  Geocoding failed")
        
        # Small delay to be respectful to geocoding service
        time.sleep(0.1)
    
    # Handle failed geocoding cases with fallback logic
    fallback_results = []
    if failed_geocoding:
        print(f"\nProcessing {len(failed_geocoding)} failed geocoding cases with fallback logic...")
        
        for idx in failed_geocoding:
            customer = customers_without_coords.loc[idx]
            city = customer.get('BILLINGCITY', '').strip()
            state = customer.get('BILLINGSTATE', '').strip()
            
            assigned = False
            assigned_au = None
            assigned_type = None
            assignment_method = None
            
            print(f"  Processing customer {idx} with city: {city}, state: {state}")
            
            # Fallback 1: Look for other customers in the same city
            if city and city.lower() != 'nan':
                city_matches = result_df[
                    (result_df['BILLINGCITY'].str.strip().str.lower() == city.lower()) |
                    (result_df['BILLINGCITY'].str.contains(city, case=False, na=False))
                ]
                
                if len(city_matches) > 0:
                    # Get the most common AU assignment in this city
                    au_counts = city_matches['ASSIGNED_AU'].value_counts()
                    most_common_au = au_counts.index[0]
                    
                    # Get the type of the most common assignment
                    au_type_match = city_matches[city_matches['ASSIGNED_AU'] == most_common_au]['TYPE'].iloc[0]
                    
                    assigned_au = most_common_au
                    assigned_type = au_type_match + '_CITY_MATCH'
                    assignment_method = 'city_match'
                    assigned = True
                    
                    print(f"    Found {len(city_matches)} customers in same city, assigned to AU {assigned_au}")
            
            # Fallback 2: Assign to any centralized portfolio
            if not assigned:
                centralized_portfolios = result_df[result_df['TYPE'].str.contains('CENTRALIZED', na=False)]
                
                if len(centralized_portfolios) > 0:
                    # Get AU with smallest centralized portfolio (for load balancing)
                    centralized_au_counts = centralized_portfolios['ASSIGNED_AU'].value_counts()
                    smallest_centralized_au = centralized_au_counts.index[-1]  # Last index has smallest count
                    
                    assigned_au = smallest_centralized_au
                    assigned_type = 'CENTRALIZED_FALLBACK'
                    assignment_method = 'centralized_fallback'
                    assigned = True
                    
                    print(f"    Assigned to centralized portfolio AU {assigned_au}")
                else:
                    # Fallback 3: Assign to any available AU (last resort)
                    if len(result_df) > 0:
                        # Get AU with smallest portfolio overall
                        all_au_counts = result_df['ASSIGNED_AU'].value_counts()
                        smallest_au = all_au_counts.index[-1]
                        
                        assigned_au = smallest_au
                        assigned_type = 'GENERAL_FALLBACK'
                        assignment_method = 'general_fallback'
                        assigned = True
                        
                        print(f"    Assigned to general portfolio AU {assigned_au}")
            
            if assigned:
                fallback_results.append({
                    'ECN': customer['ECN'],
                    'BILLINGCITY': customer['BILLINGCITY'],
                    'BILLINGSTATE': customer['BILLINGSTATE'],
                    'LAT_NUM': None,  # No coordinates available
                    'LON_NUM': None,  # No coordinates available
                    'ASSIGNED_AU': assigned_au,
                    'DISTANCE_TO_AU': None,  # Cannot calculate without coordinates
                    'TYPE': assigned_type
                })
            else:
                print(f"    Could not assign customer {idx} - no fallback options available")
    
    # Combine geocoded and fallback results
    all_results = geocoded_results + fallback_results
    final_df = pd.DataFrame(all_results)
    
    print(f"\nGeocoding and Fallback Summary:")
    print(f"Successfully geocoded and assigned: {len(geocoded_results)}")
    print(f"Assigned via fallback logic: {len(fallback_results)}")
    print(f"Total processed: {len(all_results)}")
    print(f"Still unassigned: {len(failed_geocoding) - len(fallback_results)}")
    
    if len(final_df) > 0:
        print("\nAssignments by Type:")
        type_summary = final_df.groupby('TYPE').agg({
            'ECN': 'count'
        })
        type_summary.columns = ['Customer_Count']
        print(type_summary)
        
        # Only show distance stats for customers with coordinates
        geocoded_only = final_df.dropna(subset=['LAT_NUM', 'LON_NUM'])
        if len(geocoded_only) > 0:
            print("\nDistance Statistics (geocoded customers only):")
            distance_summary = geocoded_only.groupby('TYPE').agg({
                'DISTANCE_TO_AU': ['mean', 'max']
            }).round(2)
            distance_summary.columns = ['Avg_Distance', 'Max_Distance']
            print(distance_summary)
    
    return final_df
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
    n_clusters = n_customers // max_size
    if n_clusters < 1:
        n_clusters = 1
    
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
    
    # Separate customers with and without coordinates
    customers_with_coords = customer_df.dropna(subset=['LAT_NUM', 'LON_NUM']).copy()
    customers_without_coords = customer_df[customer_df[['LAT_NUM', 'LON_NUM']].isnull().any(axis=1)].copy()
    
    print(f"Customers with coordinates: {len(customers_with_coords)}")
    print(f"Customers without coordinates: {len(customers_without_coords)}")
    
    # Step 1: Create INMARKET clusters (only for customers with coordinates)
    print("Step 1: Creating INMARKET clusters...")
    clustered_customers, cluster_info = constrained_clustering(customers_with_coords)
    
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
                customer_data = customers_with_coords.loc[customer_idx]
                
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
        unassigned_customers_df = customers_with_coords.loc[unassigned_customer_indices]
        
        centralized_assignments, remaining_customers = create_centralized_portfolios(
            unassigned_customers_df, branch_df
        )
        
        # Create CENTRALIZED results
        for branch_au, customers in centralized_assignments.items():
            for customer in customers:
                customer_idx = customer['customer_idx']
                customer_data = customers_with_coords.loc[customer_idx]
                
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
    
    # Combine all results from coordinate-based processing
    all_results = inmarket_results + centralized_results
    result_df = pd.DataFrame(all_results)
    
    # Step 5: Handle customers without coordinates by geocoding and assigning to nearest portfolio
    geocoded_results_df = pd.DataFrame()
    if len(customers_without_coords) > 0 and len(result_df) > 0:
        print("\nStep 5: Processing customers without coordinates...")
        geocoded_results_df = assign_geocoded_customers_to_portfolios(
            customers_without_coords, result_df, branch_df
        )
        
        # Combine with main results
        if len(geocoded_results_df) > 0:
            result_df = pd.concat([result_df, geocoded_results_df], ignore_index=True)
    
    # Print comprehensive summary
    print(f"\n=== FINAL COMPREHENSIVE SUMMARY ===")
    print(f"Total customers processed: {len(customer_df)}")
    print(f"INMARKET customers assigned: {len(inmarket_results)}")
    print(f"CENTRALIZED customers assigned: {len(centralized_results)}")
    print(f"Geocoded customers assigned: {len(geocoded_results_df)}")
    print(f"Final unassigned customers (with coordinates): {len(final_unassigned)}")
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
# customer_au_assignments.to_csv('customer_au_assignments_with_centralized_and_geocoded.csv', index=False)

# Note: To use geocoding functionality, you'll need to install geopy:
# pip install geopy
