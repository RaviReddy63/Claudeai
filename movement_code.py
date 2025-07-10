import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in miles"""
    try:
        R = 3959  # Earth's radius in miles
        
        # Handle NaN values - return None instead of converting to 0
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return None
        
        # Convert to float
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Calculate differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    except (ValueError, TypeError) as e:
        print(f"Error in distance calculation: {e}")
        return None

def calculate_max_customer_distance_final(tagged_to_au, new_au, customer_au_assignments, branch_df, client_groups_df):
    """
    Calculate maximum distance from TAGGED_TO_AU to customers of NEW_AU
    Uses specified coordinate column names
    """
    if pd.isna(tagged_to_au) or pd.isna(new_au):
        return None
    
    # Step 1: Get coordinates of TAGGED_TO_AU (where portfolio is moving TO)
    tagged_au_info = branch_df[branch_df['BRANCH_AU'] == tagged_to_au]
    if len(tagged_au_info) == 0:
        print(f"Warning: No branch found for TAGGED_TO_AU: {tagged_to_au}")
        return None
    
    tagged_au_lat = tagged_au_info.iloc[0]['BRANCH_LAT_NUM']
    tagged_au_lon = tagged_au_info.iloc[0]['BRANCH_LON_NUM']
    
    if pd.isna(tagged_au_lat) or pd.isna(tagged_au_lon):
        print(f"Warning: No coordinates for TAGGED_TO_AU: {tagged_to_au}")
        return None
    
    # Step 2: Get customers assigned to NEW_AU (the original portfolio)
    customers_for_new_au = customer_au_assignments[
        customer_au_assignments['ASSIGNED_AU'] == new_au
    ]['CG_ECN'].dropna().tolist()
    
    if not customers_for_new_au:
        print(f"Warning: No customers found for NEW_AU: {new_au}")
        return None
    
    # Step 3: Get customer coordinates
    customer_data = client_groups_df[client_groups_df['CG_ECN'].isin(customers_for_new_au)]
    
    if len(customer_data) == 0:
        print(f"Warning: No customer data found for NEW_AU: {new_au}")
        return None
    
    # Step 4: Calculate distances from TAGGED_TO_AU to each customer of NEW_AU
    distances = []
    customers_processed = 0
    customers_with_coords = 0
    
    for _, customer in customer_data.iterrows():
        customers_processed += 1
        cust_lat = customer['LAT_NUM']
        cust_lon = customer['LON_NUM']
        
        if pd.notna(cust_lat) and pd.notna(cust_lon):
            customers_with_coords += 1
            distance = haversine_distance(tagged_au_lat, tagged_au_lon, cust_lat, cust_lon)
            if distance is not None and distance >= 0:
                distances.append(distance)
    
    print(f"NEW_AU {new_au}: {customers_processed} customers total, {customers_with_coords} with coordinates, {len(distances)} valid distances")
    
    # Step 5: Return maximum distance - IMPROVED ERROR HANDLING
    if not distances:  # Check if list is empty
        print(f"Warning: No valid distances calculated for NEW_AU: {new_au}")
        return None
    
    try:
        max_distance = max(distances)
        print(f"Max distance for NEW_AU {new_au}: {max_distance:.2f} miles")
        return max_distance
    except ValueError as e:
        print(f"Error calculating max distance for NEW_AU {new_au}: {e}")
        return None

# Alternative approach using numpy for better performance and error handling
def calculate_max_customer_distance_numpy(tagged_to_au, new_au, customer_au_assignments, branch_df, client_groups_df):
    """
    Calculate maximum distance using numpy for better performance
    """
    if pd.isna(tagged_to_au) or pd.isna(new_au):
        return None
    
    # Step 1: Get coordinates of TAGGED_TO_AU
    tagged_au_info = branch_df[branch_df['BRANCH_AU'] == tagged_to_au]
    if len(tagged_au_info) == 0:
        return None
    
    tagged_au_lat = tagged_au_info.iloc[0]['BRANCH_LAT_NUM']
    tagged_au_lon = tagged_au_info.iloc[0]['BRANCH_LON_NUM']
    
    if pd.isna(tagged_au_lat) or pd.isna(tagged_au_lon):
        return None
    
    # Step 2: Get customers assigned to NEW_AU
    customers_for_new_au = customer_au_assignments[
        customer_au_assignments['ASSIGNED_AU'] == new_au
    ]['CG_ECN'].dropna().tolist()
    
    if not customers_for_new_au:
        return None
    
    # Step 3: Get customer coordinates
    customer_data = client_groups_df[client_groups_df['CG_ECN'].isin(customers_for_new_au)]
    
    if len(customer_data) == 0:
        return None
    
    # Step 4: Filter customers with valid coordinates
    valid_customers = customer_data.dropna(subset=['LAT_NUM', 'LON_NUM'])
    
    if len(valid_customers) == 0:
        return None
    
    # Step 5: Calculate distances using numpy arrays
    try:
        # Convert to numpy arrays for vectorized calculation
        customer_lats = valid_customers['LAT_NUM'].values
        customer_lons = valid_customers['LON_NUM'].values
        
        # Calculate distances for all customers at once
        distances = []
        for i in range(len(customer_lats)):
            dist = haversine_distance(tagged_au_lat, tagged_au_lon, customer_lats[i], customer_lons[i])
            if dist is not None and dist >= 0:
                distances.append(dist)
        
        # Return max distance with safe handling
        if distances:
            return np.max(distances)
        else:
            return None
            
    except Exception as e:
        print(f"Error in numpy distance calculation: {e}")
        return None

# Alternative approach using pandas operations
def calculate_max_customer_distance_pandas(tagged_to_au, new_au, customer_au_assignments, branch_df, client_groups_df):
    """
    Calculate maximum distance using pandas operations
    """
    if pd.isna(tagged_to_au) or pd.isna(new_au):
        return None
    
    try:
        # Get branch coordinates
        branch_coords = branch_df[branch_df['BRANCH_AU'] == tagged_to_au][['BRANCH_LAT_NUM', 'BRANCH_LON_NUM']]
        if branch_coords.empty:
            return None
        
        branch_lat, branch_lon = branch_coords.iloc[0]
        if pd.isna(branch_lat) or pd.isna(branch_lon):
            return None
        
        # Get customer assignments
        customers = customer_au_assignments[customer_au_assignments['ASSIGNED_AU'] == new_au]['CG_ECN'].dropna()
        if customers.empty:
            return None
        
        # Get customer coordinates
        customer_coords = client_groups_df[
            client_groups_df['CG_ECN'].isin(customers)
        ][['LAT_NUM', 'LON_NUM']].dropna()
        
        if customer_coords.empty:
            return None
        
        # Calculate distances
        distances = customer_coords.apply(
            lambda row: haversine_distance(branch_lat, branch_lon, row['LAT_NUM'], row['LON_NUM']),
            axis=1
        )
        
        # Filter valid distances and return max
        valid_distances = distances.dropna()
        valid_distances = valid_distances[valid_distances >= 0]
        
        if valid_distances.empty:
            return None
        
        return valid_distances.max()
        
    except Exception as e:
        print(f"Error in pandas distance calculation: {e}")
        return None

def add_movement_analysis_improved(tagging_results, customer_au_assignments, branch_df, client_groups_df):
    """
    Add movement analysis with improved error handling
    """
    print("=== ADDING MOVEMENT ANALYSIS (IMPROVED) ===")
    
    # Create a copy to avoid modifying the original
    results_with_movement = tagging_results.copy()
    
    # Initialize new columns
    results_with_movement['MAX_CUSTOMER_DISTANCE_MILES'] = None
    results_with_movement['MOVEMENT'] = 'NO MOVEMENT REQUIRED'  # Default value
    
    # Validate required columns
    required_cols = ['TAGGED_TO_AU', 'NEW_AU']
    missing_cols = [col for col in required_cols if col not in results_with_movement.columns]
    if missing_cols:
        print(f"Error: Missing required columns in tagging_results: {missing_cols}")
        return results_with_movement
    
    # Calculate max distance for each portfolio
    total_rows = len(results_with_movement)
    processed = 0
    successful_calculations = 0
    
    for index, row in results_with_movement.iterrows():
        tagged_to_au = row['TAGGED_TO_AU']
        new_au = row['NEW_AU']
        
        # Try the improved calculation method
        max_distance = calculate_max_customer_distance_final(
            tagged_to_au, 
            new_au,
            customer_au_assignments, 
            branch_df, 
            client_groups_df
        )
        
        # Update the distance column
        results_with_movement.at[index, 'MAX_CUSTOMER_DISTANCE_MILES'] = max_distance
        
        # Determine movement requirement
        if max_distance is not None:
            successful_calculations += 1
            if max_distance > 20:
                results_with_movement.at[index, 'MOVEMENT'] = 'MOVEMENT REQUIRED'
            else:
                results_with_movement.at[index, 'MOVEMENT'] = 'NO MOVEMENT REQUIRED'
        else:
            # Keep default 'NO MOVEMENT REQUIRED' when distance can't be calculated
            results_with_movement.at[index, 'MOVEMENT'] = 'NO MOVEMENT REQUIRED'
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{total_rows} portfolios... ({successful_calculations} successful)")
    
    # Print summary
    movement_required = results_with_movement[results_with_movement['MOVEMENT'] == 'MOVEMENT REQUIRED']
    no_movement_required = results_with_movement[results_with_movement['MOVEMENT'] == 'NO MOVEMENT REQUIRED']
    
    print(f"\n=== MOVEMENT ANALYSIS SUMMARY ===")
    print(f"Total portfolios analyzed: {len(results_with_movement)}")
    print(f"Successful distance calculations: {successful_calculations}")
    print(f"Portfolios requiring movement: {len(movement_required)}")
    print(f"Portfolios with no movement required: {len(no_movement_required)}")
    
    # Show distance statistics for successful calculations only
    valid_distances = results_with_movement['MAX_CUSTOMER_DISTANCE_MILES'].dropna()
    if len(valid_distances) > 0:
        print(f"\nDistance statistics (successful calculations only):")
        print(f"  Average max customer distance: {valid_distances.mean():.2f} miles")
        print(f"  Maximum customer distance: {valid_distances.max():.2f} miles")
        print(f"  Minimum customer distance: {valid_distances.min():.2f} miles")
        print(f"  Portfolios with distance > 20 miles: {len(valid_distances[valid_distances > 20])}")
    else:
        print("No valid distances calculated")
    
    return results_with_movement

# Usage example:
# tagging_results_with_movement = add_movement_analysis_improved(
#     tagging_results, 
#     customer_au_assignments, 
#     branch_df, 
#     client_groups_df
# )
