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

def calculate_max_customer_distance_corrected(tagged_to_au, new_au, customer_au_assignments, branch_df, client_groups_df):
    """
    Calculate maximum distance from TAGGED_TO_AU to customers of NEW_AU
    
    Parameters:
    tagged_to_au: The AU code where portfolio is being moved TO
    new_au: The AU code of the current portfolio (whose customers we're measuring)
    customer_au_assignments: DataFrame with customer AU assignments
    branch_df: DataFrame with branch coordinates
    client_groups_df: DataFrame with customer coordinates
    
    Returns:
    float: Maximum distance in miles, or None if cannot be calculated
    """
    if pd.isna(tagged_to_au) or pd.isna(new_au):
        return None
    
    # Step 1: Get coordinates of TAGGED_TO_AU (where portfolio is moving TO)
    tagged_au_info = branch_df[branch_df['BRANCH_AU'] == tagged_to_au]
    if len(tagged_au_info) == 0:
        print(f"Warning: TAGGED_TO_AU '{tagged_to_au}' not found in branch_df")
        return None
    
    tagged_au_lat = tagged_au_info.iloc[0]['BRANCH_LAT_NUM'] if 'BRANCH_LAT_NUM' in tagged_au_info.columns else None
    tagged_au_lon = tagged_au_info.iloc[0]['BRANCH_LON_NUM'] if 'BRANCH_LON_NUM' in tagged_au_info.columns else None
    
    if pd.isna(tagged_au_lat) or pd.isna(tagged_au_lon):
        print(f"Warning: Coordinates not found for TAGGED_TO_AU '{tagged_to_au}'")
        return None
    
    # Step 2: Get customers assigned to NEW_AU (the original portfolio)
    customers_for_new_au = customer_au_assignments[
        customer_au_assignments['ASSIGNED_AU'] == new_au
    ]['CG_ECN'].dropna().tolist()
    
    if not customers_for_new_au:
        print(f"Warning: No customers found for NEW_AU '{new_au}'")
        return None
    
    # Step 3: Get customer coordinates
    customer_data = client_groups_df[client_groups_df['CG_ECN'].isin(customers_for_new_au)]
    
    if len(customer_data) == 0:
        print(f"Warning: Customer coordinate data not found for NEW_AU '{new_au}'")
        return None
    
    # Step 4: Calculate distances from TAGGED_TO_AU to each customer of NEW_AU
    distances = []
    for _, customer in customer_data.iterrows():
        cust_lat = customer['CG_LAT_NUM'] if 'CG_LAT_NUM' in customer_data.columns else None
        cust_lon = customer['CG_LON_NUM'] if 'CG_LON_NUM' in customer_data.columns else None
        
        if pd.notna(cust_lat) and pd.notna(cust_lon):
            distance = haversine_distance(tagged_au_lat, tagged_au_lon, cust_lat, cust_lon)
            if distance is not None and distance >= 0:
                distances.append(distance)
    
    # Step 5: Return maximum distance
    if distances:
        return max(distances)
    else:
        print(f"Warning: No valid distances calculated for NEW_AU '{new_au}' to TAGGED_TO_AU '{tagged_to_au}'")
        return None

def add_movement_analysis_corrected(tagging_results, customer_au_assignments, branch_df, client_groups_df):
    """
    Add movement analysis to tagging results based on distance from TAGGED_TO_AU to customers of NEW_AU
    
    Parameters:
    tagging_results: DataFrame with tagging results (must have 'TAGGED_TO_AU' and 'NEW_AU' columns)
    customer_au_assignments: DataFrame with customer AU assignments
    branch_df: DataFrame with branch coordinates
    client_groups_df: DataFrame with customer coordinates
    
    Returns:
    DataFrame: tagging_results with additional columns:
        - MAX_CUSTOMER_DISTANCE_MILES: Maximum distance from TAGGED_TO_AU to customers of NEW_AU
        - MOVEMENT: 'MOVEMENT REQUIRED' if distance > 20, else 'NO MOVEMENT REQUIRED'
    """
    print("=== ADDING CORRECTED MOVEMENT ANALYSIS ===")
    print("Logic: Distance from TAGGED_TO_AU to customers of NEW_AU")
    
    # Create a copy to avoid modifying the original
    results_with_movement = tagging_results.copy()
    
    # Initialize new columns
    results_with_movement['MAX_CUSTOMER_DISTANCE_MILES'] = None
    results_with_movement['MOVEMENT'] = None
    
    # Validate required columns
    required_cols = ['TAGGED_TO_AU', 'NEW_AU']
    missing_cols = [col for col in required_cols if col not in results_with_movement.columns]
    if missing_cols:
        print(f"Error: Missing required columns in tagging_results: {missing_cols}")
        return results_with_movement
    
    # Calculate max distance for each portfolio
    for index, row in results_with_movement.iterrows():
        tagged_to_au = row['TAGGED_TO_AU']
        new_au = row['NEW_AU']
        
        print(f"Processing row {index}: NEW_AU='{new_au}' -> TAGGED_TO_AU='{tagged_to_au}'")
        
        # Calculate maximum customer distance
        max_distance = calculate_max_customer_distance_corrected(
            tagged_to_au, 
            new_au,
            customer_au_assignments, 
            branch_df, 
            client_groups_df
        )
        
        # Update the distance column
        results_with_movement.at[index, 'MAX_CUSTOMER_DISTANCE_MILES'] = max_distance
        
        # Determine movement requirement
        if max_distance is not None and max_distance > 20:
            results_with_movement.at[index, 'MOVEMENT'] = 'MOVEMENT REQUIRED'
            print(f"  -> Max distance: {max_distance:.2f} miles - MOVEMENT REQUIRED")
        elif max_distance is not None:
            results_with_movement.at[index, 'MOVEMENT'] = 'NO MOVEMENT REQUIRED'
            print(f"  -> Max distance: {max_distance:.2f} miles - NO MOVEMENT REQUIRED")
        else:
            results_with_movement.at[index, 'MOVEMENT'] = 'NO MOVEMENT REQUIRED'
            print(f"  -> Distance could not be calculated - NO MOVEMENT REQUIRED")
    
    # Print summary
    movement_required = results_with_movement[results_with_movement['MOVEMENT'] == 'MOVEMENT REQUIRED']
    no_movement_required = results_with_movement[results_with_movement['MOVEMENT'] == 'NO MOVEMENT REQUIRED']
    
    print(f"\n=== MOVEMENT ANALYSIS SUMMARY ===")
    print(f"Total portfolios analyzed: {len(results_with_movement)}")
    print(f"Portfolios requiring movement: {len(movement_required)}")
    print(f"Portfolios with no movement required: {len(no_movement_required)}")
    
    # Show distance statistics
    valid_distances = results_with_movement['MAX_CUSTOMER_DISTANCE_MILES'].dropna()
    if len(valid_distances) > 0:
        print(f"Distance statistics:")
        print(f"  Average max customer distance: {valid_distances.mean():.2f} miles")
        print(f"  Maximum customer distance: {valid_distances.max():.2f} miles")
        print(f"  Minimum customer distance: {valid_distances.min():.2f} miles")
        print(f"  Portfolios with distance > 20 miles: {len(valid_distances[valid_distances > 20])}")
    else:
        print("No valid distances calculated")
    
    return results_with_movement

# Usage Example:
# tagging_results_with_movement = add_movement_analysis_corrected(
#     tagging_results, 
#     customer_au_assignments, 
#     branch_df, 
#     client_groups_df
# )
# 
# # Save results
# tagging_results_with_movement.to_csv('portfolio_tagging_with_corrected_movement.csv', index=False)
# 
# # Display results for portfolios requiring movement
# movement_required = tagging_results_with_movement[
#     tagging_results_with_movement['MOVEMENT'] == 'MOVEMENT REQUIRED'
# ]
# print("\nPortfolios requiring movement:")
# print(movement_required[['NEW_AU', 'TAGGED_TO_AU', 'MAX_CUSTOMER_DISTANCE_MILES', 'MOVEMENT']])
