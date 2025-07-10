import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in miles"""
    try:
        R = 3959  # Earth's radius in miles
        
        # Handle NaN values and convert to float
        lat1 = float(lat1) if pd.notna(lat1) else 0.0
        lon1 = float(lon1) if pd.notna(lon1) else 0.0
        lat2 = float(lat2) if pd.notna(lat2) else 0.0
        lon2 = float(lon2) if pd.notna(lon2) else 0.0
        
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
        return float('inf')

def calculate_max_customer_distance(new_au, customer_au_assignments, branch_df, client_groups_df):
    """
    Calculate maximum distance from NEW_AU to its assigned customers
    
    Parameters:
    new_au: The AU code for the new portfolio
    customer_au_assignments: DataFrame with customer AU assignments
    branch_df: DataFrame with branch coordinates
    client_groups_df: DataFrame with customer coordinates
    
    Returns:
    float: Maximum distance in miles, or None if cannot be calculated
    """
    if pd.isna(new_au):
        return None
    
    # Get AU coordinates
    au_info = branch_df[branch_df['BRANCH_AU'] == new_au]
    if len(au_info) == 0:
        return None
    
    au_lat = au_info.iloc[0]['BRANCH_LAT_NUM'] if 'BRANCH_LAT_NUM' in au_info.columns else None
    au_lon = au_info.iloc[0]['BRANCH_LON_NUM'] if 'BRANCH_LON_NUM' in au_info.columns else None
    
    if pd.isna(au_lat) or pd.isna(au_lon):
        return None
    
    # Get customers assigned to this AU
    customers_for_au = customer_au_assignments[
        customer_au_assignments['ASSIGNED_AU'] == new_au
    ]['CG_ECN'].dropna().tolist()
    
    if not customers_for_au:
        return None
    
    # Get customer coordinates
    customer_data = client_groups_df[client_groups_df['CG_ECN'].isin(customers_for_au)]
    
    if len(customer_data) == 0:
        return None
    
    # Calculate distances for each customer
    distances = []
    for _, customer in customer_data.iterrows():
        cust_lat = customer['CG_LAT_NUM'] if 'CG_LAT_NUM' in customer_data.columns else None
        cust_lon = customer['CG_LON_NUM'] if 'CG_LON_NUM' in customer_data.columns else None
        
        if pd.notna(cust_lat) and pd.notna(cust_lon):
            distance = haversine_distance(au_lat, au_lon, cust_lat, cust_lon)
            if not np.isinf(distance) and distance >= 0:
                distances.append(distance)
    
    # Return maximum distance
    if distances:
        return max(distances)
    else:
        return None

def add_movement_analysis(tagging_results, customer_au_assignments, branch_df, client_groups_df):
    """
    Add movement analysis to tagging results based on maximum customer distance
    
    Parameters:
    tagging_results: DataFrame with tagging results
    customer_au_assignments: DataFrame with customer AU assignments
    branch_df: DataFrame with branch coordinates
    client_groups_df: DataFrame with customer coordinates
    
    Returns:
    DataFrame: tagging_results with additional columns:
        - MAX_CUSTOMER_DISTANCE_MILES: Maximum distance from AU to customers
        - MOVEMENT: 'MOVEMENT REQUIRED' if distance > 20, else 'NO MOVEMENT REQUIRED'
    """
    print("=== ADDING MOVEMENT ANALYSIS ===")
    
    # Create a copy to avoid modifying the original
    results_with_movement = tagging_results.copy()
    
    # Initialize new columns
    results_with_movement['MAX_CUSTOMER_DISTANCE_MILES'] = None
    results_with_movement['MOVEMENT'] = None
    
    # Calculate max distance for each portfolio
    for index, row in results_with_movement.iterrows():
        new_au = row['NEW_AU']
        
        # Calculate maximum customer distance
        max_distance = calculate_max_customer_distance(
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
        else:
            results_with_movement.at[index, 'MOVEMENT'] = 'NO MOVEMENT REQUIRED'
    
    # Print summary
    movement_required = results_with_movement[results_with_movement['MOVEMENT'] == 'MOVEMENT REQUIRED']
    no_movement_required = results_with_movement[results_with_movement['MOVEMENT'] == 'NO MOVEMENT REQUIRED']
    
    print(f"Movement analysis complete:")
    print(f"  Portfolios requiring movement: {len(movement_required)}")
    print(f"  Portfolios with no movement required: {len(no_movement_required)}")
    
    # Show distance statistics
    valid_distances = results_with_movement['MAX_CUSTOMER_DISTANCE_MILES'].dropna()
    if len(valid_distances) > 0:
        print(f"  Average max customer distance: {valid_distances.mean():.2f} miles")
        print(f"  Maximum customer distance: {valid_distances.max():.2f} miles")
        print(f"  Minimum customer distance: {valid_distances.min():.2f} miles")
    
    return results_with_movement

# Usage:
# tagging_results_with_movement = add_movement_analysis(
#     tagging_results, 
#     customer_au_assignments, 
#     branch_df, 
#     client_groups_df
# )
# 
# tagging_results_with_movement.to_csv('portfolio_tagging_with_movement.csv', index=False)
