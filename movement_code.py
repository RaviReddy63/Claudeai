import pandas as pd
import numpy as np

def distance_miles(lat1, lon1, lat2, lon2):
    """Simple distance calculation in miles"""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return 3959 * c  # Earth radius in miles

def add_movement_analysis(tagging_results, customer_au_assignments, branch_df):
    """Simple movement analysis"""
    
    results = tagging_results.copy()
    results['MAX_CUSTOMER_DISTANCE_MILES'] = 0
    results['MOVEMENT'] = 'NO MOVEMENT REQUIRED'
    
    for i, row in results.iterrows():
        tagged_to_au = row['TAGGED_TO_AU']
        new_au = row['NEW_AU']
        
        # If tagged_to_au is null, set movement to null
        if pd.isna(tagged_to_au):
            results.at[i, 'MOVEMENT'] = None
            continue
        
        # Get branch coordinates
        branch = branch_df[branch_df['BRANCH_AU'] == tagged_to_au]
        if len(branch) == 0:
            continue
            
        branch_lat = branch.iloc[0]['BRANCH_LAT_NUM']
        branch_lon = branch.iloc[0]['BRANCH_LON_NUM']
        
        # Get customers for this AU with their coordinates
        customers = customer_au_assignments[customer_au_assignments['ASSIGNED_AU'] == new_au]
        
        # Calculate distances to all customers
        max_distance = 0
        for _, customer_row in customers.iterrows():
            cust_lat = customer_row['LAT_NUM']
            cust_lon = customer_row['LON_NUM']
            
            distance = distance_miles(branch_lat, branch_lon, cust_lat, cust_lon)
            if distance > max_distance:
                max_distance = distance
        
        # Update results
        results.at[i, 'MAX_CUSTOMER_DISTANCE_MILES'] = max_distance
        if max_distance > 20:
            results.at[i, 'MOVEMENT'] = 'MOVEMENT REQUIRED'
    
    return results

# Usage:
# results_with_movement = add_movement_analysis(tagging_results, customer_au_assignments, branch_df)
