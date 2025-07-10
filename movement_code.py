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

def add_movement_analysis(tagging_results, customer_au_assignments, branch_df, CLIENT_GROUPS_DF_NEW):
    """Simple movement analysis with average distance calculations"""
    
    results = tagging_results.copy()
    results['MAX_CUSTOMER_DISTANCE_MILES'] = 0
    results['MOVEMENT'] = 'NO MOVEMENT REQUIRED'
    results['NEW_AVG_DIST_FROM_CUSTS'] = 0
    results['CURRENT_AVG_DIST_FROM_CUSTS'] = 0
    
    for i, row in results.iterrows():
        tagged_to_au = row['TAGGED_TO_AU']
        new_au = row['NEW_AU']
        
        # If tagged_to_au is null, set movement to null
        if pd.isna(tagged_to_au):
            results.at[i, 'MOVEMENT'] = None
            continue
        
        # Get NEW_AU coordinates from branch_df
        new_au_branch = branch_df[branch_df['BRANCH_AU'] == new_au]
        if len(new_au_branch) == 0:
            continue
        new_au_lat = new_au_branch.iloc[0]['BRANCH_LAT_NUM']
        new_au_lon = new_au_branch.iloc[0]['BRANCH_LON_NUM']
        
        # Get TAGGED_TO_AU coordinates from branch_df
        tagged_au_branch = branch_df[branch_df['BRANCH_AU'] == tagged_to_au]
        if len(tagged_au_branch) == 0:
            continue
        tagged_au_lat = tagged_au_branch.iloc[0]['BRANCH_LAT_NUM']
        tagged_au_lon = tagged_au_branch.iloc[0]['BRANCH_LON_NUM']
        
        # Get customers for NEW_AU from customer_au_assignments
        new_au_customers = customer_au_assignments[customer_au_assignments['ASSIGNED_AU'] == new_au]
        
        # Calculate distances from NEW_AU to its customers
        max_distance = 0
        new_au_distances = []
        for _, customer_row in new_au_customers.iterrows():
            cust_lat = customer_row['LAT_NUM']
            cust_lon = customer_row['LON_NUM']
            
            distance = distance_miles(new_au_lat, new_au_lon, cust_lat, cust_lon)
            new_au_distances.append(distance)
            if distance > max_distance:
                max_distance = distance
        
        # Calculate average distance for NEW_AU
        if new_au_distances:
            new_avg_distance = np.mean(new_au_distances)
            results.at[i, 'NEW_AVG_DIST_FROM_CUSTS'] = new_avg_distance
        
        # Get customers for TAGGED_TO_AU from CLIENT_GROUPS_DF_NEW using portfolio code
        tagged_portfolio = row['TAGGED_TO_PORTFOLIO']
        
        # Calculate distances from TAGGED_TO_AU to its customers
        current_au_distances = []
        if not pd.isna(tagged_portfolio):
            tagged_au_customers = CLIENT_GROUPS_DF_NEW[CLIENT_GROUPS_DF_NEW['CG_PORTFOLIO_CD'] == tagged_portfolio]
            
            for _, customer_row in tagged_au_customers.iterrows():
                cust_lat = customer_row['LAT_NUM']
                cust_lon = customer_row['LON_NUM']
                
                distance = distance_miles(tagged_au_lat, tagged_au_lon, cust_lat, cust_lon)
                current_au_distances.append(distance)
        
        # Calculate average distance for TAGGED_TO_AU
        if current_au_distances:
            current_avg_distance = np.mean(current_au_distances)
            results.at[i, 'CURRENT_AVG_DIST_FROM_CUSTS'] = current_avg_distance
        
        # Update results
        results.at[i, 'MAX_CUSTOMER_DISTANCE_MILES'] = max_distance
        if max_distance > 20:
            results.at[i, 'MOVEMENT'] = 'MOVEMENT REQUIRED'
    
    return results

# Usage:
# results_with_movement = add_movement_analysis(tagging_results, customer_au_assignments, branch_df, CLIENT_GROUPS_DF_NEW)
