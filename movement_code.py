import pandas as pd
import numpy as np

def debug_movement_analysis(tagging_results, customer_au_assignments, branch_df, client_groups_df):
    """
    Debug version to identify exactly where the issue is
    """
    print("=== DEBUGGING MOVEMENT ANALYSIS ===")
    
    # Check DataFrame structures first
    print("\n1. DATAFRAME STRUCTURES:")
    print(f"tagging_results columns: {tagging_results.columns.tolist()}")
    print(f"customer_au_assignments columns: {customer_au_assignments.columns.tolist()}")
    print(f"branch_df columns: {branch_df.columns.tolist()}")
    print(f"client_groups_df columns: {client_groups_df.columns.tolist()}")
    
    # Take first row for detailed debugging
    first_row = tagging_results.iloc[0]
    tagged_to_au = first_row['TAGGED_TO_AU'] if 'TAGGED_TO_AU' in first_row else None
    new_au = first_row['NEW_AU'] if 'NEW_AU' in first_row else None
    
    print(f"\n2. TESTING WITH FIRST ROW:")
    print(f"NEW_AU: {new_au}")
    print(f"TAGGED_TO_AU: {tagged_to_au}")
    
    if pd.isna(new_au) or pd.isna(tagged_to_au):
        print("ERROR: NEW_AU or TAGGED_TO_AU is NaN!")
        return
    
    # Step 1: Check TAGGED_TO_AU coordinates
    print(f"\n3. CHECKING TAGGED_TO_AU COORDINATES:")
    tagged_au_info = branch_df[branch_df['BRANCH_AU'] == tagged_to_au]
    print(f"Rows found for TAGGED_TO_AU '{tagged_to_au}': {len(tagged_au_info)}")
    
    if len(tagged_au_info) == 0:
        print("ERROR: TAGGED_TO_AU not found in branch_df!")
        print(f"Available BRANCH_AU values: {branch_df['BRANCH_AU'].unique()[:10]}")
        return
    
    # Check coordinate columns in branch_df
    lat_cols = [col for col in branch_df.columns if 'LAT' in col.upper()]
    lon_cols = [col for col in branch_df.columns if 'LON' in col.upper()]
    print(f"Latitude columns in branch_df: {lat_cols}")
    print(f"Longitude columns in branch_df: {lon_cols}")
    
    # Try to get coordinates with available column names
    if lat_cols and lon_cols:
        lat_col = lat_cols[0]  # Use first available lat column
        lon_col = lon_cols[0]  # Use first available lon column
        tagged_au_lat = tagged_au_info.iloc[0][lat_col]
        tagged_au_lon = tagged_au_info.iloc[0][lon_col]
        print(f"TAGGED_TO_AU coordinates ({lat_col}, {lon_col}): ({tagged_au_lat}, {tagged_au_lon})")
    else:
        print("ERROR: No latitude/longitude columns found in branch_df!")
        return
    
    # Step 2: Check customers for NEW_AU
    print(f"\n4. CHECKING CUSTOMERS FOR NEW_AU:")
    customers_for_new_au = customer_au_assignments[
        customer_au_assignments['ASSIGNED_AU'] == new_au
    ]['CG_ECN'].dropna().tolist()
    print(f"Customers found for NEW_AU '{new_au}': {len(customers_for_new_au)}")
    
    if not customers_for_new_au:
        print("ERROR: No customers found for NEW_AU!")
        print(f"Available ASSIGNED_AU values: {customer_au_assignments['ASSIGNED_AU'].unique()[:10]}")
        return
    
    print(f"Sample customers: {customers_for_new_au[:5]}")
    
    # Step 3: Check customer coordinates
    print(f"\n5. CHECKING CUSTOMER COORDINATES:")
    customer_data = client_groups_df[client_groups_df['CG_ECN'].isin(customers_for_new_au)]
    print(f"Customer records with coordinates: {len(customer_data)}")
    
    if len(customer_data) == 0:
        print("ERROR: No customer coordinate data found!")
        print(f"Sample CG_ECN from assignments: {customers_for_new_au[:5]}")
        print(f"Sample CG_ECN from coordinates: {client_groups_df['CG_ECN'].dropna().head(5).tolist()}")
        return
    
    # Check coordinate columns in client_groups_df
    cust_lat_cols = [col for col in client_groups_df.columns if 'LAT' in col.upper()]
    cust_lon_cols = [col for col in client_groups_df.columns if 'LON' in col.upper()]
    print(f"Customer latitude columns: {cust_lat_cols}")
    print(f"Customer longitude columns: {cust_lon_cols}")
    
    if not cust_lat_cols or not cust_lon_cols:
        print("ERROR: No latitude/longitude columns found in client_groups_df!")
        return
    
    # Step 4: Check valid coordinates
    cust_lat_col = cust_lat_cols[0]
    cust_lon_col = cust_lon_cols[0]
    
    valid_coords = customer_data[
        customer_data[cust_lat_col].notna() & customer_data[cust_lon_col].notna()
    ]
    print(f"Customers with valid coordinates: {len(valid_coords)}")
    
    if len(valid_coords) == 0:
        print("ERROR: No customers have valid coordinates!")
        print("Sample coordinate data:")
        print(customer_data[['CG_ECN', cust_lat_col, cust_lon_col]].head())
        return
    
    print("Sample valid coordinates:")
    print(valid_coords[['CG_ECN', cust_lat_col, cust_lon_col]].head(3))
    
    # Step 5: Test distance calculation
    print(f"\n6. TESTING DISTANCE CALCULATION:")
    sample_customer = valid_coords.iloc[0]
    cust_lat = sample_customer[cust_lat_col]
    cust_lon = sample_customer[cust_lon_col]
    
    print(f"Sample calculation:")
    print(f"  TAGGED_TO_AU coords: ({tagged_au_lat}, {tagged_au_lon})")
    print(f"  Customer coords: ({cust_lat}, {cust_lon})")
    
    # Simple distance calculation test
    try:
        R = 3959  # Earth's radius in miles
        lat1, lon1, lat2, lon2 = map(np.radians, [tagged_au_lat, tagged_au_lon, cust_lat, cust_lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        print(f"  Calculated distance: {distance:.2f} miles")
    except Exception as e:
        print(f"  Distance calculation error: {e}")
    
    print("\n=== DEBUG COMPLETE ===")
    print("If you see this message, the issue should be identified above.")

# Run the debug function
debug_movement_analysis(tagging_results, customer_au_assignments, branch_df, client_groups_df)
