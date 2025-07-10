import pandas as pd
import numpy as np

def get_portfolio_financial_metrics(portfolio_customers, client_groups_df):
    """Get average financial metrics for a portfolio using customer list"""
    if not portfolio_customers:
        return None, None, None
    
    if isinstance(portfolio_customers, set):
        customer_list = list(portfolio_customers)
    else:
        customer_list = portfolio_customers
    
    # Ensure customer_list contains valid values
    customer_list = [x for x in customer_list if pd.notna(x)]
    if not customer_list:
        return None, None, None
    
    portfolio_data = client_groups_df[client_groups_df['CG_ECN'].isin(customer_list)]
    
    if len(portfolio_data) == 0:
        return None, None, None
    
    # Convert to numeric and handle errors
    avg_deposit = None
    if 'DEPOSIT_BAL' in portfolio_data.columns:
        deposit_numeric = pd.to_numeric(portfolio_data['DEPOSIT_BAL'], errors='coerce')
        avg_deposit = deposit_numeric.mean() if not deposit_numeric.isna().all() else None
    
    avg_gross_sales = None
    if 'CG_GROSS_SALES' in portfolio_data.columns:
        sales_numeric = pd.to_numeric(portfolio_data['CG_GROSS_SALES'], errors='coerce')
        avg_gross_sales = sales_numeric.mean() if not sales_numeric.isna().all() else None
    
    avg_bank_revenue = None
    if 'BANK_REVENUE' in portfolio_data.columns:
        revenue_numeric = pd.to_numeric(portfolio_data['BANK_REVENUE'], errors='coerce')
        avg_bank_revenue = revenue_numeric.mean() if not revenue_numeric.isna().all() else None
    
    return avg_deposit, avg_gross_sales, avg_bank_revenue

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    try:
        R = 3959
        # Handle NaN values and convert to float
        lat1 = float(lat1) if pd.notna(lat1) else 0.0
        lon1 = float(lon1) if pd.notna(lon1) else 0.0
        lat2 = float(lat2) if pd.notna(lat2) else 0.0
        lon2 = float(lon2) if pd.notna(lon2) else 0.0
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    except (ValueError, TypeError) as e:
        print(f"Error in distance calculation: {e}")
        return float('inf')

def safe_get_value(row, column, default=''):
    """Safely get value from pandas row, handling NaN"""
    try:
        value = row[column]
        return value if pd.notna(value) else default
    except (KeyError, IndexError):
        return default

def calculate_avg_au_customer_distance(au_code, customer_list, branch_df, client_groups_df):
    """Calculate average distance between AU and its customers"""
    if not customer_list or pd.isna(au_code):
        return None
    
    # Get AU coordinates
    au_info = branch_df[branch_df['BRANCH_AU'] == au_code]
    if len(au_info) == 0:
        return None
    
    au_lat = safe_get_value(au_info.iloc[0], 'BRANCH_LAT_NUM', None)
    au_lon = safe_get_value(au_info.iloc[0], 'BRANCH_LON_NUM', None)
    
    if pd.isna(au_lat) or pd.isna(au_lon):
        return None
    
    # Filter customer list to remove NaN values
    valid_customers = [x for x in customer_list if pd.notna(x)]
    if not valid_customers:
        return None
    
    # Get customer coordinates
    customer_data = client_groups_df[client_groups_df['CG_ECN'].isin(valid_customers)]
    
    if len(customer_data) == 0:
        return None
    
    # Calculate distances for each customer
    distances = []
    for _, customer in customer_data.iterrows():
        cust_lat = safe_get_value(customer, 'CG_LAT_NUM', None)
        cust_lon = safe_get_value(customer, 'CG_LON_NUM', None)
        
        if pd.notna(cust_lat) and pd.notna(cust_lon):
            distance = haversine_distance_vectorized(au_lat, au_lon, cust_lat, cust_lon)
            if not np.isinf(distance) and distance >= 0:
                distances.append(distance)
    
    # Return average distance
    if distances:
        return np.mean(distances)
    else:
        return None

def calculate_avg_portfolio_customer_distance(portfolio_code, client_groups_df, active_portfolio_df, branch_df):
    """Calculate average distance between portfolio's AU and its customers"""
    if pd.isna(portfolio_code):
        return None
    
    # Get portfolio AU
    portfolio_info = active_portfolio_df[active_portfolio_df['PORT_CODE'] == portfolio_code]
    if len(portfolio_info) == 0:
        return None
    
    portfolio_au = safe_get_value(portfolio_info.iloc[0], 'AU', None)
    if pd.isna(portfolio_au):
        return None
    
    # Get portfolio customers
    client_groups_clean = client_groups_df.copy()
    client_groups_clean['ISPORTFOLIOACTIVE'] = pd.to_numeric(client_groups_clean['ISPORTFOLIOACTIVE'], errors='coerce').fillna(0)
    
    portfolio_customers = client_groups_clean[
        (client_groups_clean['CG_PORTFOLIO_CD'] == portfolio_code) &
        (client_groups_clean['ISPORTFOLIOACTIVE'] == 1)
    ]['CG_ECN'].dropna().tolist()
    
    if not portfolio_customers:
        return None
    
    # Calculate average distance
    return calculate_avg_au_customer_distance(portfolio_au, portfolio_customers, branch_df, client_groups_df)

def get_existing_portfolios_mojgan(active_portfolio_df, client_groups_df, branch_df):
    """Get existing portfolios under MOJGAN MADADI"""
    # Clean the dataframes first
    active_portfolio_clean = active_portfolio_df.copy()
    active_portfolio_clean['DIRECTOR_NAME'] = active_portfolio_clean['DIRECTOR_NAME'].fillna('')
    active_portfolio_clean['ROLE_TYPE'] = active_portfolio_clean['ROLE_TYPE'].fillna('')
    active_portfolio_clean['ISACTIVE'] = pd.to_numeric(active_portfolio_clean['ISACTIVE'], errors='coerce').fillna(0)
    
    # IN MARKET portfolios
    existing_inmarket = active_portfolio_clean[
        (active_portfolio_clean['ISACTIVE'] == 1) & 
        (active_portfolio_clean['ROLE_TYPE'] == 'IN MARKET') &
        (active_portfolio_clean['DIRECTOR_NAME'] == 'MOJGAN MADADI')
    ].copy()
    
    if len(existing_inmarket) > 0:
        existing_inmarket = existing_inmarket.merge(
            branch_df[['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
            left_on='AU', right_on='BRANCH_AU', how='left'
        )
    
    # CENTRALIZED portfolios
    existing_centralized = active_portfolio_clean[
        (active_portfolio_clean['ISACTIVE'] == 1) & 
        (active_portfolio_clean['ROLE_TYPE'] == 'CENTRALIZED') &
        (active_portfolio_clean['DIRECTOR_NAME'] == 'MOJGAN MADADI')
    ].copy()
    
    # Get customer lists for ALL portfolios
    portfolio_customers = {}
    
    # Clean client_groups_df
    client_groups_clean = client_groups_df.copy()
    client_groups_clean['ISPORTFOLIOACTIVE'] = pd.to_numeric(client_groups_clean['ISPORTFOLIOACTIVE'], errors='coerce').fillna(0)
    
    # Add IN MARKET portfolio customers
    for _, portfolio in existing_inmarket.iterrows():
        port_code = portfolio['PORT_CODE']
        if pd.notna(port_code):
            portfolio_customer_list = client_groups_clean[
                (client_groups_clean['CG_PORTFOLIO_CD'] == port_code) &
                (client_groups_clean['ISPORTFOLIOACTIVE'] == 1)
            ]['CG_ECN'].tolist()
            # Filter out NaN values
            portfolio_customer_list = [x for x in portfolio_customer_list if pd.notna(x)]
            portfolio_customers[port_code] = set(portfolio_customer_list)
    
    # Add CENTRALIZED portfolio customers
    for _, portfolio in existing_centralized.iterrows():
        port_code = portfolio['PORT_CODE']
        if pd.notna(port_code):
            portfolio_customer_list = client_groups_clean[
                (client_groups_clean['CG_PORTFOLIO_CD'] == port_code) &
                (client_groups_clean['ISPORTFOLIOACTIVE'] == 1)
            ]['CG_ECN'].tolist()
            # Filter out NaN values
            portfolio_customer_list = [x for x in portfolio_customer_list if pd.notna(x)]
            portfolio_customers[port_code] = set(portfolio_customer_list)
    
    return existing_inmarket, existing_centralized, portfolio_customers

def get_all_existing_portfolios(active_portfolio_df, branch_df):
    """Get ALL existing portfolios (not just MOJGAN MADADI) for distance-based manager/director assignment"""
    # Clean the dataframes first
    active_portfolio_clean = active_portfolio_df.copy()
    active_portfolio_clean['DIRECTOR_NAME'] = active_portfolio_clean['DIRECTOR_NAME'].fillna('')
    active_portfolio_clean['MANAGER_NAME'] = active_portfolio_clean['MANAGER_NAME'].fillna('')
    active_portfolio_clean['ROLE_TYPE'] = active_portfolio_clean['ROLE_TYPE'].fillna('')
    active_portfolio_clean['ISACTIVE'] = pd.to_numeric(active_portfolio_clean['ISACTIVE'], errors='coerce').fillna(0)
    
    # Get all active portfolios
    all_active_portfolios = active_portfolio_clean[
        active_portfolio_clean['ISACTIVE'] == 1
    ].copy()
    
    if len(all_active_portfolios) > 0:
        all_active_portfolios = all_active_portfolios.merge(
            branch_df[['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
            left_on='AU', right_on='BRANCH_AU', how='left'
        )
    
    return all_active_portfolios

def calculate_customer_overlap(new_customers, existing_customers):
    """Calculate customer overlap between portfolios"""
    if not existing_customers or not new_customers:
        return 0
    
    # Ensure both are sets and filter out NaN values
    new_customer_set = set([x for x in new_customers if pd.notna(x)])
    existing_customer_set = set([x for x in existing_customers if pd.notna(x)])
    
    overlap = len(new_customer_set.intersection(existing_customer_set))
    return overlap

def get_portfolio_financial_metrics_by_portfolio_code(portfolio_code, client_groups_df):
    """Get average financial metrics for a portfolio using portfolio code"""
    if pd.isna(portfolio_code):
        return None, None, None, 0
    
    # Clean client_groups_df
    client_groups_clean = client_groups_df.copy()
    client_groups_clean['ISPORTFOLIOACTIVE'] = pd.to_numeric(client_groups_clean['ISPORTFOLIOACTIVE'], errors='coerce').fillna(0)
    
    portfolio_data = client_groups_clean[
        (client_groups_clean['CG_PORTFOLIO_CD'] == portfolio_code) &
        (client_groups_clean['ISPORTFOLIOACTIVE'] == 1)
    ]
    
    if len(portfolio_data) == 0:
        return None, None, None, 0
    
    # Convert to numeric and handle errors
    avg_deposit = None
    if 'DEPOSIT_BAL' in portfolio_data.columns:
        deposit_numeric = pd.to_numeric(portfolio_data['DEPOSIT_BAL'], errors='coerce')
        avg_deposit = deposit_numeric.mean() if not deposit_numeric.isna().all() else None
    
    avg_gross_sales = None
    if 'CG_GROSS_SALES' in portfolio_data.columns:
        sales_numeric = pd.to_numeric(portfolio_data['CG_GROSS_SALES'], errors='coerce')
        avg_gross_sales = sales_numeric.mean() if not sales_numeric.isna().all() else None
    
    avg_bank_revenue = None
    if 'BANK_REVENUE' in portfolio_data.columns:
        revenue_numeric = pd.to_numeric(portfolio_data['BANK_REVENUE'], errors='coerce')
        avg_bank_revenue = revenue_numeric.mean() if not revenue_numeric.isna().all() else None
    
    portfolio_size = len(portfolio_data)
    
    return avg_deposit, avg_gross_sales, avg_bank_revenue, portfolio_size

def count_new_customers_not_in_active_portfolios(customer_au_assignments, client_groups_df):
    """Count customers in new portfolios that are not part of any active existing portfolio"""
    # Clean data
    clean_assignments = customer_au_assignments.dropna(subset=['CG_ECN'])
    new_portfolio_customers = set(clean_assignments['CG_ECN'].tolist())
    
    client_groups_clean = client_groups_df.copy()
    client_groups_clean['ISPORTFOLIOACTIVE'] = pd.to_numeric(client_groups_clean['ISPORTFOLIOACTIVE'], errors='coerce').fillna(0)
    
    active_portfolio_customers = set(client_groups_clean[
        client_groups_clean['ISPORTFOLIOACTIVE'] == 1
    ]['CG_ECN'].dropna().tolist())
    
    new_customers_not_in_active = new_portfolio_customers - active_portfolio_customers
    return len(new_customers_not_in_active)

def get_unique_new_portfolios(customer_au_assignments):
    """Get unique new portfolios with their customer counts"""
    # Clean data before grouping
    clean_assignments = customer_au_assignments.copy()
    clean_assignments = clean_assignments.dropna(subset=['ASSIGNED_AU', 'TYPE', 'CG_ECN'])
    
    # Standardize TYPE values
    clean_assignments['TYPE'] = clean_assignments['TYPE'].replace('INMARKET', 'IN MARKET')
    
    portfolio_summary = clean_assignments.groupby(['ASSIGNED_AU', 'TYPE']).agg({
        'CG_ECN': 'count'
    }).reset_index()
    portfolio_summary.columns = ['ASSIGNED_AU', 'TYPE', 'CUSTOMER_COUNT']
    
    return portfolio_summary

def find_nearest_portfolio_for_manager_director(new_au, branch_df, all_portfolios_df):
    """Find the nearest existing portfolio to assign manager/director for untagged portfolios"""
    # Get coordinates for the new AU
    new_au_info = branch_df[branch_df['BRANCH_AU'] == new_au]
    if len(new_au_info) == 0:
        return None, None, float('inf')
    
    new_lat = safe_get_value(new_au_info.iloc[0], 'BRANCH_LAT_NUM', 0)
    new_lon = safe_get_value(new_au_info.iloc[0], 'BRANCH_LON_NUM', 0)
    
    min_distance = float('inf')
    nearest_manager = None
    nearest_director = None
    
    # Check distances to all existing portfolios
    for _, existing_portfolio in all_portfolios_df.iterrows():
        existing_lat = safe_get_value(existing_portfolio, 'BRANCH_LAT_NUM', 0)
        existing_lon = safe_get_value(existing_portfolio, 'BRANCH_LON_NUM', 0)
        
        distance = haversine_distance_vectorized(new_lat, new_lon, existing_lat, existing_lon)
        
        if distance < min_distance and not np.isinf(distance):
            min_distance = distance
            nearest_manager = safe_get_value(existing_portfolio, 'MANAGER_NAME', '')
            nearest_director = safe_get_value(existing_portfolio, 'DIRECTOR_NAME', '')
    
    return nearest_manager, nearest_director, min_distance

def calculate_movement_status(new_au, new_type, tagged_to_au, tagging_criteria, customer_au_assignments, client_groups_df, branch_df):
    """Calculate movement status based on distance between existing AU and new portfolio customers"""
    
    # Only calculate for IN MARKET portfolios that are tagged to existing portfolios
    if (new_type != 'IN MARKET' or 
        tagged_to_au == '' or 
        tagged_to_au == 'N/A (CENTRALIZED)' or 
        tagging_criteria == 'NEAREST_MANAGER_DIRECTOR_ONLY'):
        return None
    
    # Get customers for the new portfolio
    new_customers = customer_au_assignments[
        (customer_au_assignments['ASSIGNED_AU'] == new_au) &
        (customer_au_assignments['TYPE'] == 'IN MARKET')
    ]['CG_ECN'].dropna().tolist()
    
    if not new_customers:
        return None
    
    # Get existing AU coordinates
    existing_au_info = branch_df[branch_df['BRANCH_AU'] == tagged_to_au]
    if len(existing_au_info) == 0:
        return None
    
    existing_au_lat = safe_get_value(existing_au_info.iloc[0], 'BRANCH_LAT_NUM', None)
    existing_au_lon = safe_get_value(existing_au_info.iloc[0], 'BRANCH_LON_NUM', None)
    
    if pd.isna(existing_au_lat) or pd.isna(existing_au_lon):
        return None
    
    # Get customer coordinates and calculate distances
    customer_data = client_groups_df[client_groups_df['CG_ECN'].isin(new_customers)]
    
    if len(customer_data) == 0:
        return None
    
    max_distance = 0
    valid_distances_found = False
    
    for _, customer in customer_data.iterrows():
        cust_lat = safe_get_value(customer, 'CG_LAT_NUM', None)
        cust_lon = safe_get_value(customer, 'CG_LON_NUM', None)
        
        if pd.notna(cust_lat) and pd.notna(cust_lon):
            distance = haversine_distance_vectorized(existing_au_lat, existing_au_lon, cust_lat, cust_lon)
            if not np.isinf(distance) and distance >= 0:
                max_distance = max(max_distance, distance)
                valid_distances_found = True
    
    # If no valid distances found, return None
    if not valid_distances_found:
        return None
    
    # Return movement status based on maximum distance
    return 'NO TRANSFER' if max_distance < 20 else 'TRANSFER'

def apply_special_au_500_tagging(tagging_results_df, customer_au_assignments, client_groups_df, branch_df):
    """Special function to tag AU 500 portfolio to manager JOSHUA NGUYEN"""
    au_500_mask = tagging_results_df['NEW_AU'] == '500'
    
    if not au_500_mask.any():
        return tagging_results_df
    
    modified_results = tagging_results_df.copy()
    
    for idx in modified_results[au_500_mask].index:
        new_type = modified_results.loc[idx, 'NEW_TYPE']
        
        new_customers = customer_au_assignments[
            (customer_au_assignments['ASSIGNED_AU'] == '500')
        ]['CG_ECN'].dropna().tolist()
        
        new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(
            new_customers, client_groups_df
        )
        
        new_avg_au_customer_distance = calculate_avg_au_customer_distance(
            '500', new_customers, branch_df, client_groups_df
        )
        
        modified_results.loc[idx, 'TAGGED_TO_PORTFOLIO'] = ''
        modified_results.loc[idx, 'TAGGED_TO_EMPLOYEE'] = ''
        modified_results.loc[idx, 'TAGGED_TO_MANAGER'] = 'JOSHUA NGUYEN'
        modified_results.loc[idx, 'TAGGED_TO_DIRECTOR'] = ''
        modified_results.loc[idx, 'TAGGED_TO_AU'] = ''
        modified_results.loc[idx, 'TAGGING_CRITERIA'] = 'SPECIAL_AU_500_ASSIGNMENT'
        modified_results.loc[idx, 'DISTANCE_MILES'] = None
        modified_results.loc[idx, 'CUSTOMER_OVERLAP_COUNT'] = 0
        modified_results.loc[idx, 'EXISTING_PORTFOLIO_SIZE'] = 0
        modified_results.loc[idx, 'EXISTING_AVG_DEPOSIT_BAL'] = None
        modified_results.loc[idx, 'EXISTING_AVG_GROSS_SALES'] = None
        modified_results.loc[idx, 'EXISTING_AVG_BANK_REVENUE'] = None
        modified_results.loc[idx, 'NEW_AVG_DEPOSIT_BAL'] = new_avg_deposit
        modified_results.loc[idx, 'NEW_AVG_GROSS_SALES'] = new_avg_gross_sales
        modified_results.loc[idx, 'NEW_AVG_BANK_REVENUE'] = new_avg_bank_revenue
        modified_results.loc[idx, 'NEW_AVG_AU_CUSTOMER_DISTANCE_MILES'] = new_avg_au_customer_distance
        modified_results.loc[idx, 'EXISTING_AVG_AU_CUSTOMER_DISTANCE_MILES'] = None
        
        movement_status = calculate_movement_status(
            '500', new_type, '', 'SPECIAL_AU_500_ASSIGNMENT',
            customer_au_assignments, client_groups_df, branch_df
        )
        modified_results.loc[idx, 'MOVEMENT'] = movement_status
    
    return modified_results

def tag_new_portfolios_to_mojgan_portfolios(customer_au_assignments, active_portfolio_df, client_groups_df, branch_df):
    """Main function to tag new portfolios to existing MOJGAN MADADI portfolios using AU locations"""
    
    print("=== TAGGING NEW PORTFOLIOS TO MOJGAN MADADI PORTFOLIOS ===")
    
    # Get unique new portfolios with customer counts
    new_portfolios = get_unique_new_portfolios(customer_au_assignments)
    new_inmarket = new_portfolios[new_portfolios['TYPE'] == 'IN MARKET']
    new_centralized = new_portfolios[new_portfolios['TYPE'] == 'CENTRALIZED']
    
    print(f"New portfolios: IN MARKET: {len(new_inmarket)}, CENTRALIZED: {len(new_centralized)}")
    
    # Get existing Mojgan portfolios
    existing_inmarket, existing_centralized, existing_portfolio_customers = get_existing_portfolios_mojgan(
        active_portfolio_df, client_groups_df, branch_df
    )
    
    # Get ALL existing portfolios for distance-based manager/director assignment
    all_existing_portfolios = get_all_existing_portfolios(active_portfolio_df, branch_df)
    
    print(f"Existing MOJGAN MADADI portfolios: IN MARKET: {len(existing_inmarket)}, CENTRALIZED: {len(existing_centralized)}")
    print(f"Total existing portfolios for distance calculation: {len(all_existing_portfolios)}")
    
    all_tags = []
    
    # PHASE 1: Tag IN MARKET portfolios by AU distance
    if len(new_inmarket) > 0 and len(existing_inmarket) > 0:
        used_existing_inmarket = set()
        
        # Create all distance combinations using AU locations
        inmarket_combinations = []
        for _, new_portfolio in new_inmarket.iterrows():
            new_au = new_portfolio['ASSIGNED_AU']
            
            # Get the coordinates of the new AU from branch_df
            new_au_info = branch_df[branch_df['BRANCH_AU'] == new_au]
            if len(new_au_info) == 0:
                continue  # Skip if AU not found in branch_df
            
            new_lat = safe_get_value(new_au_info.iloc[0], 'BRANCH_LAT_NUM', 0)
            new_lon = safe_get_value(new_au_info.iloc[0], 'BRANCH_LON_NUM', 0)
            
            # Get customers for this new portfolio
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == 'IN MARKET')
            ]['CG_ECN'].dropna().tolist()
            
            for _, existing_portfolio in existing_inmarket.iterrows():
                existing_lat = safe_get_value(existing_portfolio, 'BRANCH_LAT_NUM', 0)
                existing_lon = safe_get_value(existing_portfolio, 'BRANCH_LON_NUM', 0)
                distance = haversine_distance_vectorized(new_lat, new_lon, existing_lat, existing_lon)
                
                inmarket_combinations.append({
                    'new_au': new_au,
                    'new_customer_count': new_portfolio['CUSTOMER_COUNT'],
                    'new_customers': new_customers,
                    'existing_portfolio': safe_get_value(existing_portfolio, 'PORT_CODE', ''),
                    'existing_employee': safe_get_value(existing_portfolio, 'EMPLOYEE_NAME', ''),
                    'existing_manager': safe_get_value(existing_portfolio, 'MANAGER_NAME', ''),
                    'existing_director': safe_get_value(existing_portfolio, 'DIRECTOR_NAME', ''),
                    'existing_au': safe_get_value(existing_portfolio, 'AU', ''),
                    'distance': float(distance) if not np.isinf(distance) else float('inf')
                })
        
        # Sort by distance and assign one-to-one
        inmarket_combinations_sorted = sorted([x for x in inmarket_combinations if x['distance'] != float('inf')], 
                                            key=lambda x: x['distance'])
        used_new_inmarket = set()
        
        for combo in inmarket_combinations_sorted:
            new_au = combo['new_au']
            existing_portfolio = combo['existing_portfolio']
            
            if new_au in used_new_inmarket or existing_portfolio in used_existing_inmarket or not existing_portfolio:
                continue
            
            # Calculate customer overlap
            existing_customers = existing_portfolio_customers.get(existing_portfolio, set())
            overlap_count = calculate_customer_overlap(combo['new_customers'], existing_customers)
            
            # Get financial metrics
            existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(existing_portfolio, client_groups_df)
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(combo['new_customers'], client_groups_df)
            
            # Calculate average AU-customer distances
            new_avg_au_customer_distance = calculate_avg_au_customer_distance(new_au, combo['new_customers'], branch_df, client_groups_df)
            existing_avg_au_customer_distance = calculate_avg_portfolio_customer_distance(existing_portfolio, client_groups_df, active_portfolio_df, branch_df)
            
            all_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': 'IN MARKET',
                'NEW_CUSTOMER_COUNT': combo['new_customer_count'],
                'TAGGED_TO_PORTFOLIO': existing_portfolio,
                'TAGGED_TO_EMPLOYEE': combo['existing_employee'],
                'TAGGED_TO_MANAGER': combo['existing_manager'],
                'TAGGED_TO_DIRECTOR': combo['existing_director'],
                'TAGGED_TO_AU': combo['existing_au'],
                'TAGGING_CRITERIA': 'CLOSEST_AU_DISTANCE',
                'DISTANCE_MILES': combo['distance'],
                'CUSTOMER_OVERLAP_COUNT': overlap_count,
                'EXISTING_PORTFOLIO_SIZE': existing_portfolio_size,
                'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
                'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
                'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
                'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue,
                'NEW_AVG_AU_CUSTOMER_DISTANCE_MILES': new_avg_au_customer_distance,
                'EXISTING_AVG_AU_CUSTOMER_DISTANCE_MILES': existing_avg_au_customer_distance
            })
            
            used_new_inmarket.add(new_au)
            used_existing_inmarket.add(existing_portfolio)
    else:
        used_new_inmarket = set()
    
    # Handle untagged IN MARKET portfolios (will be processed in Phase 2)
    untagged_inmarket_portfolios = []
    for _, new_portfolio in new_inmarket.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        if new_au not in used_new_inmarket:
            untagged_inmarket_portfolios.append({
                'au': new_au,
                'type': 'IN MARKET',
                'customer_count': new_portfolio['CUSTOMER_COUNT']
            })
    
    # Add all CENTRALIZED portfolios to untagged list (will be processed in Phase 2)
    untagged_centralized_portfolios = []
    for _, new_portfolio in new_centralized.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        untagged_centralized_portfolios.append({
            'au': new_au,
            'type': 'CENTRALIZED',
            'customer_count': new_portfolio['CUSTOMER_COUNT']
        })
    
    # Combine all untagged portfolios
    all_untagged_portfolios = untagged_inmarket_portfolios + untagged_centralized_portfolios
    
    # PHASE 2: Tag remaining portfolios to CENTRALIZED existing portfolios based on customer overlap
    if len(all_untagged_portfolios) > 0 and len(existing_centralized) > 0:
        used_existing_centralized = set()
        
        # Create all overlap combinations between untagged new portfolios and existing centralized portfolios
        overlap_combinations = []
        for untagged_portfolio in all_untagged_portfolios:
            new_au = untagged_portfolio['au']
            new_type = untagged_portfolio['type']
            
            # Get customers for this untagged portfolio
            # Handle both original and standardized TYPE values
            type_filter = new_type
            if new_type == 'IN MARKET':
                # Try both 'IN MARKET' and 'INMARKET' to handle inconsistencies
                new_customers = customer_au_assignments[
                    (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                    (customer_au_assignments['TYPE'].isin(['IN MARKET', 'INMARKET']))
                ]['CG_ECN'].dropna().tolist()
            else:
                new_customers = customer_au_assignments[
                    (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                    (customer_au_assignments['TYPE'] == new_type)
                ]['CG_ECN'].dropna().tolist()
            
            for _, existing_portfolio in existing_centralized.iterrows():
                existing_port_code = safe_get_value(existing_portfolio, 'PORT_CODE', '')
                if not existing_port_code:
                    continue
                    
                existing_customers = existing_portfolio_customers.get(existing_port_code, set())
                overlap_count = calculate_customer_overlap(new_customers, existing_customers)
                
                overlap_combinations.append({
                    'new_au': new_au,
                    'new_type': new_type,
                    'new_customer_count': untagged_portfolio['customer_count'],
                    'new_customers': new_customers,
                    'existing_portfolio': existing_port_code,
                    'existing_employee': safe_get_value(existing_portfolio, 'EMPLOYEE_NAME', ''),
                    'existing_manager': safe_get_value(existing_portfolio, 'MANAGER_NAME', ''),
                    'existing_director': safe_get_value(existing_portfolio, 'DIRECTOR_NAME', ''),
                    'overlap_count': overlap_count,
                    'existing_customers': existing_customers
                })
        
        # Sort by overlap count (descending) and assign one-to-one
        overlap_combinations_sorted = sorted(overlap_combinations, key=lambda x: x['overlap_count'], reverse=True)
        used_new_untagged = set()
        
        for combo in overlap_combinations_sorted:
            new_au = combo['new_au']
            existing_portfolio = combo['existing_portfolio']
            
            if new_au in used_new_untagged or existing_portfolio in used_existing_centralized or combo['overlap_count'] == 0:
                continue
            
            # Get financial metrics
            existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(existing_portfolio, client_groups_df)
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(combo['new_customers'], client_groups_df)
            
            # Calculate average AU-customer distances
            new_avg_au_customer_distance = calculate_avg_au_customer_distance(new_au, combo['new_customers'], branch_df, client_groups_df)
            existing_avg_au_customer_distance = calculate_avg_portfolio_customer_distance(existing_portfolio, client_groups_df, active_portfolio_df, branch_df)
            
            all_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': combo['new_type'],
                'NEW_CUSTOMER_COUNT': combo['new_customer_count'],
                'TAGGED_TO_PORTFOLIO': existing_portfolio,
                'TAGGED_TO_EMPLOYEE': combo['existing_employee'],
                'TAGGED_TO_MANAGER': combo['existing_manager'],
                'TAGGED_TO_DIRECTOR': combo['existing_director'],
                'TAGGED_TO_AU': 'N/A (CENTRALIZED)',
                'TAGGING_CRITERIA': 'MAX_CUSTOMER_OVERLAP',
                'DISTANCE_MILES': None,
                'CUSTOMER_OVERLAP_COUNT': combo['overlap_count'],
                'EXISTING_PORTFOLIO_SIZE': existing_portfolio_size,
                'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
                'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
                'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
                'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue,
                'NEW_AVG_AU_CUSTOMER_DISTANCE_MILES': new_avg_au_customer_distance,
                'EXISTING_AVG_AU_CUSTOMER_DISTANCE_MILES': existing_avg_au_customer_distance
            })
            
            used_new_untagged.add(new_au)
            used_existing_centralized.add(existing_portfolio)
    else:
        used_new_untagged = set()
    
    # PHASE 3: Handle completely untagged portfolios - assign nearest manager/director only
    for untagged_portfolio in all_untagged_portfolios:
        new_au = untagged_portfolio['au']
        if new_au not in used_new_untagged:
            new_type = untagged_portfolio['type']
            # Get customers for this untagged portfolio  
            # Handle both original and standardized TYPE values
            if new_type == 'IN MARKET':
                # Try both 'IN MARKET' and 'INMARKET' to handle inconsistencies
                new_customers = customer_au_assignments[
                    (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                    (customer_au_assignments['TYPE'].isin(['IN MARKET', 'INMARKET']))
                ]['CG_ECN'].dropna().tolist()
            else:
                new_customers = customer_au_assignments[
                    (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                    (customer_au_assignments['TYPE'] == new_type)
                ]['CG_ECN'].dropna().tolist()
            
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(new_customers, client_groups_df)
            
            # Calculate average AU-customer distance for new portfolio
            new_avg_au_customer_distance = calculate_avg_au_customer_distance(new_au, new_customers, branch_df, client_groups_df)
            
            # Find nearest portfolio for manager/director assignment
            nearest_manager, nearest_director, nearest_distance = find_nearest_portfolio_for_manager_director(
                new_au, branch_df, all_existing_portfolios
            )
            
            all_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': new_type,
                'NEW_CUSTOMER_COUNT': untagged_portfolio['customer_count'],
                'TAGGED_TO_PORTFOLIO': '',  # No portfolio assignment
                'TAGGED_TO_EMPLOYEE': '',  # No employee assignment
                'TAGGED_TO_MANAGER': nearest_manager if nearest_manager else '',
                'TAGGED_TO_DIRECTOR': nearest_director if nearest_director else '',
                'TAGGED_TO_AU': '',
                'TAGGING_CRITERIA': 'NEAREST_MANAGER_DIRECTOR_ONLY',
                'DISTANCE_MILES': nearest_distance if not np.isinf(nearest_distance) else None,
                'CUSTOMER_OVERLAP_COUNT': 0,
                'EXISTING_PORTFOLIO_SIZE': 0,
                'EXISTING_AVG_DEPOSIT_BAL': None,
                'EXISTING_AVG_GROSS_SALES': None,
                'EXISTING_AVG_BANK_REVENUE': None,
                'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue,
                'NEW_AVG_AU_CUSTOMER_DISTANCE_MILES': new_avg_au_customer_distance,
                'EXISTING_AVG_AU_CUSTOMER_DISTANCE_MILES': None
            })
    
    # Calculate movement status for each tag
    for tag in all_tags:
        movement_status = calculate_movement_status(
            tag['NEW_AU'],
            tag['NEW_TYPE'], 
            tag['TAGGED_TO_AU'],
            tag['TAGGING_CRITERIA'],
            customer_au_assignments,
            client_groups_df,
            branch_df
        )
        tag['MOVEMENT'] = movement_status
    
    # Create results dataframe
    tagging_results = pd.DataFrame(all_tags)
    
    # Apply special AU 500 tagging
    tagging_results = apply_special_au_500_tagging(tagging_results, customer_au_assignments, client_groups_df, branch_df)
    
    # Count new customers not in active portfolios
    new_customers_count = count_new_customers_not_in_active_portfolios(customer_au_assignments, client_groups_df)
    
    # Print summary with distance information
    print(f"\nSUMMARY:")
    print(f"New customers not in any active portfolio: {new_customers_count}")
    if len(tagging_results) > 0:
        tagged_portfolios = tagging_results[tagging_results['TAGGED_TO_PORTFOLIO'] != '']
        manager_director_only = tagging_results[tagging_results['TAGGING_CRITERIA'] == 'NEAREST_MANAGER_DIRECTOR_ONLY']
        special_au_500 = tagging_results[tagging_results['TAGGING_CRITERIA'] == 'SPECIAL_AU_500_ASSIGNMENT']
        
        print(f"Tagged to MOJGAN MADADI portfolios: {len(tagged_portfolios)}")
        print(f"Assigned nearest manager/director only: {len(manager_director_only)}")
        print(f"Special AU 500 assignment: {len(special_au_500)}")
        
        if len(tagged_portfolios) > 0:
            print(f"Total customer overlap: {tagged_portfolios['CUSTOMER_OVERLAP_COUNT'].sum()}")
            
            # Distance statistics
            new_distances = tagged_portfolios['NEW_AVG_AU_CUSTOMER_DISTANCE_MILES'].dropna()
            existing_distances = tagged_portfolios['EXISTING_AVG_AU_CUSTOMER_DISTANCE_MILES'].dropna()
            
            if len(new_distances) > 0:
                print(f"Average AU-Customer distance for new portfolios: {new_distances.mean():.2f} miles")
            if len(existing_distances) > 0:
                print(f"Average AU-Customer distance for existing portfolios: {existing_distances.mean():.2f} miles")
            
        # Break down by tagging criteria
        distance_tagged = tagging_results[tagging_results['TAGGING_CRITERIA'] == 'CLOSEST_AU_DISTANCE']
        overlap_tagged = tagging_results[tagging_results['TAGGING_CRITERIA'] == 'MAX_CUSTOMER_OVERLAP']
        print(f"Tagged by AU distance (IN MARKET): {len(distance_tagged)}")
        print(f"Tagged by customer overlap (CENTRALIZED): {len(overlap_tagged)}")
        print(f"Assigned nearest manager/director only: {len(manager_director_only)}")
        print(f"Special AU 500 assignment: {len(special_au_500)}")
        
        # Movement summary
        movement_counts = tagging_results['MOVEMENT'].value_counts(dropna=False)
        print(f"\nMOVEMENT SUMMARY:")
        for status, count in movement_counts.items():
            status_label = 'NULL' if pd.isna(status) else status
            print(f"{status_label}: {count}")
    
    return tagging_results
