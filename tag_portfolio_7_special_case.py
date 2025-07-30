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

def get_custom_portfolios_info(custom_au_banker_portfolio_list, client_groups_df, branch_df):
    """Get portfolio information from custom AU/Banker/Portfolio list"""
    
    # Convert list to DataFrame for easier processing
    custom_portfolios_df = pd.DataFrame(custom_au_banker_portfolio_list, 
                                      columns=['AU', 'BANKER_ID', 'PORTFOLIO_CODE'])
    
    # Merge with branch info to get coordinates
    custom_portfolios_with_coords = custom_portfolios_df.merge(
        branch_df[['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
        left_on='AU', right_on='BRANCH_AU', how='left'
    )
    
    # Get customer lists for portfolios that have portfolio codes
    portfolio_customers = {}
    
    # Clean client_groups_df
    client_groups_clean = client_groups_df.copy()
    client_groups_clean['ISPORTFOLIOACTIVE'] = pd.to_numeric(client_groups_clean['ISPORTFOLIOACTIVE'], errors='coerce').fillna(0)
    
    for _, row in custom_portfolios_with_coords.iterrows():
        portfolio_code = row['PORTFOLIO_CODE']
        if pd.notna(portfolio_code) and portfolio_code is not None:
            portfolio_customer_list = client_groups_clean[
                (client_groups_clean['CG_PORTFOLIO_CD'] == portfolio_code) &
                (client_groups_clean['ISPORTFOLIOACTIVE'] == 1)
            ]['CG_ECN'].tolist()
            # Filter out NaN values
            portfolio_customer_list = [x for x in portfolio_customer_list if pd.notna(x)]
            portfolio_customers[portfolio_code] = set(portfolio_customer_list)
        else:
            portfolio_customers[f"AU_{row['AU']}"] = set()  # Empty set for null portfolio codes
    
    return custom_portfolios_with_coords, portfolio_customers

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
    if pd.isna(portfolio_code) or portfolio_code is None:
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

def safe_get_value(row, column, default=''):
    """Safely get value from pandas row, handling NaN"""
    try:
        value = row[column]
        return value if pd.notna(value) else default
    except (KeyError, IndexError):
        return default

def get_unique_new_portfolios(customer_au_assignments):
    """Get unique new portfolios with their customer counts"""
    # Clean data before grouping
    clean_assignments = customer_au_assignments.copy()
    clean_assignments = clean_assignments.dropna(subset=['ASSIGNED_AU', 'TYPE', 'CG_ECN'])
    
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

def tag_new_portfolios_to_custom_portfolios(customer_au_assignments, custom_au_banker_portfolio_list, 
                                          active_portfolio_df, client_groups_df, branch_df):
    """
    Main function to tag new portfolios to custom AU/Banker/Portfolio list using AU locations
    
    Parameters:
    - customer_au_assignments: DataFrame with new portfolio assignments
    - custom_au_banker_portfolio_list: List of lists [[AU, Banker_ID, Portfolio_Code], ...]
    - active_portfolio_df: DataFrame for fallback manager/director assignment
    - client_groups_df: DataFrame with customer/portfolio information
    - branch_df: DataFrame with branch coordinates
    """
    
    print("=== TAGGING NEW PORTFOLIOS TO CUSTOM AU/BANKER/PORTFOLIO LIST ===")
    
    # Get unique new portfolios with customer counts
    new_portfolios = get_unique_new_portfolios(customer_au_assignments)
    new_inmarket = new_portfolios[new_portfolios['TYPE'] == 'INMARKET']
    new_centralized = new_portfolios[new_portfolios['TYPE'] == 'CENTRALIZED']
    
    print(f"New portfolios: IN MARKET: {len(new_inmarket)}, CENTRALIZED: {len(new_centralized)}")
    
    # Get custom portfolio information
    custom_portfolios_with_coords, custom_portfolio_customers = get_custom_portfolios_info(
        custom_au_banker_portfolio_list, client_groups_df, branch_df
    )
    
    # Separate custom portfolios into those with and without portfolio codes
    custom_with_portfolio_codes = custom_portfolios_with_coords[
        custom_portfolios_with_coords['PORTFOLIO_CODE'].notna() & 
        (custom_portfolios_with_coords['PORTFOLIO_CODE'] != '')
    ]
    custom_without_portfolio_codes = custom_portfolios_with_coords[
        custom_portfolios_with_coords['PORTFOLIO_CODE'].isna() | 
        (custom_portfolios_with_coords['PORTFOLIO_CODE'] == '')
    ]
    
    # Get ALL existing portfolios for distance-based manager/director assignment
    all_existing_portfolios = get_all_existing_portfolios(active_portfolio_df, branch_df)
    
    print(f"Custom AUs with portfolio codes: {len(custom_with_portfolio_codes)}")
    print(f"Custom AUs without portfolio codes: {len(custom_without_portfolio_codes)}")
    print(f"Total existing portfolios for fallback distance calculation: {len(all_existing_portfolios)}")
    
    all_tags = []
    
    # PHASE 1: Tag ALL new portfolios by AU distance to custom AUs (regardless of IN MARKET/CENTRALIZED)
    all_new_portfolios = pd.concat([new_inmarket, new_centralized], ignore_index=True)
    used_new_portfolios = set()
    used_custom_portfolios = set()
    
    if len(all_new_portfolios) > 0 and len(custom_portfolios_with_coords) > 0:
        
        # Create all distance combinations using AU locations
        distance_combinations = []
        for _, new_portfolio in all_new_portfolios.iterrows():
            new_au = new_portfolio['ASSIGNED_AU']
            
            # Get the coordinates of the new AU from branch_df
            new_au_info = branch_df[branch_df['BRANCH_AU'] == new_au]
            if len(new_au_info) == 0:
                continue  # Skip if AU not found in branch_df
            
            new_lat = safe_get_value(new_au_info.iloc[0], 'BRANCH_LAT_NUM', 0)
            new_lon = safe_get_value(new_au_info.iloc[0], 'BRANCH_LON_NUM', 0)
            
            for _, custom_portfolio in custom_portfolios_with_coords.iterrows():
                custom_lat = safe_get_value(custom_portfolio, 'BRANCH_LAT_NUM', 0)
                custom_lon = safe_get_value(custom_portfolio, 'BRANCH_LON_NUM', 0)
                distance = haversine_distance_vectorized(new_lat, new_lon, custom_lat, custom_lon)
                
                distance_combinations.append({
                    'new_au': new_au,
                    'new_type': new_portfolio['TYPE'],
                    'new_customer_count': new_portfolio['CUSTOMER_COUNT'],
                    'custom_au': safe_get_value(custom_portfolio, 'AU', ''),
                    'custom_banker_id': safe_get_value(custom_portfolio, 'BANKER_ID', ''),
                    'custom_portfolio_code': safe_get_value(custom_portfolio, 'PORTFOLIO_CODE', ''),
                    'distance': float(distance) if not np.isinf(distance) else float('inf')
                })
        
        # Sort by distance and assign one-to-one
        distance_combinations_sorted = sorted([x for x in distance_combinations if x['distance'] != float('inf')], 
                                            key=lambda x: x['distance'])
        
        for combo in distance_combinations_sorted:
            new_au = combo['new_au']
            custom_au = combo['custom_au']
            
            if new_au in used_new_portfolios or custom_au in used_custom_portfolios:
                continue
            
            # Calculate customer overlap if portfolio code exists
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == combo['new_type'])
            ]['CG_ECN'].dropna().tolist()
            
            custom_portfolio_code = combo['custom_portfolio_code']
            overlap_count = 0
            existing_portfolio_size = 0
            existing_avg_deposit = None
            existing_avg_gross_sales = None
            existing_avg_bank_revenue = None
            
            if pd.notna(custom_portfolio_code) and custom_portfolio_code != '':
                existing_customers = custom_portfolio_customers.get(custom_portfolio_code, set())
                overlap_count = calculate_customer_overlap(new_customers, existing_customers)
                existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(custom_portfolio_code, client_groups_df)
            
            # Get financial metrics for new portfolio
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(new_customers, client_groups_df)
            
            all_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': combo['new_type'],
                'NEW_CUSTOMER_COUNT': combo['new_customer_count'],
                'TAGGED_TO_PORTFOLIO': custom_portfolio_code if pd.notna(custom_portfolio_code) else '',
                'TAGGED_TO_EMPLOYEE': combo['custom_banker_id'],  # Using BANKER_ID as employee
                'TAGGED_TO_MANAGER': '',  # Will be filled from fallback if needed
                'TAGGED_TO_DIRECTOR': '',  # Will be filled from fallback if needed
                'TAGGED_TO_AU': custom_au,
                'TAGGING_CRITERIA': 'CLOSEST_AU_DISTANCE',
                'DISTANCE_MILES': combo['distance'],
                'CUSTOMER_OVERLAP_COUNT': overlap_count,
                'EXISTING_PORTFOLIO_SIZE': existing_portfolio_size,
                'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
                'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
                'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
                'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
            })
            
            used_new_portfolios.add(new_au)
            used_custom_portfolios.add(custom_au)
    
    # PHASE 2: Handle completely untagged portfolios - assign nearest manager/director from active_portfolio_df
    for _, new_portfolio in all_new_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        if new_au not in used_new_portfolios:
            new_type = new_portfolio['TYPE']
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == new_type)
            ]['CG_ECN'].dropna().tolist()
            
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(new_customers, client_groups_df)
            
            # Find nearest portfolio for manager/director assignment
            nearest_manager, nearest_director, nearest_distance = find_nearest_portfolio_for_manager_director(
                new_au, branch_df, all_existing_portfolios
            )
            
            all_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': new_type,
                'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                'TAGGED_TO_PORTFOLIO': '',  # No portfolio assignment
                'TAGGED_TO_EMPLOYEE': '',  # No employee assignment
                'TAGGED_TO_MANAGER': nearest_manager if nearest_manager else '',
                'TAGGED_TO_DIRECTOR': nearest_director if nearest_director else '',
                'TAGGED_TO_AU': '',
                'TAGGING_CRITERIA': 'NEAREST_MANAGER_DIRECTOR_FALLBACK',
                'DISTANCE_MILES': nearest_distance if not np.isinf(nearest_distance) else None,
                'CUSTOMER_OVERLAP_COUNT': 0,
                'EXISTING_PORTFOLIO_SIZE': 0,
                'EXISTING_AVG_DEPOSIT_BAL': None,
                'EXISTING_AVG_GROSS_SALES': None,
                'EXISTING_AVG_BANK_REVENUE': None,
                'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
            })
    
    # Create results dataframe
    tagging_results = pd.DataFrame(all_tags)
    
    # Count new customers not in active portfolios
    new_customers_count = count_new_customers_not_in_active_portfolios(customer_au_assignments, client_groups_df)
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"New customers not in any active portfolio: {new_customers_count}")
    if len(tagging_results) > 0:
        tagged_to_custom = tagging_results[tagging_results['TAGGING_CRITERIA'] == 'CLOSEST_AU_DISTANCE']
        fallback_assignments = tagging_results[tagging_results['TAGGING_CRITERIA'] == 'NEAREST_MANAGER_DIRECTOR_FALLBACK']
        
        print(f"Tagged to custom AU list: {len(tagged_to_custom)}")
        print(f"Assigned nearest manager/director (fallback): {len(fallback_assignments)}")
        
        if len(tagged_to_custom) > 0:
            # Break down by portfolio code availability
            with_portfolio_codes = tagged_to_custom[tagged_to_custom['TAGGED_TO_PORTFOLIO'] != '']
            without_portfolio_codes = tagged_to_custom[tagged_to_custom['TAGGED_TO_PORTFOLIO'] == '']
            
            print(f"  - Tagged to AUs with portfolio codes: {len(with_portfolio_codes)}")
            print(f"  - Tagged to AUs without portfolio codes: {len(without_portfolio_codes)}")
            
            if len(with_portfolio_codes) > 0:
                print(f"  - Total customer overlap (with portfolio codes): {with_portfolio_codes['CUSTOMER_OVERLAP_COUNT'].sum()}")
                
            print(f"  - Average distance for tagged portfolios: {tagged_to_custom['DISTANCE_MILES'].mean():.2f} miles")
    
    return tagging_results

# Usage example:
# custom_list = [[1, 123, 'P1'], [2, 234, 'P2'], [3, 345, None]]  # Portfolio code can be None/null
# tagging_results = tag_new_portfolios_to_custom_portfolios(
#     customer_au_assignments, 
#     custom_list, 
#     ACTIVE_PORTFOLIO, 
#     CLIENT_GROUPS_DF_NEW, 
#     branch_df
# )
# tagging_results.to_csv('custom_portfolio_tagging_results.csv', index=False)
