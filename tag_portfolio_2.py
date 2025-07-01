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

def get_existing_portfolios_mojgan(active_portfolio_df, client_groups_df, branch_df):
    """Get existing portfolios under MOJGAN MADADI"""
    # Clean the dataframes first
    active_portfolio_clean = active_portfolio_df.copy()
    active_portfolio_clean['DIRECTOR_NAME'] = active_portfolio_clean['DIRECTOR_NAME'].fillna('')
    active_portfolio_clean['ROLE_TYPE'] = active_portfolio_clean['ROLE_TYPE'].fillna('')
    active_portfolio_clean['ISACTIVE'] = pd.to_numeric(active_portfolio_clean['ISACTIVE'], errors='coerce').fillna(0)
    
    # All existing portfolios under MOJGAN MADADI
    existing_portfolios = active_portfolio_clean[
        (active_portfolio_clean['ISACTIVE'] == 1) & 
        (active_portfolio_clean['DIRECTOR_NAME'] == 'MOJGAN MADADI')
    ].copy()
    
    # Merge with branch data to get coordinates for IN MARKET portfolios
    if len(existing_portfolios) > 0:
        existing_portfolios = existing_portfolios.merge(
            branch_df[['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
            left_on='AU', right_on='BRANCH_AU', how='left'
        )
    
    # Get customer lists for ALL portfolios
    portfolio_customers = {}
    
    # Clean client_groups_df
    client_groups_clean = client_groups_df.copy()
    client_groups_clean['ISPORTFOLIOACTIVE'] = pd.to_numeric(client_groups_clean['ISPORTFOLIOACTIVE'], errors='coerce').fillna(0)
    
    # Add portfolio customers
    for _, portfolio in existing_portfolios.iterrows():
        port_code = portfolio['PORT_CODE']
        if pd.notna(port_code):
            portfolio_customer_list = client_groups_clean[
                (client_groups_clean['CG_PORTFOLIO_CD'] == port_code) &
                (client_groups_clean['ISPORTFOLIOACTIVE'] == 1)
            ]['CG_ECN'].tolist()
            # Filter out NaN values
            portfolio_customer_list = [x for x in portfolio_customer_list if pd.notna(x)]
            portfolio_customers[port_code] = set(portfolio_customer_list)
    
    return existing_portfolios, portfolio_customers

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

def calculate_distance_for_inmarket(new_au, existing_portfolio, branch_df):
    """Calculate distance between AUs for IN MARKET portfolios"""
    # Get coordinates of the new AU
    new_au_info = branch_df[branch_df['BRANCH_AU'] == new_au]
    if len(new_au_info) == 0:
        return None
    
    new_lat = safe_get_value(new_au_info.iloc[0], 'BRANCH_LAT_NUM', 0)
    new_lon = safe_get_value(new_au_info.iloc[0], 'BRANCH_LON_NUM', 0)
    
    # Get coordinates of existing portfolio AU
    existing_lat = safe_get_value(existing_portfolio, 'BRANCH_LAT_NUM', 0)
    existing_lon = safe_get_value(existing_portfolio, 'BRANCH_LON_NUM', 0)
    
    distance = haversine_distance_vectorized(new_lat, new_lon, existing_lat, existing_lon)
    return distance if not np.isinf(distance) else None

def tag_new_portfolios_to_mojgan_portfolios(customer_au_assignments, active_portfolio_df, client_groups_df, branch_df):
    """Main function to tag new portfolios to existing MOJGAN MADADI portfolios using customer overlap"""
    
    print("=== TAGGING NEW PORTFOLIOS TO MOJGAN MADADI PORTFOLIOS (CUSTOMER OVERLAP) ===")
    
    # Get unique new portfolios with customer counts
    new_portfolios = get_unique_new_portfolios(customer_au_assignments)
    
    print(f"New portfolios: {len(new_portfolios)} total")
    print(f"  - IN MARKET: {len(new_portfolios[new_portfolios['TYPE'] == 'INMARKET'])}")
    print(f"  - CENTRALIZED: {len(new_portfolios[new_portfolios['TYPE'] == 'CENTRALIZED'])}")
    
    # Get existing Mojgan portfolios
    existing_portfolios, existing_portfolio_customers = get_existing_portfolios_mojgan(
        active_portfolio_df, client_groups_df, branch_df
    )
    
    existing_inmarket = existing_portfolios[existing_portfolios['ROLE_TYPE'] == 'IN MARKET']
    existing_centralized = existing_portfolios[existing_portfolios['ROLE_TYPE'] == 'CENTRALIZED']
    
    print(f"Existing MOJGAN MADADI portfolios: {len(existing_portfolios)} total")
    print(f"  - IN MARKET: {len(existing_inmarket)}")
    print(f"  - CENTRALIZED: {len(existing_centralized)}")
    
    all_tags = []
    
    # Create all overlap combinations between new and existing portfolios
    overlap_combinations = []
    
    for _, new_portfolio in new_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        new_type = new_portfolio['TYPE']
        
        # Get customers for this new portfolio
        new_customers = customer_au_assignments[
            (customer_au_assignments['ASSIGNED_AU'] == new_au) &
            (customer_au_assignments['TYPE'] == new_type)
        ]['CG_ECN'].dropna().tolist()
        
        for _, existing_portfolio in existing_portfolios.iterrows():
            existing_port_code = safe_get_value(existing_portfolio, 'PORT_CODE', '')
            if not existing_port_code:
                continue
                
            existing_customers = existing_portfolio_customers.get(existing_port_code, set())
            overlap_count = calculate_customer_overlap(new_customers, existing_customers)
            
            # Calculate distance if new portfolio is IN MARKET
            distance = None
            if new_type == 'INMARKET':
                distance = calculate_distance_for_inmarket(new_au, existing_portfolio, branch_df)
            
            overlap_combinations.append({
                'new_au': new_au,
                'new_type': new_type,
                'new_customer_count': new_portfolio['CUSTOMER_COUNT'],
                'new_customers': new_customers,
                'existing_portfolio': existing_port_code,
                'existing_employee': safe_get_value(existing_portfolio, 'EMPLOYEE_NAME', ''),
                'existing_manager': safe_get_value(existing_portfolio, 'MANAGER_NAME', ''),
                'existing_director': safe_get_value(existing_portfolio, 'DIRECTOR_NAME', ''),
                'existing_au': safe_get_value(existing_portfolio, 'AU', ''),
                'existing_type': safe_get_value(existing_portfolio, 'ROLE_TYPE', ''),
                'overlap_count': overlap_count,
                'distance': distance
            })
    
    # Sort by overlap count (descending) and assign one-to-one
    overlap_combinations_sorted = sorted(overlap_combinations, key=lambda x: x['overlap_count'], reverse=True)
    used_existing_portfolios = set()
    used_new_portfolios = set()
    
    for combo in overlap_combinations_sorted:
        new_au = combo['new_au']
        existing_portfolio = combo['existing_portfolio']
        
        if new_au in used_new_portfolios or existing_portfolio in used_existing_portfolios or combo['overlap_count'] == 0:
            continue
        
        # Get financial metrics
        existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(existing_portfolio, client_groups_df)
        new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(combo['new_customers'], client_groups_df)
        
        # Determine tagging criteria and AU info
        tagged_to_au = combo['existing_au'] if combo['existing_type'] == 'IN MARKET' else 'N/A (CENTRALIZED)'
        tagging_criteria = f"MAX_CUSTOMER_OVERLAP ({combo['new_type']} to {combo['existing_type']})"
        
        all_tags.append({
            'NEW_AU': new_au,
            'NEW_TYPE': combo['new_type'],
            'NEW_CUSTOMER_COUNT': combo['new_customer_count'],
            'TAGGED_TO_PORTFOLIO': existing_portfolio,
            'TAGGED_TO_EMPLOYEE': combo['existing_employee'],
            'TAGGED_TO_MANAGER': combo['existing_manager'],
            'TAGGED_TO_DIRECTOR': combo['existing_director'],
            'TAGGED_TO_AU': tagged_to_au,
            'TAGGED_TO_TYPE': combo['existing_type'],
            'TAGGING_CRITERIA': tagging_criteria,
            'DISTANCE_MILES': combo['distance'],
            'CUSTOMER_OVERLAP_COUNT': combo['overlap_count'],
            'EXISTING_PORTFOLIO_SIZE': existing_portfolio_size,
            'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
            'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
            'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
            'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
            'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
            'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
        })
        
        used_new_portfolios.add(new_au)
        used_existing_portfolios.add(existing_portfolio)
    
    # Handle untagged portfolios - tag to nearest available portfolio's manager/director
    for _, new_portfolio in new_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        if new_au not in used_new_portfolios:
            new_type = new_portfolio['TYPE']
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == new_type)
            ]['CG_ECN'].dropna().tolist()
            
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(new_customers, client_groups_df)
            
            # Find nearest available (unused) portfolio for manager/director assignment
            nearest_manager = ''
            nearest_director = ''
            nearest_distance = float('inf')
            nearest_au = ''
            nearest_type = ''
            nearest_criteria = 'UNTAGGED'
            
            # Get available (unused) existing portfolios
            available_portfolios = existing_portfolios[
                ~existing_portfolios['PORT_CODE'].isin(used_existing_portfolios)
            ]
            
            if len(available_portfolios) > 0:
                # Get coordinates of the new AU
                new_au_info = branch_df[branch_df['BRANCH_AU'] == new_au]
                if len(new_au_info) > 0:
                    new_lat = safe_get_value(new_au_info.iloc[0], 'BRANCH_LAT_NUM', 0)
                    new_lon = safe_get_value(new_au_info.iloc[0], 'BRANCH_LON_NUM', 0)
                    
                    # Check all available portfolios for distance
                    for _, available_portfolio in available_portfolios.iterrows():
                        existing_type = safe_get_value(available_portfolio, 'ROLE_TYPE', '')
                        
                        if existing_type == 'IN MARKET':
                            # Calculate distance for IN MARKET portfolios
                            existing_lat = safe_get_value(available_portfolio, 'BRANCH_LAT_NUM', 0)
                            existing_lon = safe_get_value(available_portfolio, 'BRANCH_LON_NUM', 0)
                            distance = haversine_distance_vectorized(new_lat, new_lon, existing_lat, existing_lon)
                        else:
                            # For CENTRALIZED portfolios, assign a large but finite distance
                            distance = 9999.0
                        
                        if distance < nearest_distance and not np.isinf(distance):
                            nearest_distance = distance
                            nearest_manager = safe_get_value(available_portfolio, 'MANAGER_NAME', '')
                            nearest_director = safe_get_value(available_portfolio, 'DIRECTOR_NAME', '')
                            nearest_au = safe_get_value(available_portfolio, 'AU', '') if existing_type == 'IN MARKET' else 'N/A (CENTRALIZED)'
                            nearest_type = existing_type
                            nearest_criteria = 'NEAREST_MANAGER_BY_DISTANCE' if existing_type == 'IN MARKET' else 'NEAREST_MANAGER_BY_CENTRALIZED'
                
                # If still no assignment and we have available portfolios, use the first one
                if nearest_distance == float('inf') and len(available_portfolios) > 0:
                    first_available = available_portfolios.iloc[0]
                    nearest_manager = safe_get_value(first_available, 'MANAGER_NAME', '')
                    nearest_director = safe_get_value(first_available, 'DIRECTOR_NAME', '')
                    nearest_au = 'N/A (CENTRALIZED)'
                    nearest_type = safe_get_value(first_available, 'ROLE_TYPE', '')
                    nearest_distance = None
                    nearest_criteria = 'NEAREST_MANAGER_BY_FALLBACK'
            
            all_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': new_type,
                'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                'TAGGED_TO_PORTFOLIO': '',  # Keep portfolio blank
                'TAGGED_TO_EMPLOYEE': '',   # Keep employee blank
                'TAGGED_TO_MANAGER': nearest_manager,
                'TAGGED_TO_DIRECTOR': nearest_director,
                'TAGGED_TO_AU': nearest_au,
                'TAGGED_TO_TYPE': nearest_type,
                'TAGGING_CRITERIA': nearest_criteria,
                'DISTANCE_MILES': nearest_distance,
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
        tagged_portfolios = tagging_results[tagging_results['TAGGED_TO_PORTFOLIO'] != '']
        untagged_portfolios = tagging_results[tagging_results['TAGGED_TO_PORTFOLIO'] == '']
        print(f"Tagged to MOJGAN MADADI portfolios: {len(tagged_portfolios)}")
        print(f"Untagged (left blank): {len(untagged_portfolios)}")
        
        if len(tagged_portfolios) > 0:
            print(f"Total customer overlap: {tagged_portfolios['CUSTOMER_OVERLAP_COUNT'].sum()}")
            
            # Break down by tagging combinations
            overlap_tagged = tagged_portfolios[tagged_portfolios['TAGGING_CRITERIA'].str.contains('MAX_CUSTOMER_OVERLAP', na=False)]
            manager_tagged = tagged_portfolios[tagged_portfolios['TAGGING_CRITERIA'].str.contains('NEAREST_MANAGER', na=False)]
            
            print(f"Tagging breakdown:")
            print(f"  - Tagged by customer overlap: {len(overlap_tagged)}")
            print(f"  - Tagged to manager/director only: {len(manager_tagged)}")
            
            if len(overlap_tagged) > 0:
                inmarket_to_inmarket = overlap_tagged[overlap_tagged['TAGGING_CRITERIA'].str.contains('INMARKET to IN MARKET', na=False)]
                inmarket_to_centralized = overlap_tagged[overlap_tagged['TAGGING_CRITERIA'].str.contains('INMARKET to CENTRALIZED', na=False)]
                centralized_to_inmarket = overlap_tagged[overlap_tagged['TAGGING_CRITERIA'].str.contains('CENTRALIZED to IN MARKET', na=False)]
                centralized_to_centralized = overlap_tagged[overlap_tagged['TAGGING_CRITERIA'].str.contains('CENTRALIZED to CENTRALIZED', na=False)]
                
                print(f"  Customer overlap breakdown:")
                print(f"    - IN MARKET to IN MARKET: {len(inmarket_to_inmarket)}")
                print(f"    - IN MARKET to CENTRALIZED: {len(inmarket_to_centralized)}")
                print(f"    - CENTRALIZED to IN MARKET: {len(centralized_to_inmarket)}")
                print(f"    - CENTRALIZED to CENTRALIZED: {len(centralized_to_centralized)}")
            
            if len(manager_tagged) > 0:
                distance_manager = manager_tagged[manager_tagged['TAGGING_CRITERIA'] == 'NEAREST_MANAGER_BY_DISTANCE']
                centralized_manager = manager_tagged[manager_tagged['TAGGING_CRITERIA'] == 'NEAREST_MANAGER_BY_CENTRALIZED']
                fallback_manager = manager_tagged[manager_tagged['TAGGING_CRITERIA'] == 'NEAREST_MANAGER_BY_FALLBACK']
                
                print(f"  Manager/director tagging breakdown:")
                print(f"    - By nearest AU distance: {len(distance_manager)}")
                print(f"    - By centralized assignment: {len(centralized_manager)}")
                print(f"    - By fallback: {len(fallback_manager)}")
            
            # Show distance statistics for IN MARKET new portfolios
            inmarket_tagged = tagged_portfolios[tagged_portfolios['NEW_TYPE'] == 'INMARKET']
            if len(inmarket_tagged) > 0:
                valid_distances = inmarket_tagged[inmarket_tagged['DISTANCE_MILES'].notna() & (inmarket_tagged['DISTANCE_MILES'] != 9999.0)]['DISTANCE_MILES']
                if len(valid_distances) > 0:
                    print(f"Distance statistics for IN MARKET portfolios:")
                    print(f"  - Average distance: {valid_distances.mean():.2f} miles")
                    print(f"  - Min distance: {valid_distances.min():.2f} miles")
                    print(f"  - Max distance: {valid_distances.max():.2f} miles")
    
    return tagging_results

# Usage:
# tagging_results = tag_new_portfolios_to_mojgan_portfolios(customer_au_assignments, ACTIVE_PORTFOLIO, CLIENT_GROUPS_DF_NEW, branch_df)
# tagging_results.to_csv('portfolio_tagging_results.csv', index=False)
