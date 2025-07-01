import pandas as pd
import numpy as np

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    R = 3959
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_portfolio_centroids(customer_au_assignments):
    """Calculate centroid coordinates for each new portfolio"""
    portfolio_centroids = customer_au_assignments.groupby(['ASSIGNED_AU', 'TYPE']).agg({
        'LAT_NUM': 'mean',
        'LON_NUM': 'mean',
        'CG_ECN': 'count'
    }).reset_index()
    portfolio_centroids.columns = ['ASSIGNED_AU', 'TYPE', 'CENTROID_LAT', 'CENTROID_LON', 'CUSTOMER_COUNT']
    return portfolio_centroids

def get_existing_portfolios_mojgan(active_portfolio_df, client_groups_df, branch_df):
    """Get existing portfolios under Mojgan Madadi"""
    # IN MARKET portfolios
    existing_inmarket = active_portfolio_df[
        (active_portfolio_df['ISACTIVE'] == 1) & 
        (active_portfolio_df['ROLE_TYPE'] == 'IN MARKET') &
        (active_portfolio_df['ROLE_TYPE'].notna()) &
        (active_portfolio_df['DIRECTOR_NAME'] == 'Mojgan Madadi')
    ].copy()
    
    if len(existing_inmarket) > 0:
        existing_inmarket = existing_inmarket.merge(
            branch_df[['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
            left_on='AU', right_on='BRANCH_AU', how='left'
        )
    
    # CENTRALIZED portfolios
    existing_centralized = active_portfolio_df[
        (active_portfolio_df['ISACTIVE'] == 1) & 
        (active_portfolio_df['ROLE_TYPE'] == 'CENTRALIZED') &
        (active_portfolio_df['ROLE_TYPE'].notna()) &
        (active_portfolio_df['DIRECTOR_NAME'] == 'Mojgan Madadi')
    ].copy()
    
    # Get customer lists for ALL portfolios (both IN MARKET and CENTRALIZED)
    portfolio_customers = {}
    
    # Add IN MARKET portfolio customers
    for _, portfolio in existing_inmarket.iterrows():
        port_code = portfolio['PORT_CODE']
        portfolio_customer_list = client_groups_df[
            (client_groups_df['CG_PORTFOLIO_CD'] == port_code) &
            (client_groups_df['ISPORTFOLIOACTIVE'] == 1)
        ]['CG_ECN'].tolist()
        portfolio_customers[port_code] = set(portfolio_customer_list)
    
    # Add CENTRALIZED portfolio customers
    for _, portfolio in existing_centralized.iterrows():
        port_code = portfolio['PORT_CODE']
        portfolio_customer_list = client_groups_df[
            (client_groups_df['CG_PORTFOLIO_CD'] == port_code) &
            (client_groups_df['ISPORTFOLIOACTIVE'] == 1)
        ]['CG_ECN'].tolist()
        portfolio_customers[port_code] = set(portfolio_customer_list)
    
    return existing_inmarket, existing_centralized, portfolio_customers

def calculate_customer_overlap(new_customers, existing_customers):
    """Calculate customer overlap between portfolios"""
    if not existing_customers:
        return 0
    new_customer_set = set(new_customers)
    overlap = len(new_customer_set.intersection(existing_customers))
    return overlap

def get_portfolio_financial_metrics(portfolio_customers, client_groups_df):
    """Get average financial metrics for a portfolio using customer list"""
    if not portfolio_customers:
        return None, None, None
    
    if isinstance(portfolio_customers, set):
        customer_list = list(portfolio_customers)
    else:
        customer_list = portfolio_customers
    
    portfolio_data = client_groups_df[client_groups_df['CG_ECN'].isin(customer_list)]
    
    if len(portfolio_data) == 0:
        return None, None, None
    
    avg_deposit = portfolio_data['DEPOSIT_BAL'].mean() if 'DEPOSIT_BAL' in portfolio_data.columns else None
    avg_gross_sales = portfolio_data['CG_GROSS_SALES'].mean() if 'CG_GROSS_SALES' in portfolio_data.columns else None
    avg_bank_revenue = portfolio_data['BANK_REVENUE'].mean() if 'BANK_REVENUE' in portfolio_data.columns else None
    
    return avg_deposit, avg_gross_sales, avg_bank_revenue

def get_portfolio_financial_metrics_by_portfolio_code(portfolio_code, client_groups_df):
    """Get average financial metrics for a portfolio using portfolio code from CLIENT_GROUPS_DF_NEW"""
    portfolio_data = client_groups_df[
        (client_groups_df['CG_PORTFOLIO_CD'] == portfolio_code) &
        (client_groups_df['ISPORTFOLIOACTIVE'] == 1)
    ]
    
    if len(portfolio_data) == 0:
        return None, None, None, 0
    
    avg_deposit = portfolio_data['DEPOSIT_BAL'].mean() if 'DEPOSIT_BAL' in portfolio_data.columns else None
    avg_gross_sales = portfolio_data['CG_GROSS_SALES'].mean() if 'CG_GROSS_SALES' in portfolio_data.columns else None
    avg_bank_revenue = portfolio_data['BANK_REVENUE'].mean() if 'BANK_REVENUE' in portfolio_data.columns else None
    portfolio_size = len(portfolio_data)
    
    return avg_deposit, avg_gross_sales, avg_bank_revenue, portfolio_size

def count_new_customers_not_in_active_portfolios(customer_au_assignments, client_groups_df):
    """Count customers in new portfolios that are not part of any active existing portfolio"""
    new_portfolio_customers = set(customer_au_assignments['CG_ECN'].tolist())
    active_portfolio_customers = set(client_groups_df[
        client_groups_df['ISPORTFOLIOACTIVE'] == 1
    ]['CG_ECN'].tolist())
    new_customers_not_in_active = new_portfolio_customers - active_portfolio_customers
    return len(new_customers_not_in_active)

def tag_new_portfolios_to_mojgan_portfolios(customer_au_assignments, active_portfolio_df, client_groups_df, branch_df):
    """Main function to tag new portfolios to existing Mojgan Madadi portfolios"""
    
    print("=== TAGGING NEW PORTFOLIOS TO MOJGAN MADADI PORTFOLIOS ===")
    
    # Calculate centroids for new portfolios
    new_portfolio_centroids = calculate_portfolio_centroids(customer_au_assignments)
    new_inmarket = new_portfolio_centroids[new_portfolio_centroids['TYPE'] == 'INMARKET']
    new_centralized = new_portfolio_centroids[new_portfolio_centroids['TYPE'] == 'CENTRALIZED']
    
    print(f"New portfolios: IN MARKET: {len(new_inmarket)}, CENTRALIZED: {len(new_centralized)}")
    
    # Get existing Mojgan portfolios
    existing_inmarket, existing_centralized, existing_portfolio_customers = get_existing_portfolios_mojgan(
        active_portfolio_df, client_groups_df, branch_df
    )
    
    print(f"Existing Mojgan portfolios: IN MARKET: {len(existing_inmarket)}, CENTRALIZED: {len(existing_centralized)}")
    
    all_tags = []
    
    # Tag IN MARKET portfolios by AU-to-AU distance
    if len(new_inmarket) > 0:
        # Get branch coordinates for new portfolios' AUs
        new_inmarket_with_coords = new_inmarket.merge(
            branch_df[['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
            left_on='ASSIGNED_AU', right_on='BRANCH_AU', how='left'
        )
        
        used_existing_inmarket = set()
        
        # Create all distance combinations
        inmarket_combinations = []
        for _, new_portfolio in new_inmarket_with_coords.iterrows():
            new_lat = new_portfolio['BRANCH_LAT_NUM']    # FROM: New AU branch location
            new_lon = new_portfolio['BRANCH_LON_NUM']    # FROM: New AU branch location
            new_au = new_portfolio['ASSIGNED_AU']
            
            for _, existing_portfolio in existing_inmarket.iterrows():
                existing_lat = existing_portfolio['BRANCH_LAT_NUM']  # TO: Existing AU branch location
                existing_lon = existing_portfolio['BRANCH_LON_NUM']  # TO: Existing AU branch location
                distance = haversine_distance_vectorized(new_lat, new_lon, existing_lat, existing_lon)
                
                inmarket_combinations.append({
                    'new_au': new_au,
                    'new_customer_count': new_portfolio['CUSTOMER_COUNT'],
                    'existing_portfolio': existing_portfolio['PORT_CODE'],
                    'existing_employee': existing_portfolio['EMPLOYEE_NAME'],
                    'existing_manager': existing_portfolio['MANAGER_NAME'],
                    'existing_director': existing_portfolio['DIRECTOR_NAME'],
                    'existing_au': existing_portfolio['AU'],
                    'distance': float(distance)
                })
        
        # Sort by distance and assign one-to-one
        inmarket_combinations_sorted = sorted(inmarket_combinations, key=lambda x: x['distance'])
        used_new_inmarket = set()
        
        for combo in inmarket_combinations_sorted:
            new_au = combo['new_au']
            existing_portfolio = combo['existing_portfolio']
            
            if new_au in used_new_inmarket or existing_portfolio in used_existing_inmarket:
                continue
            
            # Calculate customer overlap
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == 'INMARKET')
            ]['CG_ECN'].tolist()
            
            existing_customers = existing_portfolio_customers.get(existing_portfolio, set())
            overlap_count = calculate_customer_overlap(new_customers, existing_customers)
            
            # Get financial metrics using portfolio code for existing portfolio
            existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(existing_portfolio, client_groups_df)
            
            # Get financial metrics for new portfolio using customer list
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(new_customers, client_groups_df)
            
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
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
            })
            
            used_new_inmarket.add(new_au)
            used_existing_inmarket.add(existing_portfolio)
        
        # Handle untagged IN MARKET portfolios - assign to closest AU manager
        untagged_new_inmarket = new_inmarket_with_coords[
            ~new_inmarket_with_coords['ASSIGNED_AU'].isin(used_new_inmarket)
        ]
        
        for _, new_portfolio in untagged_new_inmarket.iterrows():
            new_au = new_portfolio['ASSIGNED_AU']
            new_lat = new_portfolio['BRANCH_LAT_NUM']    # FROM: New AU branch location
            new_lon = new_portfolio['BRANCH_LON_NUM']    # FROM: New AU branch location
            
            # Find closest AU manager from existing portfolios
            min_distance = float('inf')
            closest_manager_info = None
            
            for _, existing_portfolio in existing_inmarket.iterrows():
                existing_lat = existing_portfolio['BRANCH_LAT_NUM']  # TO: Existing AU branch location
                existing_lon = existing_portfolio['BRANCH_LON_NUM']  # TO: Existing AU branch location
                distance = haversine_distance_vectorized(new_lat, new_lon, existing_lat, existing_lon)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_manager_info = {
                        'portfolio': existing_portfolio['PORT_CODE'],
                        'employee': existing_portfolio['EMPLOYEE_NAME'],
                        'manager': existing_portfolio['MANAGER_NAME'],
                        'director': existing_portfolio['DIRECTOR_NAME'],
                        'au': existing_portfolio['AU'],
                        'distance': float(distance)
                    }
            
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == 'INMARKET')
            ]['CG_ECN'].tolist()
            
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(new_customers, client_groups_df)
            
            # If we found a closest manager, get their portfolio info, otherwise leave blank
            if closest_manager_info:
                existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(closest_manager_info['portfolio'], client_groups_df)
                
                # Calculate overlap with closest manager's portfolio
                existing_customers = existing_portfolio_customers.get(closest_manager_info['portfolio'], set())
                overlap_count = calculate_customer_overlap(new_customers, existing_customers)
                
                all_tags.append({
                    'NEW_AU': new_au,
                    'NEW_TYPE': 'IN MARKET',
                    'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                    'TAGGED_TO_PORTFOLIO': closest_manager_info['portfolio'],
                    'TAGGED_TO_EMPLOYEE': closest_manager_info['employee'],
                    'TAGGED_TO_MANAGER': closest_manager_info['manager'],
                    'TAGGED_TO_DIRECTOR': closest_manager_info['director'],
                    'TAGGED_TO_AU': closest_manager_info['au'],
                    'TAGGING_CRITERIA': 'CLOSEST_AU_MANAGER_UNTAGGED',
                    'DISTANCE_MILES': closest_manager_info['distance'],
                    'CUSTOMER_OVERLAP_COUNT': overlap_count,
                    'EXISTING_PORTFOLIO_SIZE': existing_portfolio_size,
                    'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
                    'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
                    'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
                    'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                    'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                    'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
                })
            else:
                # No existing portfolios found
                all_tags.append({
                    'NEW_AU': new_au,
                    'NEW_TYPE': 'IN MARKET',
                    'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                    'TAGGED_TO_PORTFOLIO': '',
                    'TAGGED_TO_EMPLOYEE': '',
                    'TAGGED_TO_MANAGER': '',
                    'TAGGED_TO_DIRECTOR': '',
                    'TAGGED_TO_AU': '',
                    'TAGGING_CRITERIA': 'UNTAGGED',
                    'DISTANCE_MILES': None,
                    'CUSTOMER_OVERLAP_COUNT': 0,
                    'EXISTING_PORTFOLIO_SIZE': 0,
                    'EXISTING_AVG_DEPOSIT_BAL': None,
                    'EXISTING_AVG_GROSS_SALES': None,
                    'EXISTING_AVG_BANK_REVENUE': None,
                    'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                    'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                    'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
                })
    
    # Tag CENTRALIZED portfolios by customer overlap
    if len(new_centralized) > 0:
        used_existing_centralized = set()
        
        # Create all overlap combinations
        centralized_combinations = []
        for _, new_portfolio in new_centralized.iterrows():
            new_au = new_portfolio['ASSIGNED_AU']
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == 'CENTRALIZED')
            ]['CG_ECN'].tolist()
            
            for _, existing_portfolio in existing_centralized.iterrows():
                existing_port_code = existing_portfolio['PORT_CODE']
                existing_customers = existing_portfolio_customers.get(existing_port_code, set())
                overlap_count = calculate_customer_overlap(new_customers, existing_customers)
                
                centralized_combinations.append({
                    'new_au': new_au,
                    'new_customer_count': new_portfolio['CUSTOMER_COUNT'],
                    'new_customers': new_customers,
                    'existing_portfolio': existing_port_code,
                    'existing_employee': existing_portfolio['EMPLOYEE_NAME'],
                    'existing_manager': existing_portfolio['MANAGER_NAME'],
                    'existing_director': existing_portfolio['DIRECTOR_NAME'],
                    'overlap_count': overlap_count,
                    'existing_customers': existing_customers
                })
        
        # Sort by overlap count and assign one-to-one
        centralized_combinations_sorted = sorted(centralized_combinations, key=lambda x: x['overlap_count'], reverse=True)
        used_new_centralized = set()
        
        for combo in centralized_combinations_sorted:
            new_au = combo['new_au']
            existing_portfolio = combo['existing_portfolio']
            
            if new_au in used_new_centralized or existing_portfolio in used_existing_centralized or combo['overlap_count'] == 0:
                continue
            
            # Get financial metrics using portfolio code for existing portfolio
            existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(existing_portfolio, client_groups_df)
            
            # Get financial metrics for new portfolio using customer list
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(combo['new_customers'], client_groups_df)
            
            all_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': 'CENTRALIZED',
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
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
            })
            
            used_new_centralized.add(new_au)
            used_existing_centralized.add(existing_portfolio)
        
        # Handle untagged CENTRALIZED portfolios - assign to manager with highest overlap
        for _, new_portfolio in new_centralized.iterrows():
            new_au = new_portfolio['ASSIGNED_AU']
            if new_au not in used_new_centralized:
                new_customers = customer_au_assignments[
                    (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                    (customer_au_assignments['TYPE'] == 'CENTRALIZED')
                ]['CG_ECN'].tolist()
                
                # Find manager with highest customer overlap
                max_overlap = 0
                best_manager_info = None
                
                for _, existing_portfolio in existing_centralized.iterrows():
                    existing_port_code = existing_portfolio['PORT_CODE']
                    existing_customers = existing_portfolio_customers.get(existing_port_code, set())
                    overlap_count = calculate_customer_overlap(new_customers, existing_customers)
                    
                    if overlap_count > max_overlap:
                        max_overlap = overlap_count
                        best_manager_info = {
                            'portfolio': existing_port_code,
                            'employee': existing_portfolio['EMPLOYEE_NAME'],
                            'manager': existing_portfolio['MANAGER_NAME'],
                            'director': existing_portfolio['DIRECTOR_NAME'],
                            'overlap': overlap_count
                        }
                
                new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(new_customers, client_groups_df)
                
                # If we found a manager with overlap, get their portfolio info, otherwise leave blank
                if best_manager_info and max_overlap > 0:
                    existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue, existing_portfolio_size = get_portfolio_financial_metrics_by_portfolio_code(best_manager_info['portfolio'], client_groups_df)
                    
                    all_tags.append({
                        'NEW_AU': new_au,
                        'NEW_TYPE': 'CENTRALIZED',
                        'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                        'TAGGED_TO_PORTFOLIO': best_manager_info['portfolio'],
                        'TAGGED_TO_EMPLOYEE': best_manager_info['employee'],
                        'TAGGED_TO_MANAGER': best_manager_info['manager'],
                        'TAGGED_TO_DIRECTOR': best_manager_info['director'],
                        'TAGGED_TO_AU': 'N/A (CENTRALIZED)',
                        'TAGGING_CRITERIA': 'BEST_OVERLAP_UNTAGGED',
                        'DISTANCE_MILES': None,
                        'CUSTOMER_OVERLAP_COUNT': best_manager_info['overlap'],
                        'EXISTING_PORTFOLIO_SIZE': existing_portfolio_size,
                        'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
                        'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
                        'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
                        'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                        'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                        'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
                    })
                else:
                    # No overlap found with any existing portfolio
                    all_tags.append({
                        'NEW_AU': new_au,
                        'NEW_TYPE': 'CENTRALIZED',
                        'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                        'TAGGED_TO_PORTFOLIO': '',
                        'TAGGED_TO_EMPLOYEE': '',
                        'TAGGED_TO_MANAGER': '',
                        'TAGGED_TO_DIRECTOR': '',
                        'TAGGED_TO_AU': '',
                        'TAGGING_CRITERIA': 'UNTAGGED',
                        'DISTANCE_MILES': None,
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
        print(f"Tagged to Mojgan portfolios: {len(tagged_portfolios)}")
        print(f"Untagged (left blank): {len(untagged_portfolios)}")
        
        if len(tagged_portfolios) > 0:
            print(f"Total customer overlap: {tagged_portfolios['CUSTOMER_OVERLAP_COUNT'].sum()}")
    
    return tagging_results

# Usage:
# tagging_results = tag_new_portfolios_to_mojgan_portfolios(customer_au_assignments, ACTIVE_PORTFOLIO, CLIENT_GROUPS_DF_NEW, branch_df)
# tagging_results.to_csv('portfolio_tagging_results.csv', index=False)
