def calculate_customer_overlap(new_customers, existing_customers):
    """Calculate the number of common customers between new and existing portfolios"""
    if not existing_customers:
        return 0
    
    new_customer_set = set(new_customers)
    existing_customer_set = existing_customers
    
    overlap = len(new_customer_set.intersection(existing_customer_set))
    return overlap

def get_portfolio_financial_metrics(portfolio_customers, client_groups_df, portfolio_type="EXISTING"):
    """Get average financial metrics for a portfolio"""
    if not portfolio_customers:
        return None, None, None
    
    # Convert to list if it's a set
    if isinstance(portfolio_customers, set):
        customer_list = list(portfolio_customers)
    else:
        customer_list = portfolio_customers
    
    # Get financial data for these customers
    portfolio_data = client_groups_df[
        client_groups_df['CG_ECN'].isin(customer_list)
    ]
    
    if len(portfolio_data) == 0:
        return None, None, None
    
    avg_deposit = portfolio_data['DEPOSIT_BAL'].mean() if 'DEPOSIT_BAL' in portfolio_data.columns else None
    avg_gross_sales = portfolio_data['CG_GROSS_SALES'].mean() if 'CG_GROSS_SALES' in portfolio_data.columns else None
    avg_bank_revenue = portfolio_data['BANK_REVENUE'].mean() if 'BANK_REVENUE' in portfolio_data.columns else None
    
    return avg_deposit, avg_gross_sales, avg_bank_revenue

def count_new_customers_not_in_active_portfolios(customer_au_assignments, client_groups_df):
    """Count customers in new portfolios that are not part of any active existing portfolio"""
    # Get all customers in new portfolios
    new_portfolio_customers = set(customer_au_assignments['CG_ECN'].tolist())
    
    # Get all customers in active existing portfolios
    active_portfolio_customers = set(client_groups_df[
        client_groups_df['ISPORTFOLIOACTIVE'] == 1
    ]['CG_ECN'].tolist())
    
    # Count customers that are in new portfolios but not in any active portfolio
    new_customers_not_in_active = new_portfolio_customers - active_portfolio_customers
    
    return len(new_customers_not_in_active)import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_portfolio_centroids(customer_au_assignments):
    """Calculate centroid coordinates for each new portfolio"""
    portfolio_centroids = customer_au_assignments.groupby(['ASSIGNED_AU', 'TYPE']).agg({
        'LAT_NUM': 'mean',
        'LON_NUM': 'mean',
        'CG_ECN': 'count'
    }).reset_index()
    
    portfolio_centroids.columns = ['ASSIGNED_AU', 'TYPE', 'CENTROID_LAT', 'CENTROID_LON', 'CUSTOMER_COUNT']
    return portfolio_centroids

def get_existing_inmarket_portfolios_with_coords(active_portfolio_df, client_groups_df, branch_df):
    """Get existing IN MARKET portfolios with their coordinates - filtered by Mojgan Madadi"""
    # Filter active IN MARKET portfolios for Mojgan Madadi only
    existing_inmarket = active_portfolio_df[
        (active_portfolio_df['ISACTIVE'] == 1) & 
        (active_portfolio_df['ROLE_TYPE'] == 'IN MARKET') &
        (active_portfolio_df['ROLE_TYPE'].notna()) &
        (active_portfolio_df['DIRECTOR_NAME'] == 'Mojgan Madadi')
    ].copy()
    
    if len(existing_inmarket) == 0:
        return pd.DataFrame()
    
    # Get branch coordinates for these portfolios
    existing_with_coords = existing_inmarket.merge(
        branch_df[['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']], 
        left_on='AU', 
        right_on='BRANCH_AU', 
        how='left'
    )
    
    return existing_with_coords

def get_existing_centralized_portfolios_with_customers(active_portfolio_df, client_groups_df):
    """Get existing CENTRALIZED portfolios with their customer lists - filtered by Mojgan Madadi"""
    # Filter active CENTRALIZED portfolios for Mojgan Madadi only
    existing_centralized = active_portfolio_df[
        (active_portfolio_df['ISACTIVE'] == 1) & 
        (active_portfolio_df['ROLE_TYPE'] == 'CENTRALIZED') &
        (active_portfolio_df['ROLE_TYPE'].notna()) &
        (active_portfolio_df['DIRECTOR_NAME'] == 'Mojgan Madadi')
    ].copy()
    
    if len(existing_centralized) == 0:
        return pd.DataFrame(), {}
    
    # Get customers for each existing centralized portfolio
    portfolio_customers = {}
    
    for _, portfolio in existing_centralized.iterrows():
        port_code = portfolio['PORT_CODE']
        
        # Get customers for this portfolio
        portfolio_customer_list = client_groups_df[
            (client_groups_df['CG_PORTFOLIO_CD'] == port_code) &
            (client_groups_df['ISPORTFOLIOACTIVE'] == 1)
        ]['CG_ECN'].tolist()
        
        portfolio_customers[port_code] = set(portfolio_customer_list)
    
    return existing_centralized, portfolio_customers

def calculate_inmarket_customer_overlap(customer_au_assignments, new_au, existing_portfolio_customers, existing_port_code):
    """Calculate customer overlap for IN MARKET portfolios"""
    # Get customers in the new IN MARKET portfolio
    new_customers = customer_au_assignments[
        (customer_au_assignments['ASSIGNED_AU'] == new_au) &
        (customer_au_assignments['TYPE'] == 'INMARKET')
    ]['CG_ECN'].tolist()
    
    # Get existing portfolio customers
    existing_customers = existing_portfolio_customers.get(existing_port_code, set())
    
    # Calculate overlap
    new_customer_set = set(new_customers)
    overlap = len(new_customer_set.intersection(existing_customers))
    
    return overlap

def tag_inmarket_portfolios_by_distance(new_inmarket_portfolios, existing_inmarket_portfolios, 
                                       customer_au_assignments, existing_portfolio_customers, client_groups_df):
    """Tag new IN MARKET portfolios to existing ones based on closest distance - one-to-one mapping"""
    if len(existing_inmarket_portfolios) == 0:
        print("No existing IN MARKET portfolios under Mojgan Madadi found for tagging")
        return []
    
    inmarket_tags = []
    used_existing_portfolios = set()  # Track which existing portfolios are already used
    
    print(f"Tagging {len(new_inmarket_portfolios)} new IN MARKET portfolios...")
    
    # Create list of (new_portfolio, distance, existing_portfolio) for all combinations
    all_combinations = []
    
    for new_idx, new_portfolio in new_inmarket_portfolios.iterrows():
        new_lat = new_portfolio['CENTROID_LAT']
        new_lon = new_portfolio['CENTROID_LON']
        new_au = new_portfolio['ASSIGNED_AU']
        
        for existing_idx, existing_portfolio in existing_inmarket_portfolios.iterrows():
            existing_lat = existing_portfolio['BRANCH_LAT_NUM']
            existing_lon = existing_portfolio['BRANCH_LON_NUM']
            
            # Calculate haversine distance
            distance = haversine_distance_vectorized(
                new_lat, new_lon, existing_lat, existing_lon
            )
            
            all_combinations.append({
                'new_au': new_au,
                'new_idx': new_idx,
                'new_customer_count': new_portfolio['CUSTOMER_COUNT'],
                'existing_portfolio': existing_portfolio['PORT_CODE'],
                'existing_employee': existing_portfolio['EMPLOYEE_NAME'],
                'existing_manager': existing_portfolio['MANAGER_NAME'],
                'existing_director': existing_portfolio['DIRECTOR_NAME'],
                'existing_au': existing_portfolio['AU'],
                'distance': float(distance)
            })
    
    # Sort all combinations by distance (closest first)
    all_combinations_sorted = sorted(all_combinations, key=lambda x: x['distance'])
    
    # Assign one-to-one: each new portfolio gets assigned to closest available existing portfolio
    used_new_portfolios = set()
    
    for combo in all_combinations_sorted:
        new_au = combo['new_au']
        existing_portfolio = combo['existing_portfolio']
        
        # Skip if this new portfolio or existing portfolio is already used
        if new_au in used_new_portfolios or existing_portfolio in used_existing_portfolios:
            continue
        
        # Calculate customer overlap for IN MARKET
        overlap_count = calculate_inmarket_customer_overlap(
            customer_au_assignments, new_au, existing_portfolio_customers, existing_portfolio
        )
        
        # Get financial metrics for new portfolio
        new_customers = customer_au_assignments[
            (customer_au_assignments['ASSIGNED_AU'] == new_au) &
            (customer_au_assignments['TYPE'] == 'INMARKET')
        ]['CG_ECN'].tolist()
        
        new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(
            new_customers, client_groups_df, "NEW"
        )
        
        # Get financial metrics for existing portfolio
        existing_customers = existing_portfolio_customers.get(existing_portfolio, set())
        existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue = get_portfolio_financial_metrics(
            existing_customers, client_groups_df, "EXISTING"
        )
        
        # Make the assignment
        inmarket_tags.append({
            'NEW_AU': new_au,
            'NEW_TYPE': 'IN MARKET',
            'NEW_CUSTOMER_COUNT': combo['new_customer_count'],
            'TAGGED_TO_PORTFOLIO': existing_portfolio,
            'TAGGED_TO_EMPLOYEE': combo['existing_employee'],
            'TAGGED_TO_MANAGER': combo['existing_manager'],
            'TAGGED_TO_DIRECTOR': combo['existing_director'],
            'TAGGED_TO_AU': combo['existing_au'],
            'TAGGING_CRITERIA': 'CLOSEST_DISTANCE',
            'DISTANCE_MILES': combo['distance'],
            'CUSTOMER_OVERLAP_COUNT': overlap_count,
            'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
            'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
            'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
            'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
            'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
            'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
        })
        
        # Mark both as used
        used_new_portfolios.add(new_au)
        used_existing_portfolios.add(existing_portfolio)
        
        # Stop if all new portfolios are assigned
        if len(used_new_portfolios) == len(new_inmarket_portfolios):
            break
    
    # Handle any unassigned new portfolios (if there are more new than existing or no Mojgan portfolios)
    for new_idx, new_portfolio in new_inmarket_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        if new_au not in used_new_portfolios:
            # Get financial metrics for untagged new portfolio
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == 'INMARKET')
            ]['CG_ECN'].tolist()
            
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(
                new_customers, client_groups_df, "NEW"
            )
            
            inmarket_tags.append({
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
                'EXISTING_AVG_DEPOSIT_BAL': None,
                'EXISTING_AVG_GROSS_SALES': None,
                'EXISTING_AVG_BANK_REVENUE': None,
                'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
            })
    
    return inmarket_tags

def tag_centralized_portfolios_by_overlap(new_centralized_portfolios, customer_au_assignments, 
                                        existing_centralized_portfolios, existing_portfolio_customers, client_groups_df):
    """Tag new CENTRALIZED portfolios to existing ones based on customer overlap - one-to-one mapping"""
    if len(existing_centralized_portfolios) == 0:
        print("No existing CENTRALIZED portfolios under Mojgan Madadi found for tagging")
        return []
    
    centralized_tags = []
    used_existing_portfolios = set()  # Track which existing portfolios are already used
    
    print(f"Tagging {len(new_centralized_portfolios)} new CENTRALIZED portfolios...")
    
    # Create list of (new_portfolio, overlap, existing_portfolio) for all combinations
    all_combinations = []
    
    for new_idx, new_portfolio in new_centralized_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        
        # Get customers in this new centralized portfolio
        new_customers = customer_au_assignments[
            (customer_au_assignments['ASSIGNED_AU'] == new_au) &
            (customer_au_assignments['TYPE'] == 'CENTRALIZED')
        ]['CG_ECN'].tolist()
        
        for existing_idx, existing_portfolio in existing_centralized_portfolios.iterrows():
            existing_port_code = existing_portfolio['PORT_CODE']
            existing_customers = existing_portfolio_customers.get(existing_port_code, set())
            
            overlap_count = calculate_customer_overlap(new_customers, existing_customers)
            
            all_combinations.append({
                'new_au': new_au,
                'new_idx': new_idx,
                'new_customer_count': new_portfolio['CUSTOMER_COUNT'],
                'new_customers': new_customers,
                'existing_portfolio': existing_port_code,
                'existing_employee': existing_portfolio['EMPLOYEE_NAME'],
                'existing_manager': existing_portfolio['MANAGER_NAME'],
                'existing_director': existing_portfolio['DIRECTOR_NAME'],
                'overlap_count': overlap_count,
                'existing_customer_count': len(existing_customers),
                'existing_customers': existing_customers
            })
    
    # Sort all combinations by overlap count (highest first), then by existing portfolio size (largest first)
    all_combinations_sorted = sorted(all_combinations, 
                                   key=lambda x: (x['overlap_count'], x['existing_customer_count']), 
                                   reverse=True)
    
    # Assign one-to-one: each new portfolio gets assigned to best available existing portfolio
    used_new_portfolios = set()
    
    for combo in all_combinations_sorted:
        new_au = combo['new_au']
        existing_portfolio = combo['existing_portfolio']
        
        # Skip if this new portfolio or existing portfolio is already used
        if new_au in used_new_portfolios or existing_portfolio in used_existing_portfolios:
            continue
        
        # Only assign if there's some overlap
        if combo['overlap_count'] > 0:
            # Get financial metrics for new portfolio
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(
                combo['new_customers'], client_groups_df, "NEW"
            )
            
            # Get financial metrics for existing portfolio
            existing_avg_deposit, existing_avg_gross_sales, existing_avg_bank_revenue = get_portfolio_financial_metrics(
                combo['existing_customers'], client_groups_df, "EXISTING"
            )
            
            centralized_tags.append({
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
                'EXISTING_PORTFOLIO_SIZE': combo['existing_customer_count'],
                'EXISTING_AVG_DEPOSIT_BAL': existing_avg_deposit,
                'EXISTING_AVG_GROSS_SALES': existing_avg_gross_sales,
                'EXISTING_AVG_BANK_REVENUE': existing_avg_bank_revenue,
                'NEW_AVG_DEPOSIT_BAL': new_avg_deposit,
                'NEW_AVG_GROSS_SALES': new_avg_gross_sales,
                'NEW_AVG_BANK_REVENUE': new_avg_bank_revenue
            })
            
            # Mark both as used
            used_new_portfolios.add(new_au)
            used_existing_portfolios.add(existing_portfolio)
        
        # Stop if all new portfolios are assigned
        if len(used_new_portfolios) == len(new_centralized_portfolios):
            break
    
    # Handle any unassigned new portfolios
    for new_idx, new_portfolio in new_centralized_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        if new_au not in used_new_portfolios:
            # Get financial metrics for untagged new portfolio
            new_customers = customer_au_assignments[
                (customer_au_assignments['ASSIGNED_AU'] == new_au) &
                (customer_au_assignments['TYPE'] == 'CENTRALIZED')
            ]['CG_ECN'].tolist()
            
            new_avg_deposit, new_avg_gross_sales, new_avg_bank_revenue = get_portfolio_financial_metrics(
                new_customers, client_groups_df, "NEW"
            )
            
            centralized_tags.append({
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
    
    return centralized_tagsAU': new_au,
                'NEW_TYPE': 'CENTRALIZED',
                'NEW_CUSTOMER_COUNT': combo['new_customer_count'],
                'TAGGED_TO_PORTFOLIO': existing_portfolio,
                'TAGGED_TO_EMPLOYEE': combo['existing_employee'],
                'TAGGED_TO_MANAGER': combo['existing_manager'],
                'TAGGED_TO_DIRECTOR': combo['existing_director'],
                'TAGGED_TO_AU': 'N/A (CENTRALIZED)',
                'TAGGING_CRITERIA': 'MAX_CUSTOMER_OVERLAP',
                'CUSTOMER_OVERLAP_COUNT': combo['overlap_count'],
                'EXISTING_PORTFOLIO_SIZE': combo['existing_customer_count']
            })
            
            # Mark both as used
            used_new_portfolios.add(new_au)
            used_existing_portfolios.add(existing_portfolio)
        
        # Stop if all new portfolios are assigned
        if len(used_new_portfolios) == len(new_centralized_portfolios):
            break
    
    # Handle any unassigned new portfolios
    for new_idx, new_portfolio in new_centralized_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        if new_au not in used_new_portfolios:
            centralized_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': 'CENTRALIZED',
                'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                'TAGGED_TO_PORTFOLIO': 'UNTAGGED - NO OVERLAP',
                'TAGGED_TO_EMPLOYEE': None,
                'TAGGED_TO_MANAGER': None,
                'TAGGED_TO_DIRECTOR': None,
                'TAGGED_TO_AU': 'N/A (CENTRALIZED)',
                'TAGGING_CRITERIA': 'NO_OVERLAP_FOUND',
                'CUSTOMER_OVERLAP_COUNT': 0,
                'EXISTING_PORTFOLIO_SIZE': 0
            })
    
    return centralized_tags

def tag_new_portfolios_to_existing(customer_au_assignments, active_portfolio_df, 
                                 client_groups_df, branch_df):
    """
    Main function to tag new portfolios to existing ones under Mojgan Madadi
    
    Parameters:
    - customer_au_assignments: Result from create_customer_au_dataframe_with_centralized_clusters
    - active_portfolio_df: ACTIVE_PORTFOLIO dataframe
    - client_groups_df: CLIENT_GROUPS_DF_NEW dataframe
    - branch_df: Branch dataframe with coordinates
    
    Returns:
    - DataFrame with tagging results
    """
    
    print("=== STARTING PORTFOLIO TAGGING PROCESS (Mojgan Madadi Only) ===")
    
    # Step 1: Calculate centroids for new portfolios
    print("Step 1: Calculating centroids for new portfolios...")
    new_portfolio_centroids = calculate_portfolio_centroids(customer_au_assignments)
    
    new_inmarket = new_portfolio_centroids[new_portfolio_centroids['TYPE'] == 'INMARKET']
    new_centralized = new_portfolio_centroids[new_portfolio_centroids['TYPE'] == 'CENTRALIZED']
    
    print(f"New portfolios created:")
    print(f"  - IN MARKET: {len(new_inmarket)}")
    print(f"  - CENTRALIZED: {len(new_centralized)}")
    
    # Step 2: Get existing portfolios under Mojgan Madadi
    print("\nStep 2: Loading existing portfolios under Mojgan Madadi...")
    
    # Get existing IN MARKET portfolios with coordinates
    existing_inmarket = get_existing_inmarket_portfolios_with_coords(
        active_portfolio_df, client_groups_df, branch_df
    )
    print(f"Existing IN MARKET portfolios under Mojgan Madadi: {len(existing_inmarket)}")
    
    # Get existing CENTRALIZED portfolios with customer lists
    existing_centralized, existing_portfolio_customers = get_existing_centralized_portfolios_with_customers(
        active_portfolio_df, client_groups_df
    )
    print(f"Existing CENTRALIZED portfolios under Mojgan Madadi: {len(existing_centralized)}")
    
    # Step 3: Tag IN MARKET portfolios by distance
    print("\nStep 3: Tagging IN MARKET portfolios by distance...")
    inmarket_tags = tag_inmarket_portfolios_by_distance(
        new_inmarket, existing_inmarket, customer_au_assignments, 
        existing_portfolio_customers, client_groups_df
    )
    
    # Step 4: Tag CENTRALIZED portfolios by customer overlap
    print("\nStep 4: Tagging CENTRALIZED portfolios by customer overlap...")
    centralized_tags = tag_centralized_portfolios_by_overlap(
        new_centralized, customer_au_assignments, existing_centralized, 
        existing_portfolio_customers, client_groups_df
    )
    
    # Step 5: Count new customers not in any active portfolio
    print("\nStep 5: Calculating new customers not in active portfolios...")
    new_customers_count = count_new_customers_not_in_active_portfolios(
        customer_au_assignments, client_groups_df
    )
    
    # Step 6: Combine results
    all_tags = inmarket_tags + centralized_tags
    tagging_results = pd.DataFrame(all_tags)
    
    # Step 7: Print summary
    print("\n=== PORTFOLIO TAGGING SUMMARY ===")
    print(f"New customers not in any active portfolio: {new_customers_count}")
    
    if len(tagging_results) > 0:
        print(f"Total new portfolios: {len(tagging_results)}")
        
        # Count tagged vs untagged
        tagged_portfolios = tagging_results[tagging_results['TAGGED_TO_PORTFOLIO'] != '']
        untagged_portfolios = tagging_results[tagging_results['TAGGED_TO_PORTFOLIO'] == '']
        
        print(f"Successfully tagged to Mojgan Madadi portfolios: {len(tagged_portfolios)}")
        print(f"Untagged (left blank): {len(untagged_portfolios)}")
        
        # Summary by type
        type_summary = tagging_results.groupby(['NEW_TYPE']).agg({
            'NEW_AU': 'count',
            'NEW_CUSTOMER_COUNT': 'sum'
        })
        type_summary.columns = ['Portfolio_Count', 'Total_Customers']
        print("\nPortfolio Count by Type:")
        print(type_summary)
        
        # IN MARKET statistics
        inmarket_results = tagging_results[tagging_results['NEW_TYPE'] == 'IN MARKET']
        if len(inmarket_results) > 0:
            tagged_inmarket = inmarket_results[inmarket_results['TAGGED_TO_PORTFOLIO'] != '']
            if len(tagged_inmarket) > 0:
                print(f"\nIN MARKET Tagged Portfolio Statistics:")
                print(f"  Average distance: {tagged_inmarket['DISTANCE_MILES'].mean():.2f} miles")
                print(f"  Average customer overlap: {tagged_inmarket['CUSTOMER_OVERLAP_COUNT'].mean():.1f}")
                print(f"  Total customer overlap: {tagged_inmarket['CUSTOMER_OVERLAP_COUNT'].sum()}")
        
        # CENTRALIZED statistics
        centralized_results = tagging_results[tagging_results['NEW_TYPE'] == 'CENTRALIZED']
        if len(centralized_results) > 0:
            tagged_centralized = centralized_results[centralized_results['TAGGED_TO_PORTFOLIO'] != '']
            if len(tagged_centralized) > 0:
                print(f"\nCENTRALIZED Tagged Portfolio Statistics:")
                print(f"  Average customer overlap: {tagged_centralized['CUSTOMER_OVERLAP_COUNT'].mean():.1f}")
                print(f"  Total customer overlap: {tagged_centralized['CUSTOMER_OVERLAP_COUNT'].sum()}")
        
        # Financial metrics summary for tagged portfolios
        if len(tagged_portfolios) > 0:
            print(f"\nFinancial Metrics Summary (Tagged Portfolios):")
            
            # Existing portfolio metrics
            existing_metrics = tagged_portfolios.dropna(subset=['EXISTING_AVG_DEPOSIT_BAL'])
            if len(existing_metrics) > 0:
                print(f"  Existing Portfolios Average:")
                print(f"    Deposit Balance: ${existing_metrics['EXISTING_AVG_DEPOSIT_BAL'].mean():,.2f}")
                print(f"    Gross Sales: ${existing_metrics['EXISTING_AVG_GROSS_SALES'].mean():,.2f}")
                print(f"    Bank Revenue: ${existing_metrics['EXISTING_AVG_BANK_REVENUE'].mean():,.2f}")
            
            # New portfolio metrics
            new_metrics = tagged_portfolios.dropna(subset=['NEW_AVG_DEPOSIT_BAL'])
            if len(new_metrics) > 0:
                print(f"  New Portfolios Average:")
                print(f"    Deposit Balance: ${new_metrics['NEW_AVG_DEPOSIT_BAL'].mean():,.2f}")
                print(f"    Gross Sales: ${new_metrics['NEW_AVG_GROSS_SALES'].mean():,.2f}")
                print(f"    Bank Revenue: ${new_metrics['NEW_AVG_BANK_REVENUE'].mean():,.2f}")
        
        # Show which existing portfolios got new assignments
        if len(tagged_portfolios) > 0:
            print(f"\nExisting Portfolio Assignments:")
            assignment_summary = tagged_portfolios.groupby(['TAGGED_TO_EMPLOYEE', 'NEW_TYPE']).agg({
                'NEW_AU': 'count',
                'NEW_CUSTOMER_COUNT': 'sum'
            }).reset_index()
            assignment_summary.columns = ['Employee', 'Portfolio_Type', 'New_Portfolios', 'New_Customers']
            
            for _, row in assignment_summary.iterrows():
                print(f"  {row['Employee']} ({row['Portfolio_Type']}): {row['New_Portfolios']} portfolios, {row['New_Customers']} customers")
    
    else:
        print("No portfolios were created for tagging")
    
    return tagging_results

# Example usage:
# tagging_results = tag_new_portfolios_to_existing(
#     customer_au_assignments, 
#     active_portfolio_df, 
#     client_groups_df, 
#     branch_df
# )
# 
# # Save results
# tagging_results.to_csv('portfolio_tagging_results.csv', index=False)
# print("\nTagging results saved to 'portfolio_tagging_results.csv'")

# You can also export detailed customer assignments with tagging info
def create_detailed_customer_assignments_with_tags(customer_au_assignments, tagging_results):
    """Create detailed customer assignments with tagging information"""
    
    # Create a mapping from new portfolio to existing portfolio info
    tag_mapping = {}
    for _, tag in tagging_results.iterrows():
        key = (tag['NEW_AU'], tag['NEW_TYPE'])
        tag_mapping[key] = {
            'TAGGED_TO_PORTFOLIO': tag['TAGGED_TO_PORTFOLIO'],
            'TAGGED_TO_EMPLOYEE': tag['TAGGED_TO_EMPLOYEE'],
            'TAGGED_TO_MANAGER': tag['TAGGED_TO_MANAGER'],
            'TAGGED_TO_DIRECTOR': tag['TAGGED_TO_DIRECTOR'],
            'TAGGING_CRITERIA': tag['TAGGING_CRITERIA']
        }
    
    # Add tagging info to customer assignments
    detailed_assignments = customer_au_assignments.copy()
    
    # Add tagging columns
    detailed_assignments['TAGGED_TO_PORTFOLIO'] = None
    detailed_assignments['TAGGED_TO_EMPLOYEE'] = None
    detailed_assignments['TAGGED_TO_MANAGER'] = None
    detailed_assignments['TAGGED_TO_DIRECTOR'] = None
    detailed_assignments['TAGGING_CRITERIA'] = None
    
    # Fill in tagging information
    for idx, row in detailed_assignments.iterrows():
        key = (row['ASSIGNED_AU'], row['TYPE'])
        if key in tag_mapping:
            tag_info = tag_mapping[key]
            detailed_assignments.loc[idx, 'TAGGED_TO_PORTFOLIO'] = tag_info['TAGGED_TO_PORTFOLIO']
            detailed_assignments.loc[idx, 'TAGGED_TO_EMPLOYEE'] = tag_info['TAGGED_TO_EMPLOYEE']
            detailed_assignments.loc[idx, 'TAGGED_TO_MANAGER'] = tag_info['TAGGED_TO_MANAGER']
            detailed_assignments.loc[idx, 'TAGGED_TO_DIRECTOR'] = tag_info['TAGGED_TO_DIRECTOR']
            detailed_assignments.loc[idx, 'TAGGING_CRITERIA'] = tag_info['TAGGING_CRITERIA']
    
    return detailed_assignments

# Example usage for detailed export:
# detailed_assignments = create_detailed_customer_assignments_with_tags(
#     customer_au_assignments, tagging_results
# )
# detailed_assignments.to_csv('detailed_customer_assignments_with_tagging.csv', index=False)
