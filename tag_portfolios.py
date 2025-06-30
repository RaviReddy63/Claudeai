import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_portfolio_centroids(customer_au_assignments):
    """Calculate centroid coordinates for each new portfolio"""
    portfolio_centroids = customer_au_assignments.groupby(['ASSIGNED_AU', 'TYPE']).agg({
        'LAT_NUM': 'mean',
        'LON_NUM': 'mean',
        'ECN': 'count'
    }).reset_index()
    
    portfolio_centroids.columns = ['ASSIGNED_AU', 'TYPE', 'CENTROID_LAT', 'CENTROID_LON', 'CUSTOMER_COUNT']
    return portfolio_centroids

def get_existing_inmarket_portfolios_with_coords(active_portfolio_df, client_groups_df, branch_df):
    """Get existing IN MARKET portfolios with their coordinates"""
    # Filter active IN MARKET portfolios
    existing_inmarket = active_portfolio_df[
        (active_portfolio_df['ISACTIVE'] == 1) & 
        (active_portfolio_df['ROLE_TYPE'] == 'IN MARKET') &
        (active_portfolio_df['ROLE_TYPE'].notna())
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
    """Get existing CENTRALIZED portfolios with their customer lists"""
    # Filter active CENTRALIZED portfolios
    existing_centralized = active_portfolio_df[
        (active_portfolio_df['ISACTIVE'] == 1) & 
        (active_portfolio_df['ROLE_TYPE'] == 'CENTRALIZED') &
        (active_portfolio_df['ROLE_TYPE'].notna())
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
        ]['ECN'].tolist()
        
        portfolio_customers[port_code] = set(portfolio_customer_list)
    
    return existing_centralized, portfolio_customers

def calculate_customer_overlap(new_customers, existing_customers):
    """Calculate the number of common customers between new and existing portfolios"""
    if not existing_customers:
        return 0
    
    new_customer_set = set(new_customers)
    existing_customer_set = existing_customers
    
    overlap = len(new_customer_set.intersection(existing_customer_set))
    return overlap

def tag_inmarket_portfolios_by_distance(new_inmarket_portfolios, existing_inmarket_portfolios):
    """Tag new IN MARKET portfolios to existing ones based on closest distance"""
    if len(existing_inmarket_portfolios) == 0:
        print("No existing IN MARKET portfolios found for tagging")
        return []
    
    inmarket_tags = []
    
    print(f"Tagging {len(new_inmarket_portfolios)} new IN MARKET portfolios...")
    
    for _, new_portfolio in new_inmarket_portfolios.iterrows():
        new_lat = new_portfolio['CENTROID_LAT']
        new_lon = new_portfolio['CENTROID_LON']
        new_au = new_portfolio['ASSIGNED_AU']
        
        # Calculate distances to all existing IN MARKET portfolios
        distances = []
        for _, existing_portfolio in existing_inmarket_portfolios.iterrows():
            existing_lat = existing_portfolio['BRANCH_LAT_NUM']
            existing_lon = existing_portfolio['BRANCH_LON_NUM']
            
            # Calculate haversine distance
            distance = haversine_distance_vectorized(
                new_lat, new_lon, existing_lat, existing_lon
            )
            
            distances.append({
                'existing_portfolio': existing_portfolio['PORT_CODE'],
                'existing_employee': existing_portfolio['EMPLOYEE_NAME'],
                'existing_manager': existing_portfolio['MANAGER_NAME'],
                'existing_director': existing_portfolio['DIRECTOR_NAME'],
                'existing_au': existing_portfolio['AU'],
                'distance': float(distance)
            })
        
        # Find closest existing portfolio
        if distances:
            closest = min(distances, key=lambda x: x['distance'])
            
            inmarket_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': 'IN MARKET',
                'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                'TAGGED_TO_PORTFOLIO': closest['existing_portfolio'],
                'TAGGED_TO_EMPLOYEE': closest['existing_employee'],
                'TAGGED_TO_MANAGER': closest['existing_manager'],
                'TAGGED_TO_DIRECTOR': closest['existing_director'],
                'TAGGED_TO_AU': closest['existing_au'],
                'TAGGING_CRITERIA': 'CLOSEST_DISTANCE',
                'DISTANCE_MILES': round(closest['distance'], 2)
            })
    
    return inmarket_tags

def tag_centralized_portfolios_by_overlap(new_centralized_portfolios, customer_au_assignments, 
                                        existing_centralized_portfolios, existing_portfolio_customers):
    """Tag new CENTRALIZED portfolios to existing ones based on customer overlap"""
    if len(existing_centralized_portfolios) == 0:
        print("No existing CENTRALIZED portfolios found for tagging")
        return []
    
    centralized_tags = []
    
    print(f"Tagging {len(new_centralized_portfolios)} new CENTRALIZED portfolios...")
    
    for _, new_portfolio in new_centralized_portfolios.iterrows():
        new_au = new_portfolio['ASSIGNED_AU']
        
        # Get customers in this new centralized portfolio
        new_customers = customer_au_assignments[
            (customer_au_assignments['ASSIGNED_AU'] == new_au) &
            (customer_au_assignments['TYPE'] == 'CENTRALIZED')
        ]['ECN'].tolist()
        
        # Calculate overlap with each existing centralized portfolio
        overlaps = []
        for _, existing_portfolio in existing_centralized_portfolios.iterrows():
            existing_port_code = existing_portfolio['PORT_CODE']
            existing_customers = existing_portfolio_customers.get(existing_port_code, set())
            
            overlap_count = calculate_customer_overlap(new_customers, existing_customers)
            
            if overlap_count > 0:  # Only consider portfolios with some overlap
                overlaps.append({
                    'existing_portfolio': existing_port_code,
                    'existing_employee': existing_portfolio['EMPLOYEE_NAME'],
                    'existing_manager': existing_portfolio['MANAGER_NAME'],
                    'existing_director': existing_portfolio['DIRECTOR_NAME'],
                    'overlap_count': overlap_count,
                    'existing_customer_count': len(existing_customers)
                })
        
        # Find portfolio with highest overlap
        if overlaps:
            best_match = max(overlaps, key=lambda x: x['overlap_count'])
            
            centralized_tags.append({
                'NEW_AU': new_au,
                'NEW_TYPE': 'CENTRALIZED',
                'NEW_CUSTOMER_COUNT': new_portfolio['CUSTOMER_COUNT'],
                'TAGGED_TO_PORTFOLIO': best_match['existing_portfolio'],
                'TAGGED_TO_EMPLOYEE': best_match['existing_employee'],
                'TAGGED_TO_MANAGER': best_match['existing_manager'],
                'TAGGED_TO_DIRECTOR': best_match['existing_director'],
                'TAGGED_TO_AU': 'N/A (CENTRALIZED)',
                'TAGGING_CRITERIA': 'MAX_CUSTOMER_OVERLAP',
                'CUSTOMER_OVERLAP_COUNT': best_match['overlap_count'],
                'EXISTING_PORTFOLIO_SIZE': best_match['existing_customer_count']
            })
        else:
            # No overlap found - mark as untagged
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
    Main function to tag new portfolios to existing ones
    
    Parameters:
    - customer_au_assignments: Result from create_customer_au_dataframe_with_centralized_clusters
    - active_portfolio_df: ACTIVE_PORTFOLIO dataframe
    - client_groups_df: CLIENT_GROUPS_DF_NEW dataframe
    - branch_df: Branch dataframe with coordinates
    
    Returns:
    - DataFrame with tagging results
    """
    
    print("=== STARTING PORTFOLIO TAGGING PROCESS ===")
    
    # Step 1: Calculate centroids for new portfolios
    print("Step 1: Calculating centroids for new portfolios...")
    new_portfolio_centroids = calculate_portfolio_centroids(customer_au_assignments)
    
    new_inmarket = new_portfolio_centroids[new_portfolio_centroids['TYPE'] == 'INMARKET']
    new_centralized = new_portfolio_centroids[new_portfolio_centroids['TYPE'] == 'CENTRALIZED']
    
    print(f"New portfolios created:")
    print(f"  - IN MARKET: {len(new_inmarket)}")
    print(f"  - CENTRALIZED: {len(new_centralized)}")
    
    # Step 2: Get existing portfolios
    print("\nStep 2: Loading existing portfolios...")
    
    # Get existing IN MARKET portfolios with coordinates
    existing_inmarket = get_existing_inmarket_portfolios_with_coords(
        active_portfolio_df, client_groups_df, branch_df
    )
    print(f"Existing IN MARKET portfolios: {len(existing_inmarket)}")
    
    # Get existing CENTRALIZED portfolios with customer lists
    existing_centralized, existing_portfolio_customers = get_existing_centralized_portfolios_with_customers(
        active_portfolio_df, client_groups_df
    )
    print(f"Existing CENTRALIZED portfolios: {len(existing_centralized)}")
    
    # Step 3: Tag IN MARKET portfolios by distance
    print("\nStep 3: Tagging IN MARKET portfolios by distance...")
    inmarket_tags = tag_inmarket_portfolios_by_distance(new_inmarket, existing_inmarket)
    
    # Step 4: Tag CENTRALIZED portfolios by customer overlap
    print("\nStep 4: Tagging CENTRALIZED portfolios by customer overlap...")
    centralized_tags = tag_centralized_portfolios_by_overlap(
        new_centralized, customer_au_assignments, existing_centralized, existing_portfolio_customers
    )
    
    # Step 5: Combine results
    all_tags = inmarket_tags + centralized_tags
    tagging_results = pd.DataFrame(all_tags)
    
    # Step 6: Print summary
    print("\n=== PORTFOLIO TAGGING SUMMARY ===")
    
    if len(tagging_results) > 0:
        print(f"Total new portfolios tagged: {len(tagging_results)}")
        
        # Summary by type
        type_summary = tagging_results.groupby(['NEW_TYPE']).agg({
            'NEW_AU': 'count',
            'NEW_CUSTOMER_COUNT': 'sum'
        })
        type_summary.columns = ['Portfolio_Count', 'Total_Customers']
        print("\nTagging Summary by Type:")
        print(type_summary)
        
        # IN MARKET distance statistics
        inmarket_results = tagging_results[tagging_results['NEW_TYPE'] == 'IN MARKET']
        if len(inmarket_results) > 0:
            print(f"\nIN MARKET Distance Statistics:")
            print(f"  Average distance to existing portfolio: {inmarket_results['DISTANCE_MILES'].mean():.2f} miles")
            print(f"  Max distance: {inmarket_results['DISTANCE_MILES'].max():.2f} miles")
            print(f"  Min distance: {inmarket_results['DISTANCE_MILES'].min():.2f} miles")
        
        # CENTRALIZED overlap statistics
        centralized_results = tagging_results[tagging_results['NEW_TYPE'] == 'CENTRALIZED']
        if len(centralized_results) > 0:
            tagged_centralized = centralized_results[centralized_results['CUSTOMER_OVERLAP_COUNT'] > 0]
            untagged_centralized = centralized_results[centralized_results['CUSTOMER_OVERLAP_COUNT'] == 0]
            
            print(f"\nCENTRALIZED Overlap Statistics:")
            print(f"  Successfully tagged: {len(tagged_centralized)}")
            print(f"  Untagged (no overlap): {len(untagged_centralized)}")
            
            if len(tagged_centralized) > 0:
                print(f"  Average customer overlap: {tagged_centralized['CUSTOMER_OVERLAP_COUNT'].mean():.1f}")
                print(f"  Max customer overlap: {tagged_centralized['CUSTOMER_OVERLAP_COUNT'].max()}")
        
        # Show tagging breakdown by existing portfolio managers
        print(f"\nTagging Results by Existing Portfolio Owner:")
        owner_summary = tagging_results.groupby(['TAGGED_TO_EMPLOYEE', 'NEW_TYPE']).agg({
            'NEW_AU': 'count',
            'NEW_CUSTOMER_COUNT': 'sum'
        }).reset_index()
        owner_summary.columns = ['Existing_Employee', 'Portfolio_Type', 'New_Portfolios_Count', 'Total_New_Customers']
        
        for _, row in owner_summary.iterrows():
            if pd.notna(row['Existing_Employee']):
                print(f"  {row['Existing_Employee']} ({row['Portfolio_Type']}): {row['New_Portfolios_Count']} portfolios, {row['Total_New_Customers']} customers")
    
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
