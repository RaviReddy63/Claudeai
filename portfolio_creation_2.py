import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in miles between two points"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 3959  # Earth radius in miles

def find_customers_within_radius(customer_df, branch_df, max_radius=20):
    """Find customers within radius for each branch and determine minimum radius needed"""
    eligible_branches = []
    
    for _, branch in branch_df.iterrows():
        # Calculate distances to all customers
        distances = []
        customer_indices = []
        
        for idx, customer in customer_df.iterrows():
            dist = haversine_distance(
                branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM'],
                customer['LAT_NUM'], customer['LON_NUM']
            )
            distances.append(dist)
            customer_indices.append(idx)
        
        # Sort by distance
        sorted_data = sorted(zip(distances, customer_indices))
        
        # Find minimum radius that gives at least 200 customers
        min_radius = None
        customers_in_radius = []
        
        for i, (dist, cust_idx) in enumerate(sorted_data):
            if dist <= max_radius:
                customers_in_radius.append((cust_idx, dist))
                if len(customers_in_radius) >= 200:
                    min_radius = dist
                    break
        
        # If we have at least 200 customers within max_radius
        if len(customers_in_radius) >= 200:
            eligible_branches.append({
                'branch_id': branch['BRANCH_AU'],
                'branch_lat': branch['BRANCH_LAT_NUM'],
                'branch_lon': branch['BRANCH_LON_NUM'],
                'min_radius': min_radius,
                'customers_in_radius': customers_in_radius,
                'total_customers': len(customers_in_radius)
            })
    
    return eligible_branches

def create_portfolios(eligible_branches, customer_df, max_portfolio_size=280):
    """Create customer portfolios for each eligible branch"""
    portfolios = []
    
    for branch in eligible_branches:
        customers_data = branch['customers_in_radius']
        
        # If customers <= max_portfolio_size, create single portfolio
        if len(customers_data) <= max_portfolio_size:
            customer_list = [cust_idx for cust_idx, _ in customers_data]
            portfolios.append({
                'branch_id': branch['branch_id'],
                'branch_lat': branch['branch_lat'],
                'branch_lon': branch['branch_lon'],
                'portfolio_id': f"{branch['branch_id']}_P1",
                'customers': customer_list,
                'customer_count': len(customer_list),
                'avg_distance': np.mean([dist for _, dist in customers_data])
            })
        else:
            # Multiple portfolios needed - use clustering
            n_portfolios = (len(customers_data) + max_portfolio_size - 1) // max_portfolio_size
            
            # Get customer coordinates for clustering
            customer_coords = []
            customer_indices = []
            for cust_idx, _ in customers_data:
                customer_coords.append([
                    customer_df.loc[cust_idx, 'LAT_NUM'],
                    customer_df.loc[cust_idx, 'LON_NUM']
                ])
                customer_indices.append(cust_idx)
            
            # Cluster customers
            kmeans = KMeans(n_clusters=n_portfolios, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(customer_coords)
            
            # Create portfolios from clusters
            for i in range(n_portfolios):
                cluster_customers = [customer_indices[j] for j in range(len(clusters)) if clusters[j] == i]
                
                # Calculate average distance for this portfolio
                avg_dist = np.mean([
                    haversine_distance(
                        branch['branch_lat'], branch['branch_lon'],
                        customer_df.loc[cust_idx, 'LAT_NUM'], 
                        customer_df.loc[cust_idx, 'LON_NUM']
                    ) for cust_idx in cluster_customers
                ])
                
                portfolios.append({
                    'branch_id': branch['branch_id'],
                    'branch_lat': branch['branch_lat'],
                    'branch_lon': branch['branch_lon'],
                    'portfolio_id': f"{branch['branch_id']}_P{i+1}",
                    'customers': cluster_customers,
                    'customer_count': len(cluster_customers),
                    'avg_distance': avg_dist
                })
    
    return portfolios

def resolve_overlapping_assignments(portfolios, customer_df):
    """Ensure each customer is assigned to only one portfolio (closest branch)"""
    customer_assignments = {}
    
    # Collect all potential assignments
    for portfolio in portfolios:
        for cust_idx in portfolio['customers']:
            if cust_idx not in customer_assignments:
                customer_assignments[cust_idx] = []
            
            distance = haversine_distance(
                portfolio['branch_lat'], portfolio['branch_lon'],
                customer_df.loc[cust_idx, 'LAT_NUM'], 
                customer_df.loc[cust_idx, 'LON_NUM']
            )
            
            customer_assignments[cust_idx].append({
                'portfolio_id': portfolio['portfolio_id'],
                'branch_id': portfolio['branch_id'],
                'distance': distance
            })
    
    # Assign each customer to closest branch only
    final_assignments = {}
    for cust_idx, options in customer_assignments.items():
        best_option = min(options, key=lambda x: x['distance'])
        final_assignments[cust_idx] = best_option
    
    # Update portfolios to remove duplicates
    updated_portfolios = []
    for portfolio in portfolios:
        assigned_customers = [
            cust_idx for cust_idx in portfolio['customers']
            if final_assignments.get(cust_idx, {}).get('portfolio_id') == portfolio['portfolio_id']
        ]
        
        if assigned_customers:  # Only keep portfolios with customers
            updated_portfolio = {
                **portfolio,
                'customers': assigned_customers,
                'customer_count': len(assigned_customers)
            }
            # Recalculate average distance
            updated_portfolio['avg_distance'] = np.mean([
                haversine_distance(
                    portfolio['branch_lat'], portfolio['branch_lon'],
                    customer_df.loc[cust_idx, 'LAT_NUM'], 
                    customer_df.loc[cust_idx, 'LON_NUM']
                ) for cust_idx in assigned_customers
            ])
            updated_portfolios.append(updated_portfolio)
    
    return updated_portfolios

# Main execution
print("Analyzing branches and customers...")

# Find eligible branches
eligible_branches = find_customers_within_radius(customer_df, branch_df)

print(f"\nFound {len(eligible_branches)} eligible branches:")
for branch in eligible_branches:
    print(f"Branch {branch['branch_id']}: {branch['total_customers']} customers within {branch['min_radius']:.1f} miles")

# Create portfolios
portfolios = create_portfolios(eligible_branches, customer_df)

print(f"\nBEFORE resolving overlaps: {len(portfolios)} portfolios")

# Resolve overlapping assignments to ensure each customer is in only one portfolio
portfolios = resolve_overlapping_assignments(portfolios, customer_df)

print(f"AFTER resolving overlaps: {len(portfolios)} portfolios")

print(f"\nFinal portfolios:")
for portfolio in portfolios:
    print(f"Portfolio {portfolio['portfolio_id']}: {portfolio['customer_count']} customers, avg distance: {portfolio['avg_distance']:.1f} miles")

# Create summary DataFrames
branch_summary = pd.DataFrame([
    {
        'Branch_ID': branch['branch_id'],
        'Branch_Lat': branch['branch_lat'],
        'Branch_Lon': branch['branch_lon'],
        'Min_Radius_Miles': round(branch['min_radius'], 2),
        'Total_Customers_In_Radius': branch['total_customers'],
        'Final_Portfolios': len([p for p in portfolios if p['branch_id'] == branch['branch_id']])
    }
    for branch in eligible_branches
])

portfolio_summary = pd.DataFrame([
    {
        'Portfolio_ID': p['portfolio_id'],
        'Branch_ID': p['branch_id'],
        'Customer_Count': p['customer_count'],
        'Avg_Distance_Miles': round(p['avg_distance'], 2)
    }
    for p in portfolios
])

# Create detailed customer assignments
customer_assignments = []
for portfolio in portfolios:
    for cust_idx in portfolio['customers']:
        distance = haversine_distance(
            portfolio['branch_lat'], portfolio['branch_lon'],
            customer_df.loc[cust_idx, 'LAT_NUM'], 
            customer_df.loc[cust_idx, 'LON_NUM']
        )
        customer_assignments.append({
            'ECN': customer_df.loc[cust_idx, 'ECN'],
            'Customer_Lat': customer_df.loc[cust_idx, 'LAT_NUM'],
            'Customer_Lon': customer_df.loc[cust_idx, 'LON_NUM'],
            'Billing_Street': customer_df.loc[cust_idx, 'BILLINGSTREET'],
            'Billing_City': customer_df.loc[cust_idx, 'BILLINGCITY'],
            'Billing_State': customer_df.loc[cust_idx, 'BILLINGSTATE'],
            'Portfolio_ID': portfolio['portfolio_id'],
            'Branch_ID': portfolio['branch_id'],
            'Distance_Miles': round(distance, 2)
        })

customer_assignment_df = pd.DataFrame(customer_assignments)

# Validation: Check for duplicate customer assignments
duplicate_customers = customer_assignment_df.groupby('ECN').size()
duplicates = duplicate_customers[duplicate_customers > 1]

print(f"\n=== VALIDATION ===")
print(f"Total unique customers assigned: {customer_assignment_df['ECN'].nunique()}")
print(f"Total assignment records: {len(customer_assignment_df)}")
print(f"Duplicate assignments: {len(duplicates)}")

if len(duplicates) > 0:
    print("WARNING: Found duplicate customer assignments!")
    print(duplicates.head())
else:
    print("✓ No duplicate assignments - each customer in exactly one portfolio")

print(f"\n=== SUMMARY ===")
print(f"Eligible branches: {len(eligible_branches)}")
print(f"Total portfolios: {len(portfolios)}")
print(f"Total customers assigned: {len(customer_assignments)}")
print(f"Bankers to hire: {len(portfolios)}")

# Display results
print("\n=== BRANCH SUMMARY ===")
print(branch_summary.to_string(index=False))

print("\n=== PORTFOLIO SUMMARY ===")
print(portfolio_summary.to_string(index=False))

print("\n=== SAMPLE CUSTOMER ASSIGNMENTS ===")
print(customer_assignment_df.head(10).to_string(index=False))

# Additional analytics
print("\n=== PORTFOLIO SIZE DISTRIBUTION ===")
portfolio_sizes = portfolio_summary['Customer_Count'].describe()
print(portfolio_sizes)

print("\n=== DISTANCE ANALYTICS ===")
distance_stats = customer_assignment_df['Distance_Miles'].describe()
print(distance_stats)

print(f"\nCustomers within 5 miles: {len(customer_assignment_df[customer_assignment_df['Distance_Miles'] <= 5])}")
print(f"Customers within 10 miles: {len(customer_assignment_df[customer_assignment_df['Distance_Miles'] <= 10])}")
print(f"Customers within 15 miles: {len(customer_assignment_df[customer_assignment_df['Distance_Miles'] <= 15])}")
print(f"Customers within 20 miles: {len(customer_assignment_df[customer_assignment_df['Distance_Miles'] <= 20])}")

# Export results (optional)
# branch_summary.to_csv('branch_summary.csv', index=False)
# portfolio_summary.to_csv('portfolio_summary.csv', index=False)
# customer_assignment_df.to_csv('customer_assignments.csv', index=False)
