import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict

class BranchCustomerAnalyzer:
    def __init__(self):
        self.customer_portfolios = {}
        self.qualifying_branches = []
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in miles"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 3959 * c  # Earth's radius in miles
    
    def optimize_cluster_radius(self, branch_lat, branch_lon, customer_data, 
                              min_customers=200, max_customers=280, max_radius=20):
        """
        Find optimal radius and customer set for a branch
        """
        # Calculate distances to all customers
        distances = []
        customer_indices = []
        
        for idx, customer in customer_data.iterrows():
            distance = self.haversine_distance(
                branch_lat, branch_lon,
                customer['LAT_NUM'], customer['LON_NUM']
            )
            if distance <= max_radius:
                distances.append(distance)
                customer_indices.append(idx)
        
        if len(distances) < min_customers:
            return None, [], []
        
        # Sort by distance and select customers within portfolio limits
        sorted_indices = np.argsort(distances)
        
        # Take minimum customers needed, up to maximum allowed
        num_customers = min(len(distances), max_customers)
        if num_customers < min_customers:
            return None, [], []
        
        # Get optimal customer set
        selected_customers = []
        selected_distances = []
        
        for i in range(num_customers):
            idx = sorted_indices[i]
            selected_customers.append(customer_indices[idx])
            selected_distances.append(distances[idx])
        
        optimal_radius = max(selected_distances)
        return optimal_radius, selected_customers, selected_distances
    
    def analyze_branches(self, customer_df, branch_df, min_customers=200, 
                        max_customers=280, max_radius=20):
        """
        Main analysis function
        """
        print(f"Analyzing {len(branch_df)} branches for customer portfolios...")
        print(f"Portfolio size: {min_customers}-{max_customers} customers")
        print(f"Maximum radius: {max_radius} miles")
        
        self.qualifying_branches = []
        
        for idx, branch in branch_df.iterrows():
            branch_id = branch['BRANCH_AU']
            branch_lat = branch['BRANCH_LAT_NUM']
            branch_lon = branch['BRANCH_LON_NUM']
            
            # Find optimal portfolio for this branch
            optimal_radius, customer_indices, distances = self.optimize_cluster_radius(
                branch_lat, branch_lon, customer_df, 
                min_customers, max_customers, max_radius
            )
            
            if optimal_radius is not None:
                branch_info = {
                    'BRANCH_AU': branch_id,
                    'BRANCH_LAT_NUM': branch_lat,
                    'BRANCH_LON_NUM': branch_lon,
                    'optimal_radius': optimal_radius,
                    'customer_count': len(customer_indices),
                    'avg_distance': np.mean(distances),
                    'max_distance': max(distances),
                    'customer_indices': customer_indices,
                    'customer_distances': distances
                }
                self.qualifying_branches.append(branch_info)
        
        print(f"Found {len(self.qualifying_branches)} qualifying branches")
        return self.qualifying_branches
    
    def resolve_conflicts(self, customer_df):
        """
        Assign customers to closest branch when they're in multiple portfolios
        """
        print("Resolving customer assignment conflicts...")
        
        customer_assignments = {}
        
        # Find best assignment for each customer
        for branch in self.qualifying_branches:
            branch_id = branch['BRANCH_AU']
            for i, customer_idx in enumerate(branch['customer_indices']):
                distance = branch['customer_distances'][i]
                
                if customer_idx not in customer_assignments:
                    customer_assignments[customer_idx] = (branch_id, distance)
                else:
                    current_branch, current_distance = customer_assignments[customer_idx]
                    if distance < current_distance:
                        customer_assignments[customer_idx] = (branch_id, distance)
        
        # Create final portfolios
        self.customer_portfolios = defaultdict(list)
        
        for customer_idx, (branch_id, distance) in customer_assignments.items():
            customer_info = customer_df.iloc[customer_idx].to_dict()
            customer_info['distance_to_branch'] = distance
            self.customer_portfolios[branch_id].append(customer_info)
        
        # Update branch statistics
        for branch in self.qualifying_branches:
            branch_id = branch['BRANCH_AU']
            final_customers = self.customer_portfolios[branch_id]
            branch['final_customer_count'] = len(final_customers)
            if final_customers:
                branch['final_avg_distance'] = np.mean([c['distance_to_branch'] for c in final_customers])
                branch['final_max_distance'] = max([c['distance_to_branch'] for c in final_customers])
            else:
                branch['final_avg_distance'] = 0
                branch['final_max_distance'] = 0
        
        # Remove branches that don't meet minimum after conflict resolution
        min_customers = 200
        self.qualifying_branches = [b for b in self.qualifying_branches 
                                  if b['final_customer_count'] >= min_customers]
        
        print(f"After conflict resolution: {len(self.qualifying_branches)} branches qualify")
        print(f"Total customers assigned: {len(customer_assignments)}")
    
    def get_results_summary(self):
        """Return analysis summary as DataFrame"""
        if not self.qualifying_branches:
            return pd.DataFrame()
        
        summary_data = []
        for branch in self.qualifying_branches:
            summary_data.append({
                'Branch_ID': branch['BRANCH_AU'],
                'Branch_Lat': branch['BRANCH_LAT_NUM'],
                'Branch_Lon': branch['BRANCH_LON_NUM'],
                'Optimal_Radius_Miles': round(branch['optimal_radius'], 2),
                'Portfolio_Size': branch['final_customer_count'],
                'Avg_Distance_Miles': round(branch['final_avg_distance'], 2),
                'Max_Distance_Miles': round(branch['final_max_distance'], 2)
            })
        
        return pd.DataFrame(summary_data)
    
    def get_customer_portfolios(self):
        """Return customer portfolios as DataFrame"""
        portfolio_data = []
        
        for branch_id, customers in self.customer_portfolios.items():
            for customer in customers:
                portfolio_data.append({
                    'Branch_ID': branch_id,
                    'Customer_ECN': customer['ECN'],
                    'Customer_Lat': customer['LAT_NUM'],
                    'Customer_Lon': customer['LON_NUM'],
                    'Distance_Miles': round(customer['distance_to_branch'], 2),
                    'Billing_Street': customer['BILLINGSTREET'],
                    'Billing_City': customer['BILLINGCITY'],
                    'Billing_State': customer['BILLINGSTATE']
                })
        
        return pd.DataFrame(portfolio_data)
    
    def analyze(self, customer_df, branch_df, min_customers=200, max_customers=280, max_radius=20):
        """
        Complete analysis pipeline
        """
        # Run branch analysis
        self.analyze_branches(customer_df, branch_df, min_customers, max_customers, max_radius)
        
        if not self.qualifying_branches:
            print("No branches qualify for hiring bankers!")
            return None, None
        
        # Resolve conflicts
        self.resolve_conflicts(customer_df)
        
        # Print summary
        total_customers = sum(len(customers) for customers in self.customer_portfolios.values())
        avg_radius = np.mean([b['optimal_radius'] for b in self.qualifying_branches])
        avg_portfolio_size = total_customers / len(self.qualifying_branches) if self.qualifying_branches else 0
        
        print(f"\nANALYSIS COMPLETE")
        print(f"Qualifying branches: {len(self.qualifying_branches)}")
        print(f"Total customers assigned: {total_customers}")
        print(f"Average portfolio size: {avg_portfolio_size:.0f}")
        print(f"Average optimal radius: {avg_radius:.2f} miles")
        
        # Return results as DataFrames
        return self.get_results_summary(), self.get_customer_portfolios()

# Usage example
def analyze_branch_customers(customer_df, branch_df, min_customers=200, max_customers=280, max_radius=20):
    """
    Convenience function for quick analysis
    
    Parameters:
    - customer_df: DataFrame with columns ['ECN', 'LAT_NUM', 'LON_NUM', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE']
    - branch_df: DataFrame with columns ['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
    - min_customers: Minimum customers per portfolio (default: 200)
    - max_customers: Maximum customers per portfolio (default: 280)
    - max_radius: Maximum service radius in miles (default: 20)
    
    Returns:
    - branch_summary_df: Summary of qualifying branches
    - customer_portfolios_df: Customer assignments to branches
    """
    analyzer = BranchCustomerAnalyzer()
    return analyzer.analyze(customer_df, branch_df, min_customers, max_customers, max_radius)

# Example usage:
if __name__ == "__main__":
    # Example with sample data
    print("Creating sample data for demonstration...")
    
    # Sample customer data
    np.random.seed(42)
    n_customers = 1000
    base_lat, base_lon = 40.7128, -74.0060
    
    customer_data = []
    for i in range(n_customers):
        customer_data.append({
            'ECN': f'CUST{i+1:04d}',
            'LAT_NUM': base_lat + np.random.normal(0, 0.3),
            'LON_NUM': base_lon + np.random.normal(0, 0.3),
            'BILLINGSTREET': f'{np.random.randint(1, 999)} Main St',
            'BILLINGCITY': f'City{np.random.randint(1, 10)}',
            'BILLINGSTATE': np.random.choice(['NY', 'NJ', 'CT'])
        })
    
    customer_df = pd.DataFrame(customer_data)
    
    # Sample branch data
    branch_data = []
    for i in range(15):
        branch_data.append({
            'BRANCH_AU': f'BR{i+1:03d}',
            'BRANCH_LAT_NUM': base_lat + np.random.normal(0, 0.2),
            'BRANCH_LON_NUM': base_lon + np.random.normal(0, 0.2)
        })
    
    branch_df = pd.DataFrame(branch_data)
    
    # Run analysis
    print(f"Sample data: {len(customer_df)} customers, {len(branch_df)} branches")
    branch_summary, customer_portfolios = analyze_branch_customers(customer_df, branch_df)
    
    if branch_summary is not None:
        print("\nBranch Summary:")
        print(branch_summary)
        print(f"\nCustomer Portfolios shape: {customer_portfolios.shape}")
        print("First few portfolio assignments:")
        print(customer_portfolios.head())

# Simple usage with your DataFrames
branch_summary, customer_portfolios = analyze_branch_customers(
    customer_df,  # Your customer DataFrame
    branch_df     # Your branch DataFrame
)

# Or with custom parameters
branch_summary, customer_portfolios = analyze_branch_customers(
    customer_df, 
    branch_df,
    min_customers=200,
    max_customers=280,
    max_radius=20
)
