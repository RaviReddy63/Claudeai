import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class BankerOptimizer:
    def __init__(self, branches_df, customers_df, total_bankers=300, book_size=250, 
                 proximity_radius=20, in_branch_ratio=0.7):
        """
        Initialize the banker optimization system.
        
        Parameters:
        - branches_df: DataFrame with columns ['branch_id', 'latitude', 'longitude']
        - customers_df: DataFrame with columns ['customer_id', 'latitude', 'longitude']
        - total_bankers: Total number of bankers available
        - book_size: Target number of customers per banker
        - proximity_radius: Radius in miles for in-branch banker coverage
        - in_branch_ratio: Ratio of in-branch to total bankers
        """
        self.branches_df = branches_df.copy()
        self.customers_df = customers_df.copy()
        self.total_bankers = total_bankers
        self.book_size = book_size
        self.proximity_radius = proximity_radius
        self.in_branch_bankers = int(total_bankers * in_branch_ratio)
        self.remote_bankers = total_bankers - self.in_branch_bankers
        
        # Results storage
        self.selected_branches = None
        self.in_branch_assignments = None
        self.remote_assignments = None
        self.unassigned_customers = None
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in miles."""
        return geodesic((lat1, lon1), (lat2, lon2)).miles
    
    def find_customers_near_branch(self, branch_lat, branch_lon, customers_df):
        """Find customers within proximity radius of a branch."""
        distances = customers_df.apply(
            lambda row: self.calculate_distance(
                branch_lat, branch_lon, row['latitude'], row['longitude']
            ), axis=1
        )
        return customers_df[distances <= self.proximity_radius]
    
    def select_optimal_branches(self):
        """
        Select branches for in-branch bankers using a greedy approach
        that maximizes customer coverage.
        """
        print(f"Selecting {self.in_branch_bankers} branches for in-branch bankers...")
        
        selected_branches = []
        covered_customers = set()
        remaining_branches = self.branches_df.copy()
        
        for i in range(self.in_branch_bankers):
            best_branch = None
            best_coverage = 0
            best_new_customers = set()
            
            for _, branch in remaining_branches.iterrows():
                # Find customers near this branch
                nearby_customers = self.find_customers_near_branch(
                    branch['latitude'], branch['longitude'], self.customers_df
                )
                
                # Calculate new customers that would be covered
                nearby_customer_ids = set(nearby_customers['customer_id'])
                new_customers = nearby_customer_ids - covered_customers
                
                if len(new_customers) > best_coverage:
                    best_coverage = len(new_customers)
                    best_branch = branch
                    best_new_customers = new_customers
            
            if best_branch is not None:
                selected_branches.append(best_branch)
                covered_customers.update(best_new_customers)
                remaining_branches = remaining_branches[
                    remaining_branches['branch_id'] != best_branch['branch_id']
                ]
                
                print(f"Selected branch {best_branch['branch_id']} - "
                      f"New customers covered: {best_coverage}, "
                      f"Total covered: {len(covered_customers)}")
        
        self.selected_branches = pd.DataFrame(selected_branches)
        return self.selected_branches
    
    def assign_in_branch_customers(self):
        """Assign customers to in-branch bankers."""
        print("Assigning customers to in-branch bankers...")
        
        assignments = {}
        assigned_customers = set()
        
        for _, branch in self.selected_branches.iterrows():
            branch_id = branch['branch_id']
            
            # Find customers within proximity
            nearby_customers = self.find_customers_near_branch(
                branch['latitude'], branch['longitude'], self.customers_df
            )
            
            # Remove already assigned customers
            available_customers = nearby_customers[
                ~nearby_customers['customer_id'].isin(assigned_customers)
            ]
            
            # Calculate distances and sort by proximity
            distances = available_customers.apply(
                lambda row: self.calculate_distance(
                    branch['latitude'], branch['longitude'], 
                    row['latitude'], row['longitude']
                ), axis=1
            )
            
            available_customers = available_customers.copy()
            available_customers['distance'] = distances
            available_customers = available_customers.sort_values('distance')
            
            # Assign up to book_size customers
            customers_to_assign = available_customers.head(self.book_size)
            
            assignments[branch_id] = {
                'banker_type': 'in_branch',
                'branch_id': branch_id,
                'branch_lat': branch['latitude'],
                'branch_lon': branch['longitude'],
                'customers': customers_to_assign['customer_id'].tolist(),
                'customer_count': len(customers_to_assign)
            }
            
            assigned_customers.update(customers_to_assign['customer_id'])
            
            print(f"Branch {branch_id}: Assigned {len(customers_to_assign)} customers")
        
        self.in_branch_assignments = assignments
        return assignments
    
    def assign_remote_customers(self):
        """Assign remaining customers to remote bankers using clustering."""
        print("Assigning remaining customers to remote bankers...")
        
        # Get customers not assigned to in-branch bankers
        if self.in_branch_assignments:
            assigned_customer_ids = set()
            for assignment in self.in_branch_assignments.values():
                assigned_customer_ids.update(assignment['customers'])
            
            unassigned_customers = self.customers_df[
                ~self.customers_df['customer_id'].isin(assigned_customer_ids)
            ].copy()
        else:
            unassigned_customers = self.customers_df.copy()
        
        if len(unassigned_customers) == 0:
            print("No customers left for remote bankers")
            self.remote_assignments = {}
            return {}
        
        print(f"Assigning {len(unassigned_customers)} customers to {self.remote_bankers} remote bankers")
        
        # Use K-means clustering to group customers for remote bankers
        if len(unassigned_customers) > self.remote_bankers:
            # Prepare coordinates for clustering
            coordinates = unassigned_customers[['latitude', 'longitude']].values
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.remote_bankers, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(coordinates)
            unassigned_customers['cluster'] = clusters
            
            assignments = {}
            for cluster_id in range(self.remote_bankers):
                cluster_customers = unassigned_customers[
                    unassigned_customers['cluster'] == cluster_id
                ]
                
                # Calculate cluster center
                center_lat = cluster_customers['latitude'].mean()
                center_lon = cluster_customers['longitude'].mean()
                
                assignments[f'remote_banker_{cluster_id + 1}'] = {
                    'banker_type': 'remote',
                    'banker_id': f'remote_banker_{cluster_id + 1}',
                    'cluster_center_lat': center_lat,
                    'cluster_center_lon': center_lon,
                    'customers': cluster_customers['customer_id'].tolist(),
                    'customer_count': len(cluster_customers)
                }
                
                print(f"Remote banker {cluster_id + 1}: Assigned {len(cluster_customers)} customers")
        
        else:
            # If fewer customers than remote bankers, assign individually
            assignments = {}
            for i, (_, customer) in enumerate(unassigned_customers.iterrows()):
                banker_id = f'remote_banker_{i + 1}'
                assignments[banker_id] = {
                    'banker_type': 'remote',
                    'banker_id': banker_id,
                    'cluster_center_lat': customer['latitude'],
                    'cluster_center_lon': customer['longitude'],
                    'customers': [customer['customer_id']],
                    'customer_count': 1
                }
        
        self.remote_assignments = assignments
        return assignments
    
    def optimize(self):
        """Run the complete optimization process."""
        print("Starting banker optimization process...")
        print(f"Total branches: {len(self.branches_df)}")
        print(f"Total customers: {len(self.customers_df)}")
        print(f"Total bankers: {self.total_bankers}")
        print(f"In-branch bankers: {self.in_branch_bankers}")
        print(f"Remote bankers: {self.remote_bankers}")
        print(f"Target book size: {self.book_size}")
        print("-" * 50)
        
        # Step 1: Select optimal branches for in-branch bankers
        self.select_optimal_branches()
        
        # Step 2: Assign customers to in-branch bankers
        self.assign_in_branch_customers()
        
        # Step 3: Assign remaining customers to remote bankers
        self.assign_remote_customers()
        
        # Generate summary
        self.generate_summary()
        
        return {
            'selected_branches': self.selected_branches,
            'in_branch_assignments': self.in_branch_assignments,
            'remote_assignments': self.remote_assignments
        }
    
    def generate_summary(self):
        """Generate and print optimization summary."""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        # In-branch summary
        total_in_branch_customers = 0
        if self.in_branch_assignments:
            for assignment in self.in_branch_assignments.values():
                total_in_branch_customers += assignment['customer_count']
        
        print(f"IN-BRANCH BANKERS:")
        print(f"  Number of branches selected: {len(self.selected_branches) if self.selected_branches is not None else 0}")
        print(f"  Total customers assigned: {total_in_branch_customers}")
        print(f"  Average book size: {total_in_branch_customers / self.in_branch_bankers if self.in_branch_bankers > 0 else 0:.1f}")
        
        # Remote summary
        total_remote_customers = 0
        if self.remote_assignments:
            for assignment in self.remote_assignments.values():
                total_remote_customers += assignment['customer_count']
        
        print(f"\nREMOTE BANKERS:")
        print(f"  Number of remote bankers: {self.remote_bankers}")
        print(f"  Total customers assigned: {total_remote_customers}")
        print(f"  Average book size: {total_remote_customers / self.remote_bankers if self.remote_bankers > 0 else 0:.1f}")
        
        # Overall summary
        total_assigned = total_in_branch_customers + total_remote_customers
        print(f"\nOVERALL:")
        print(f"  Total customers assigned: {total_assigned}")
        print(f"  Total customers in dataset: {len(self.customers_df)}")
        print(f"  Coverage: {total_assigned / len(self.customers_df) * 100:.1f}%")
        print(f"  Average book size across all bankers: {total_assigned / self.total_bankers:.1f}")
    
    def export_results(self, filename_prefix='banker_optimization'):
        """Export results to CSV files."""
        if self.selected_branches is not None:
            self.selected_branches.to_csv(f'{filename_prefix}_selected_branches.csv', index=False)
        
        # Export in-branch assignments
        if self.in_branch_assignments:
            in_branch_data = []
            for branch_id, assignment in self.in_branch_assignments.items():
                for customer_id in assignment['customers']:
                    in_branch_data.append({
                        'banker_type': 'in_branch',
                        'branch_id': branch_id,
                        'banker_id': f'in_branch_{branch_id}',
                        'customer_id': customer_id,
                        'branch_lat': assignment['branch_lat'],
                        'branch_lon': assignment['branch_lon']
                    })
            
            pd.DataFrame(in_branch_data).to_csv(f'{filename_prefix}_in_branch_assignments.csv', index=False)
        
        # Export remote assignments
        if self.remote_assignments:
            remote_data = []
            for banker_id, assignment in self.remote_assignments.items():
                for customer_id in assignment['customers']:
                    remote_data.append({
                        'banker_type': 'remote',
                        'banker_id': banker_id,
                        'customer_id': customer_id,
                        'cluster_center_lat': assignment['cluster_center_lat'],
                        'cluster_center_lon': assignment['cluster_center_lon']
                    })
            
            pd.DataFrame(remote_data).to_csv(f'{filename_prefix}_remote_assignments.csv', index=False)
        
        print(f"\nResults exported to CSV files with prefix '{filename_prefix}'")


# Example usage:
def main():
    """
    Example usage of the BankerOptimizer class.
    Replace this with your actual data loading.
    """
    
    # Example: Load your data
    # branches_df = pd.read_csv('branches.csv')  # columns: branch_id, latitude, longitude
    # customers_df = pd.read_csv('customers.csv')  # columns: customer_id, latitude, longitude
    
    # For demonstration, create sample data
    np.random.seed(42)
    
    # Create sample branches (2000 branches across US-like coordinates)
    branches_data = []
    for i in range(2000):
        branches_data.append({
            'branch_id': f'branch_{i+1}',
            'latitude': np.random.uniform(25.0, 49.0),  # US latitude range
            'longitude': np.random.uniform(-125.0, -65.0)  # US longitude range
        })
    branches_df = pd.DataFrame(branches_data)
    
    # Create sample customers (assume 75000 customers for 300 bankers * 250 book size)
    customers_data = []
    for i in range(75000):
        customers_data.append({
            'customer_id': f'customer_{i+1}',
            'latitude': np.random.uniform(25.0, 49.0),
            'longitude': np.random.uniform(-125.0, -65.0)
        })
    customers_df = pd.DataFrame(customers_data)
    
    print("Sample data created for demonstration")
    print(f"Branches: {len(branches_df)}")
    print(f"Customers: {len(customers_df)}")
    
    # Initialize optimizer
    optimizer = BankerOptimizer(
        branches_df=branches_df,
        customers_df=customers_df,
        total_bankers=300,
        book_size=250,
        proximity_radius=20,
        in_branch_ratio=0.7  # 70% in-branch, 30% remote
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Export results
    optimizer.export_results('banker_optimization_results')
    
    return optimizer, results

if __name__ == "__main__":
    # Required packages (install with pip):
    # pip install pandas numpy scikit-learn geopy scipy
    
    optimizer, results = main()
