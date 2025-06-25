import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BranchCustomerAnalyzer:
    def __init__(self):
        self.customer_data = None
        self.branch_data = None
        self.qualifying_branches = []
        self.customer_portfolios = {}
        self.analysis_results = {}
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        Returns distance in miles
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in miles
        r = 3959
        return c * r
    
    def load_data(self, customer_file_path, branch_file_path):
        """Load customer and branch data from CSV files"""
        try:
            self.customer_data = pd.read_csv(customer_file_path)
            self.branch_data = pd.read_csv(branch_file_path)
            
            # Validate required columns
            required_customer_cols = ['ECN', 'LAT_NUM', 'LON_NUM', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE']
            required_branch_cols = ['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
            
            for col in required_customer_cols:
                if col not in self.customer_data.columns:
                    raise ValueError(f"Required column '{col}' not found in customer data")
            
            for col in required_branch_cols:
                if col not in self.branch_data.columns:
                    raise ValueError(f"Required column '{col}' not found in branch data")
            
            print(f"Loaded {len(self.customer_data)} customers and {len(self.branch_data)} branches")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def generate_sample_data(self, n_customers=1000, n_branches=15):
        """Generate sample data for testing"""
        np.random.seed(42)
        
        # Generate customer data around NYC area
        base_lat, base_lon = 40.7128, -74.0060
        
        customer_data = []
        for i in range(n_customers):
            lat = base_lat + np.random.normal(0, 0.3)  # ~20 mile spread
            lon = base_lon + np.random.normal(0, 0.3)
            
            customer_data.append({
                'ECN': f'CUST{i+1:04d}',
                'LAT_NUM': lat,
                'LON_NUM': lon,
                'BILLINGSTREET': f'{np.random.randint(1, 999)} Main St',
                'BILLINGCITY': f'City{np.random.randint(1, 10)}',
                'BILLINGSTATE': np.random.choice(['NY', 'NJ', 'CT'])
            })
        
        # Generate branch data
        branch_data = []
        for i in range(n_branches):
            lat = base_lat + np.random.normal(0, 0.2)  # Branches closer to center
            lon = base_lon + np.random.normal(0, 0.2)
            
            branch_data.append({
                'BRANCH_AU': f'BR{i+1:03d}',
                'BRANCH_LAT_NUM': lat,
                'BRANCH_LON_NUM': lon
            })
        
        self.customer_data = pd.DataFrame(customer_data)
        self.branch_data = pd.DataFrame(branch_data)
        
        print(f"Generated {len(self.customer_data)} sample customers and {len(self.branch_data)} sample branches")
    
    def find_customers_within_radius(self, branch_lat, branch_lon, radius_miles=20):
        """Find all customers within specified radius of a branch"""
        customers_within_radius = []
        distances = []
        
        for idx, customer in self.customer_data.iterrows():
            distance = self.haversine_distance(
                branch_lat, branch_lon,
                customer['LAT_NUM'], customer['LON_NUM']
            )
            
            if distance <= radius_miles:
                customers_within_radius.append(idx)
                distances.append(distance)
        
        return customers_within_radius, distances
    
    def optimize_cluster_radius(self, branch_lat, branch_lon, min_customers=200, max_radius=20):
        """
        Find the minimum radius that captures at least min_customers customers
        """
        # Get all customers and their distances to this branch
        all_distances = []
        customer_indices = []
        
        for idx, customer in self.customer_data.iterrows():
            distance = self.haversine_distance(
                branch_lat, branch_lon,
                customer['LAT_NUM'], customer['LON_NUM']
            )
            if distance <= max_radius:
                all_distances.append(distance)
                customer_indices.append(idx)
        
        if len(all_distances) < min_customers:
            return None, [], []  # Not enough customers within max radius
        
        # Sort by distance and take the minimum radius that includes min_customers
        sorted_indices = np.argsort(all_distances)
        optimal_radius = all_distances[sorted_indices[min_customers - 1]]
        
        # Get customers within optimal radius
        optimal_customers = []
        optimal_distances = []
        
        for i in range(min_customers):
            idx = sorted_indices[i]
            optimal_customers.append(customer_indices[idx])
            optimal_distances.append(all_distances[idx])
        
        return optimal_radius, optimal_customers, optimal_distances
    
    def analyze_branches(self, min_customers=200, max_radius=20):
        """Analyze branches to find qualifying ones with optimized radius"""
        print("Analyzing branches...")
        
        self.qualifying_branches = []
        branch_analysis = []
        
        for idx, branch in self.branch_data.iterrows():
            branch_id = branch['BRANCH_AU']
            branch_lat = branch['BRANCH_LAT_NUM']
            branch_lon = branch['BRANCH_LON_NUM']
            
            # Find customers within max radius
            customers_max_radius, distances_max_radius = self.find_customers_within_radius(
                branch_lat, branch_lon, max_radius
            )
            
            # Optimize radius for minimum cluster size
            optimal_radius, optimal_customers, optimal_distances = self.optimize_cluster_radius(
                branch_lat, branch_lon, min_customers, max_radius
            )
            
            branch_info = {
                'BRANCH_AU': branch_id,
                'BRANCH_LAT_NUM': branch_lat,
                'BRANCH_LON_NUM': branch_lon,
                'customers_within_max_radius': len(customers_max_radius),
                'optimal_radius': optimal_radius,
                'optimal_customer_count': len(optimal_customers) if optimal_customers else 0,
                'avg_distance_optimal': np.mean(optimal_distances) if optimal_distances else 0,
                'max_distance_optimal': max(optimal_distances) if optimal_distances else 0,
                'qualifies': optimal_radius is not None,
                'customer_indices': optimal_customers if optimal_customers else [],
                'customer_distances': optimal_distances if optimal_distances else []
            }
            
            branch_analysis.append(branch_info)
            
            if optimal_radius is not None:
                self.qualifying_branches.append(branch_info)
        
        self.analysis_results['branch_analysis'] = branch_analysis
        print(f"Found {len(self.qualifying_branches)} qualifying branches out of {len(self.branch_data)}")
        
        return self.qualifying_branches
    
    def resolve_customer_conflicts(self):
        """
        Resolve conflicts when customers are within range of multiple branches
        Assign each customer to the closest qualifying branch
        """
        print("Resolving customer assignment conflicts...")
        
        customer_assignments = {}  # customer_idx -> (branch_id, distance)
        
        # Collect all customer-branch assignments
        for branch in self.qualifying_branches:
            branch_id = branch['BRANCH_AU']
            for i, customer_idx in enumerate(branch['customer_indices']):
                distance = branch['customer_distances'][i]
                
                if customer_idx not in customer_assignments:
                    customer_assignments[customer_idx] = (branch_id, distance)
                else:
                    # Keep the assignment with shorter distance
                    current_branch, current_distance = customer_assignments[customer_idx]
                    if distance < current_distance:
                        customer_assignments[customer_idx] = (branch_id, distance)
        
        # Create final portfolios
        self.customer_portfolios = defaultdict(list)
        
        for customer_idx, (branch_id, distance) in customer_assignments.items():
            customer_info = self.customer_data.iloc[customer_idx].to_dict()
            customer_info['distance_to_branch'] = distance
            customer_info['customer_index'] = customer_idx
            self.customer_portfolios[branch_id].append(customer_info)
        
        # Update qualifying branches with final counts
        for branch in self.qualifying_branches:
            branch_id = branch['BRANCH_AU']
            final_customers = self.customer_portfolios[branch_id]
            branch['final_customer_count'] = len(final_customers)
            branch['final_avg_distance'] = np.mean([c['distance_to_branch'] for c in final_customers]) if final_customers else 0
            branch['final_max_distance'] = max([c['distance_to_branch'] for c in final_customers]) if final_customers else 0
        
        # Remove branches that don't meet minimum after conflict resolution
        min_customers = 200
        self.qualifying_branches = [b for b in self.qualifying_branches if b['final_customer_count'] >= min_customers]
        
        print(f"After conflict resolution: {len(self.qualifying_branches)} branches still qualify")
        print(f"Total unique customers assigned: {len(customer_assignments)}")
    
    def create_enhanced_clusters(self, use_dbscan=True):
        """
        Create enhanced clusters using DBSCAN for better geographic clustering
        """
        if not use_dbscan:
            return
        
        print("Creating enhanced clusters using DBSCAN...")
        
        # Prepare data for clustering
        customer_coords = self.customer_data[['LAT_NUM', 'LON_NUM']].values
        
        # Convert miles to approximate degrees (rough approximation)
        eps_degrees = 20 / 69  # Approximately 20 miles in degrees
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps_degrees, min_samples=200, metric='haversine')
        
        # Scale coordinates for better clustering
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(customer_coords)
        
        # Fit DBSCAN
        cluster_labels = dbscan.fit_predict(coords_scaled)
        
        # Analyze DBSCAN results
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"DBSCAN found {n_clusters} clusters with {n_noise} noise points")
        
        # Store DBSCAN results
        self.analysis_results['dbscan_labels'] = cluster_labels
        self.analysis_results['dbscan_n_clusters'] = n_clusters
        self.analysis_results['dbscan_n_noise'] = n_noise
    
    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("BRANCH CUSTOMER ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.qualifying_branches:
            print("No qualifying branches found!")
            return
        
        print(f"\nQUALIFYING BRANCHES: {len(self.qualifying_branches)}")
        print("-" * 50)
        
        total_customers = 0
        total_optimal_radius = 0
        
        for i, branch in enumerate(self.qualifying_branches, 1):
            print(f"\n{i}. Branch: {branch['BRANCH_AU']}")
            print(f"   Location: ({branch['BRANCH_LAT_NUM']:.4f}, {branch['BRANCH_LON_NUM']:.4f})")
            print(f"   Optimal Radius: {branch['optimal_radius']:.2f} miles")
            print(f"   Portfolio Size: {branch['final_customer_count']} customers")
            print(f"   Avg Distance: {branch['final_avg_distance']:.2f} miles")
            print(f"   Max Distance: {branch['final_max_distance']:.2f} miles")
            
            total_customers += branch['final_customer_count']
            total_optimal_radius += branch['optimal_radius']
        
        print(f"\nOVERALL STATISTICS")
        print("-" * 50)
        print(f"Total Customers Assigned: {total_customers}")
        print(f"Average Optimal Radius: {total_optimal_radius/len(self.qualifying_branches):.2f} miles")
        print(f"Average Portfolio Size: {total_customers/len(self.qualifying_branches):.0f} customers")
        print(f"Coverage: {total_customers}/{len(self.customer_data)} customers ({100*total_customers/len(self.customer_data):.1f}%)")
    
    def export_results(self, output_dir="./"):
        """Export results to CSV files"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export branch summary
        branch_summary = []
        for branch in self.qualifying_branches:
            branch_summary.append({
                'Branch_ID': branch['BRANCH_AU'],
                'Branch_Lat': branch['BRANCH_LAT_NUM'],
                'Branch_Lon': branch['BRANCH_LON_NUM'],
                'Optimal_Radius_Miles': round(branch['optimal_radius'], 2),
                'Portfolio_Size': branch['final_customer_count'],
                'Avg_Distance_Miles': round(branch['final_avg_distance'], 2),
                'Max_Distance_Miles': round(branch['final_max_distance'], 2),
                'Customers_Within_20_Miles': branch['customers_within_max_radius']
            })
        
        branch_df = pd.DataFrame(branch_summary)
        branch_file = os.path.join(output_dir, 'qualifying_branches_summary.csv')
        branch_df.to_csv(branch_file, index=False)
        print(f"Branch summary exported to: {branch_file}")
        
        # Export customer portfolios
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
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_file = os.path.join(output_dir, 'customer_portfolios.csv')
        portfolio_df.to_csv(portfolio_file, index=False)
        print(f"Customer portfolios exported to: {portfolio_file}")
        
        return branch_file, portfolio_file
    
    def visualize_results(self, figsize=(15, 10)):
        """Create visualizations of the analysis results"""
        if not self.qualifying_branches:
            print("No qualifying branches to visualize!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Branch Customer Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Geographic plot
        ax1 = axes[0, 0]
        
        # Plot all customers
        ax1.scatter(self.customer_data['LON_NUM'], self.customer_data['LAT_NUM'], 
                   c='lightblue', alpha=0.5, s=10, label='All Customers')
        
        # Plot qualifying branches and their optimal radius
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.qualifying_branches)))
        
        for i, branch in enumerate(self.qualifying_branches):
            # Plot branch
            ax1.scatter(branch['BRANCH_LON_NUM'], branch['BRANCH_LAT_NUM'], 
                       c=[colors[i]], s=200, marker='s', 
                       label=f"Branch {branch['BRANCH_AU']}", edgecolors='black', linewidth=2)
            
            # Plot optimal radius circle (approximate)
            circle = plt.Circle((branch['BRANCH_LON_NUM'], branch['BRANCH_LAT_NUM']), 
                              branch['optimal_radius']/69, fill=False, 
                              color=colors[i], linestyle='--', alpha=0.7)
            ax1.add_patch(circle)
            
            # Plot assigned customers
            customers = self.customer_portfolios[branch['BRANCH_AU']]
            customer_lons = [c['LON_NUM'] for c in customers]
            customer_lats = [c['LAT_NUM'] for c in customers]
            ax1.scatter(customer_lons, customer_lats, c=[colors[i]], s=15, alpha=0.7)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Geographic Distribution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio sizes
        ax2 = axes[0, 1]
        branch_ids = [b['BRANCH_AU'] for b in self.qualifying_branches]
        portfolio_sizes = [b['final_customer_count'] for b in self.qualifying_branches]
        
        bars = ax2.bar(range(len(branch_ids)), portfolio_sizes, color=colors[:len(branch_ids)])
        ax2.set_xlabel('Branch')
        ax2.set_ylabel('Number of Customers')
        ax2.set_title('Portfolio Sizes')
        ax2.set_xticks(range(len(branch_ids)))
        ax2.set_xticklabels(branch_ids, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, size in zip(bars, portfolio_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(size), ha='center', va='bottom', fontweight='bold')
        
        # 3. Optimal radius distribution
        ax3 = axes[1, 0]
        optimal_radii = [b['optimal_radius'] for b in self.qualifying_branches]
        
        ax3.hist(optimal_radii, bins=min(10, len(optimal_radii)), 
                color='skyblue', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(optimal_radii), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(optimal_radii):.2f} miles')
        ax3.axvline(20, color='orange', linestyle='--', 
                   label='Max Radius: 20 miles')
        ax3.set_xlabel('Optimal Radius (miles)')
        ax3.set_ylabel('Number of Branches')
        ax3.set_title('Distribution of Optimal Radii')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Distance statistics
        ax4 = axes[1, 1]
        avg_distances = [b['final_avg_distance'] for b in self.qualifying_branches]
        max_distances = [b['final_max_distance'] for b in self.qualifying_branches]
        
        x = range(len(branch_ids))
        width = 0.35
        
        ax4.bar([i - width/2 for i in x], avg_distances, width, 
               label='Average Distance', color='lightgreen', alpha=0.8)
        ax4.bar([i + width/2 for i in x], max_distances, width,
               label='Maximum Distance', color='salmon', alpha=0.8)
        
        ax4.set_xlabel('Branch')
        ax4.set_ylabel('Distance (miles)')
        ax4.set_title('Customer Distance Statistics')
        ax4.set_xticks(x)
        ax4.set_xticklabels(branch_ids, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, customer_file=None, branch_file=None, 
                            use_sample_data=False, min_customers=200, max_radius=20):
        """Run the complete analysis pipeline"""
        print("Starting Branch Customer Analysis...")
        print("="*50)
        
        # Load data
        if use_sample_data:
            self.generate_sample_data()
        else:
            if not customer_file or not branch_file:
                raise ValueError("Please provide customer_file and branch_file paths, or set use_sample_data=True")
            if not self.load_data(customer_file, branch_file):
                return False
        
        # Run analysis
        self.analyze_branches(min_customers, max_radius)
        
        if not self.qualifying_branches:
            print("No branches qualify for hiring bankers!")
            return False
        
        # Resolve conflicts and create final portfolios
        self.resolve_customer_conflicts()
        
        # Create enhanced clusters (optional)
        self.create_enhanced_clusters(use_dbscan=True)
        
        # Print summary
        self.print_analysis_summary()
        
        # Export results
        self.export_results()
        
        # Create visualizations
        self.visualize_results()
        
        return True

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BranchCustomerAnalyzer()
    
    print("Branch Customer Analysis Tool")
    print("="*50)
    print("This tool helps identify branches for hiring bankers based on:")
    print("- Minimum 200 customers within service radius")
    print("- Optimized cluster radius (≤20 miles)")
    print("- Conflict resolution for overlapping territories")
    print()
    
    # Run with sample data
    print("Running analysis with sample data...")
    success = analyzer.run_complete_analysis(use_sample_data=True)
    
    if success:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Check the exported CSV files:")
        print("- qualifying_branches_summary.csv")
        print("- customer_portfolios.csv")
        print()
        print("To use with your own data:")
        print("analyzer.run_complete_analysis(")
        print("    customer_file='path/to/customer_data.csv',")
        print("    branch_file='path/to/branch_data.csv'")
        print(")")
    
    # Example of accessing results programmatically
    if analyzer.qualifying_branches:
        print(f"\nProgrammatic access example:")
        print(f"Number of qualifying branches: {len(analyzer.qualifying_branches)}")
        
        for branch in analyzer.qualifying_branches[:3]:  # Show first 3
            branch_id = branch['BRANCH_AU']
            portfolio_size = branch['final_customer_count']
            avg_distance = branch['final_avg_distance']
            print(f"Branch {branch_id}: {portfolio_size} customers, avg distance {avg_distance:.2f} miles")


analyzer = BranchCustomerAnalyzer()
success = analyzer.run_complete_analysis(
    customer_file='your_customer_data.csv',
    branch_file='your_branch_data.csv'
)

analyzer = BranchCustomerAnalyzer()
success = analyzer.run_complete_analysis(use_sample_data=True)
