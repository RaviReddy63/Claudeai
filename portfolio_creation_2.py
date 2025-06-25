import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerPortfolioClusterer:
    def __init__(self, max_distance_miles: float = 20, min_cluster_size: int = 200):
        """
        Initialize the clusterer with distance and size constraints.
        
        Args:
            max_distance_miles: Maximum distance in miles for customers to be in a cluster
            min_cluster_size: Minimum number of customers required for a viable cluster
        """
        self.max_distance_miles = max_distance_miles
        self.min_cluster_size = min_cluster_size
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the haversine distance between two points on Earth in miles.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance in miles
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in miles
        r = 3956
        return c * r
    
    def create_distance_matrix(self, customers_df: pd.DataFrame, branches_df: pd.DataFrame) -> np.ndarray:
        """
        Create a distance matrix between all customers and branches.
        
        Args:
            customers_df: DataFrame with customer data (ECN, LAT_NUM, LON_NUM)
            branches_df: DataFrame with branch data (BRANCH_AU, BRANCH_LAT_NUM, BRANCH_LON_NUM)
            
        Returns:
            Distance matrix where rows are customers and columns are branches
        """
        n_customers = len(customers_df)
        n_branches = len(branches_df)
        distance_matrix = np.zeros((n_customers, n_branches))
        
        for i, (_, customer) in enumerate(customers_df.iterrows()):
            for j, (_, branch) in enumerate(branches_df.iterrows()):
                distance = self.haversine_distance(
                    customer['LAT_NUM'], customer['LON_NUM'],
                    branch['BRANCH_LAT_NUM'], branch['BRANCH_LON_NUM']
                )
                distance_matrix[i, j] = distance
                
        return distance_matrix
    
    def assign_customers_to_branches(self, customers_df: pd.DataFrame, branches_df: pd.DataFrame) -> Dict:
        """
        Assign customers to branches based on proximity and create clusters.
        
        Args:
            customers_df: DataFrame with customer data
            branches_df: DataFrame with branch data
            
        Returns:
            Dictionary containing cluster information
        """
        # Calculate distance matrix
        distance_matrix = self.create_distance_matrix(customers_df, branches_df)
        
        # Initialize clusters
        clusters = {}
        viable_clusters = {}
        customer_assignments = {}
        
        # For each branch, find customers within the distance threshold
        for branch_idx, (_, branch) in enumerate(branches_df.iterrows()):
            branch_au = branch['BRANCH_AU']
            
            # Find customers within max_distance_miles
            customer_indices = np.where(distance_matrix[:, branch_idx] <= self.max_distance_miles)[0]
            
            # Get customer details for this cluster
            cluster_customers = customers_df.iloc[customer_indices].copy()
            cluster_customers['DISTANCE_TO_BRANCH'] = distance_matrix[customer_indices, branch_idx]
            
            # Sort by distance to branch
            cluster_customers = cluster_customers.sort_values('DISTANCE_TO_BRANCH')
            
            clusters[branch_au] = {
                'branch_info': branch.to_dict(),
                'customers': cluster_customers,
                'customer_count': len(cluster_customers),
                'viable': len(cluster_customers) >= self.min_cluster_size
            }
            
            # Track viable clusters separately
            if len(cluster_customers) >= self.min_cluster_size:
                viable_clusters[branch_au] = clusters[branch_au]
                
                # Assign customers to this branch
                for _, customer in cluster_customers.iterrows():
                    customer_assignments[customer['ECN']] = branch_au
        
        return {
            'all_clusters': clusters,
            'viable_clusters': viable_clusters,
            'customer_assignments': customer_assignments,
            'summary': self._generate_summary(clusters, viable_clusters)
        }
    
    def optimize_overlapping_assignments(self, customers_df: pd.DataFrame, branches_df: pd.DataFrame) -> Dict:
        """
        Handle overlapping assignments by assigning each customer to the nearest viable branch.
        
        Args:
            customers_df: DataFrame with customer data
            branches_df: DataFrame with branch data
            
        Returns:
            Optimized cluster assignments
        """
        # First, get initial assignments
        initial_results = self.assign_customers_to_branches(customers_df, branches_df)
        
        # Get list of viable branches
        viable_branch_aus = list(initial_results['viable_clusters'].keys())
        
        if not viable_branch_aus:
            return initial_results
        
        # Filter branches to only viable ones
        viable_branches_df = branches_df[branches_df['BRANCH_AU'].isin(viable_branch_aus)].copy()
        
        # Recalculate distance matrix for viable branches only
        distance_matrix = self.create_distance_matrix(customers_df, viable_branches_df)
        
        # Initialize optimized clusters
        optimized_clusters = {}
        customer_assignments = {}
        
        # Initialize empty clusters for viable branches
        for branch_au in viable_branch_aus:
            branch_info = branches_df[branches_df['BRANCH_AU'] == branch_au].iloc[0]
            optimized_clusters[branch_au] = {
                'branch_info': branch_info.to_dict(),
                'customers': pd.DataFrame(),
                'customer_count': 0,
                'viable': False
            }
        
        # Assign each customer to nearest viable branch within distance threshold
        for customer_idx, (_, customer) in enumerate(customers_df.iterrows()):
            # Find distances to all viable branches
            customer_distances = distance_matrix[customer_idx, :]
            
            # Find branches within threshold
            within_threshold = customer_distances <= self.max_distance_miles
            
            if np.any(within_threshold):
                # Assign to nearest branch within threshold
                nearest_branch_idx = np.argmin(np.where(within_threshold, customer_distances, np.inf))
                nearest_branch_au = viable_branches_df.iloc[nearest_branch_idx]['BRANCH_AU']
                
                # Add customer to cluster
                customer_data = customer.to_dict()
                customer_data['DISTANCE_TO_BRANCH'] = customer_distances[nearest_branch_idx]
                
                if optimized_clusters[nearest_branch_au]['customers'].empty:
                    optimized_clusters[nearest_branch_au]['customers'] = pd.DataFrame([customer_data])
                else:
                    optimized_clusters[nearest_branch_au]['customers'] = pd.concat([
                        optimized_clusters[nearest_branch_au]['customers'],
                        pd.DataFrame([customer_data])
                    ], ignore_index=True)
                
                customer_assignments[customer['ECN']] = nearest_branch_au
        
        # Update cluster statistics
        final_viable_clusters = {}
        for branch_au, cluster in optimized_clusters.items():
            cluster['customer_count'] = len(cluster['customers'])
            cluster['viable'] = cluster['customer_count'] >= self.min_cluster_size
            
            if cluster['viable']:
                # Sort customers by distance
                cluster['customers'] = cluster['customers'].sort_values('DISTANCE_TO_BRANCH')
                final_viable_clusters[branch_au] = cluster
        
        return {
            'all_clusters': optimized_clusters,
            'viable_clusters': final_viable_clusters,
            'customer_assignments': customer_assignments,
            'summary': self._generate_summary(optimized_clusters, final_viable_clusters)
        }
    
    def _generate_summary(self, all_clusters: Dict, viable_clusters: Dict) -> Dict:
        """Generate summary statistics for the clustering results."""
        total_branches = len(all_clusters)
        viable_branches = len(viable_clusters)
        
        # Calculate customer statistics
        total_customers_assigned = sum(cluster['customer_count'] for cluster in viable_clusters.values())
        
        cluster_sizes = [cluster['customer_count'] for cluster in viable_clusters.values()]
        
        summary = {
            'total_branches_analyzed': total_branches,
            'viable_branches': viable_branches,
            'branches_to_hire_bankers': list(viable_clusters.keys()),
            'total_customers_in_viable_clusters': total_customers_assigned,
            'average_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'cluster_size_distribution': {
                'mean': np.mean(cluster_sizes) if cluster_sizes else 0,
                'std': np.std(cluster_sizes) if cluster_sizes else 0,
                'median': np.median(cluster_sizes) if cluster_sizes else 0
            }
        }
        
        return summary
    
    def visualize_results(self, results: Dict, customers_df: pd.DataFrame, branches_df: pd.DataFrame):
        """
        Create visualizations of the clustering results.
        
        Args:
            results: Results from the clustering algorithm
            customers_df: Original customer data
            branches_df: Original branch data
        """
        viable_clusters = results['viable_clusters']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Geographic distribution
        colors = plt.cm.Set3(np.linspace(0, 1, len(viable_clusters)))
        
        for i, (branch_au, cluster) in enumerate(viable_clusters.items()):
            customers = cluster['customers']
            branch_info = cluster['branch_info']
            
            # Plot customers
            ax1.scatter(customers['LON_NUM'], customers['LAT_NUM'], 
                       c=[colors[i]], alpha=0.6, s=20, label=f'AU {branch_au}')
            
            # Plot branch location
            ax1.scatter(branch_info['BRANCH_LON_NUM'], branch_info['BRANCH_LAT_NUM'], 
                       c=[colors[i]], s=200, marker='*', edgecolors='black', linewidth=2)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Geographic Distribution of Viable Clusters')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster sizes
        branch_names = list(viable_clusters.keys())
        cluster_sizes = [cluster['customer_count'] for cluster in viable_clusters.values()]
        
        bars = ax2.bar(range(len(branch_names)), cluster_sizes, color=colors[:len(branch_names)])
        ax2.axhline(y=self.min_cluster_size, color='red', linestyle='--', 
                   label=f'Min Cluster Size ({self.min_cluster_size})')
        ax2.set_xlabel('Branch AU')
        ax2.set_ylabel('Number of Customers')
        ax2.set_title('Cluster Sizes for Viable Branches')
        ax2.set_xticks(range(len(branch_names)))
        ax2.set_xticklabels([f'AU {au}' for au in branch_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(size), ha='center', va='bottom')
        
        # Plot 3: Distance distribution
        all_distances = []
        for cluster in viable_clusters.values():
            all_distances.extend(cluster['customers']['DISTANCE_TO_BRANCH'].tolist())
        
        ax3.hist(all_distances, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(x=self.max_distance_miles, color='red', linestyle='--', 
                   label=f'Max Distance ({self.max_distance_miles} miles)')
        ax3.set_xlabel('Distance to Branch (miles)')
        ax3.set_ylabel('Number of Customers')
        ax3.set_title('Distribution of Customer Distances to Assigned Branches')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4.axis('off')
        summary = results['summary']
        summary_text = f"""
        CLUSTERING SUMMARY
        
        Total Branches Analyzed: {summary['total_branches_analyzed']}
        Viable Branches (Hire Bankers): {summary['viable_branches']}
        
        Customer Assignment:
        • Total Customers in Clusters: {summary['total_customers_in_viable_clusters']:,}
        • Average Cluster Size: {summary['average_cluster_size']:.1f}
        • Min Cluster Size: {summary['min_cluster_size']}
        • Max Cluster Size: {summary['max_cluster_size']}
        
        Branches to Hire Bankers:
        {', '.join([f'AU {au}' for au in summary['branches_to_hire_bankers']])}
        
        Parameters Used:
        • Max Distance: {self.max_distance_miles} miles
        • Min Cluster Size: {self.min_cluster_size} customers
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, results: Dict, filename_prefix: str = 'customer_clusters'):
        """
        Export clustering results to CSV files.
        
        Args:
            results: Results from clustering algorithm
            filename_prefix: Prefix for output files
        """
        viable_clusters = results['viable_clusters']
        
        # Export cluster assignments
        cluster_assignments = []
        for branch_au, cluster in viable_clusters.items():
            for _, customer in cluster['customers'].iterrows():
                cluster_assignments.append({
                    'ECN': customer['ECN'],
                    'ASSIGNED_BRANCH_AU': branch_au,
                    'DISTANCE_TO_BRANCH': customer['DISTANCE_TO_BRANCH'],
                    'CUSTOMER_LAT': customer['LAT_NUM'],
                    'CUSTOMER_LON': customer['LON_NUM'],
                    'BRANCH_LAT': cluster['branch_info']['BRANCH_LAT_NUM'],
                    'BRANCH_LON': cluster['branch_info']['BRANCH_LON_NUM']
                })
        
        assignments_df = pd.DataFrame(cluster_assignments)
        assignments_df.to_csv(f'{filename_prefix}_assignments.csv', index=False)
        
        # Export cluster summary
        cluster_summary = []
        for branch_au, cluster in viable_clusters.items():
            cluster_summary.append({
                'BRANCH_AU': branch_au,
                'BRANCH_LAT': cluster['branch_info']['BRANCH_LAT_NUM'],
                'BRANCH_LON': cluster['branch_info']['BRANCH_LON_NUM'],
                'CUSTOMER_COUNT': cluster['customer_count'],
                'AVG_DISTANCE': cluster['customers']['DISTANCE_TO_BRANCH'].mean(),
                'MAX_DISTANCE': cluster['customers']['DISTANCE_TO_BRANCH'].max(),
                'MIN_DISTANCE': cluster['customers']['DISTANCE_TO_BRANCH'].min()
            })
        
        summary_df = pd.DataFrame(cluster_summary)
        summary_df.to_csv(f'{filename_prefix}_summary.csv', index=False)
        
        print(f"Results exported to:")
        print(f"- {filename_prefix}_assignments.csv")
        print(f"- {filename_prefix}_summary.csv")

# Example usage
def example_usage():
    """
    Example of how to use the CustomerPortfolioClusterer class.
    """
    
    # Sample data creation (replace with your actual data loading)
    np.random.seed(42)
    
    # Create sample customer data
    n_customers = 5000
    customers_data = {
        'ECN': [f'CUST_{i:05d}' for i in range(n_customers)],
        'LAT_NUM': np.random.normal(40.7128, 2, n_customers),  # Around NYC
        'LON_NUM': np.random.normal(-74.0060, 2, n_customers),
        'BILLINGSTREET': [f'{np.random.randint(1, 9999)} Main St' for _ in range(n_customers)],
        'BILLINGCITY': ['New York'] * n_customers,
        'BILLINGSTATE': ['NY'] * n_customers
    }
    customers_df = pd.DataFrame(customers_data)
    
    # Create sample branch data
    n_branches = 20
    branches_data = {
        'BRANCH_AU': [f'AU_{i:03d}' for i in range(n_branches)],
        'BRANCH_LAT_NUM': np.random.normal(40.7128, 1.5, n_branches),
        'BRANCH_LON_NUM': np.random.normal(-74.0060, 1.5, n_branches)
    }
    branches_df = pd.DataFrame(branches_data)
    
    # Initialize clusterer
    clusterer = CustomerPortfolioClusterer(max_distance_miles=20, min_cluster_size=200)
    
    # Perform clustering
    print("Performing customer clustering...")
    results = clusterer.optimize_overlapping_assignments(customers_df, branches_df)
    
    # Print summary
    print("\n" + "="*50)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*50)
    print(f"Total branches analyzed: {results['summary']['total_branches_analyzed']}")
    print(f"Viable branches for hiring bankers: {results['summary']['viable_branches']}")
    print(f"Branches to hire bankers: {results['summary']['branches_to_hire_bankers']}")
    print(f"Total customers assigned: {results['summary']['total_customers_in_viable_clusters']:,}")
    print(f"Average cluster size: {results['summary']['average_cluster_size']:.1f}")
    
    # Visualize results
    clusterer.visualize_results(results, customers_df, branches_df)
    
    # Export results
    clusterer.export_results(results, 'customer_portfolio_clusters')
    
    return results

if __name__ == "__main__":
    results = example_usage()
