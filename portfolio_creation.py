import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, ceil
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import itertools

@dataclass
class Customer:
    ecn: str
    lat: float
    lon: float
    billing_street: str
    billing_city: str
    billing_state: str

@dataclass
class Branch:
    branch_au: str
    lat: float
    lon: float

@dataclass
class Portfolio:
    portfolio_id: str
    portfolio_type: str  # 'in-market' or 'centralized'
    branch_au: Optional[str]  # None for centralized
    customers: List[Customer]
    
class PortfolioOptimizer:
    def __init__(self, min_portfolio_size: int = 220, max_portfolio_size: int = 250, 
                 proximity_miles: float = 20.0):
        self.min_portfolio_size = min_portfolio_size
        self.max_portfolio_size = max_portfolio_size
        self.proximity_miles = proximity_miles
        
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees) in miles
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
    
    def get_customers_within_proximity(self, branch: Branch, customers: List[Customer]) -> List[Customer]:
        """Get all customers within proximity_miles of a branch"""
        nearby_customers = []
        for customer in customers:
            distance = self.haversine_distance(
                branch.lat, branch.lon, customer.lat, customer.lon
            )
            if distance <= self.proximity_miles:
                nearby_customers.append(customer)
        return nearby_customers
    
    def create_in_market_portfolios(self, branches: List[Branch], 
                                  customers: List[Customer]) -> Tuple[List[Portfolio], List[Customer]]:
        """
        Create in-market portfolios by assigning customers to nearby branches
        Returns: (portfolios, unassigned_customers)
        """
        portfolios = []
        assigned_customers = set()
        
        # Sort branches by potential customer count (descending) for better optimization
        branch_customer_counts = []
        for branch in branches:
            nearby = self.get_customers_within_proximity(branch, customers)
            branch_customer_counts.append((branch, len(nearby), nearby))
        
        # Sort by customer count descending
        branch_customer_counts.sort(key=lambda x: x[1], reverse=True)
        
        for branch, count, nearby_customers in branch_customer_counts:
            # Filter out already assigned customers
            available_customers = [c for c in nearby_customers if c.ecn not in assigned_customers]
            
            if len(available_customers) >= self.min_portfolio_size:
                # Create portfolios for this branch
                portfolios_for_branch = self._create_portfolios_for_branch(
                    branch, available_customers
                )
                portfolios.extend(portfolios_for_branch)
                
                # Mark customers as assigned
                for portfolio in portfolios_for_branch:
                    for customer in portfolio.customers:
                        assigned_customers.add(customer.ecn)
        
        # Return unassigned customers
        unassigned_customers = [c for c in customers if c.ecn not in assigned_customers]
        return portfolios, unassigned_customers
    
    def _create_portfolios_for_branch(self, branch: Branch, 
                                    customers: List[Customer]) -> List[Portfolio]:
        """Create multiple portfolios for a single branch if needed"""
        portfolios = []
        remaining_customers = customers.copy()
        portfolio_counter = 1
        
        while len(remaining_customers) >= self.min_portfolio_size:
            # Take up to max_portfolio_size customers
            portfolio_customers = remaining_customers[:self.max_portfolio_size]
            remaining_customers = remaining_customers[self.max_portfolio_size:]
            
            portfolio_id = f"{branch.branch_au}_P{portfolio_counter}"
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                portfolio_type="in-market",
                branch_au=branch.branch_au,
                customers=portfolio_customers
            )
            portfolios.append(portfolio)
            portfolio_counter += 1
        
        # If remaining customers are less than min but more than 0,
        # try to distribute them to existing portfolios
        if remaining_customers and portfolios:
            self._redistribute_remaining_customers(portfolios, remaining_customers)
        
        return portfolios
    
    def _redistribute_remaining_customers(self, portfolios: List[Portfolio], 
                                       remaining_customers: List[Customer]):
        """Redistribute remaining customers to existing portfolios if possible"""
        for customer in remaining_customers:
            # Find portfolio with least customers that can accommodate one more
            best_portfolio = None
            min_size = float('inf')
            
            for portfolio in portfolios:
                if len(portfolio.customers) < self.max_portfolio_size and len(portfolio.customers) < min_size:
                    best_portfolio = portfolio
                    min_size = len(portfolio.customers)
            
            if best_portfolio:
                best_portfolio.customers.append(customer)
    
    def create_centralized_portfolios(self, unassigned_customers: List[Customer]) -> List[Portfolio]:
        """Create centralized portfolios from unassigned customers"""
        portfolios = []
        remaining_customers = unassigned_customers.copy()
        portfolio_counter = 1
        
        while len(remaining_customers) >= self.min_portfolio_size:
            # Take up to max_portfolio_size customers
            portfolio_customers = remaining_customers[:self.max_portfolio_size]
            remaining_customers = remaining_customers[self.max_portfolio_size:]
            
            portfolio_id = f"CENTRALIZED_P{portfolio_counter}"
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                portfolio_type="centralized",
                branch_au=None,
                customers=portfolio_customers
            )
            portfolios.append(portfolio)
            portfolio_counter += 1
        
        # Handle remaining customers by distributing to existing centralized portfolios
        if remaining_customers and portfolios:
            self._redistribute_remaining_customers(portfolios, remaining_customers)
        elif remaining_customers and not portfolios:
            # If we have customers but no portfolios (less than min_size), 
            # create one portfolio anyway
            portfolio_id = f"CENTRALIZED_P1"
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                portfolio_type="centralized",
                branch_au=None,
                customers=remaining_customers
            )
            portfolios.append(portfolio)
        
        return portfolios
    
    def optimize_portfolios(self, customer_data: pd.DataFrame, 
                          branch_data: pd.DataFrame) -> Tuple[List[Portfolio], Dict]:
        """
        Main optimization function
        Returns: (all_portfolios, summary_stats)
        """
        # Convert data to objects
        customers = [
            Customer(
                ecn=row['ECN'],
                lat=row['LAT_NUM'],
                lon=row['LON_NUM'],
                billing_street=row['BILLINGSTREET'],
                billing_city=row['BILLINGCITY'],
                billing_state=row['BILLINGSTATE']
            )
            for _, row in customer_data.iterrows()
        ]
        
        branches = [
            Branch(
                branch_au=row['BRANCH_AU'],
                lat=row['BRANCH_LAT_NUM'],
                lon=row['BRANCH_LON_NUM']
            )
            for _, row in branch_data.iterrows()
        ]
        
        # Create in-market portfolios
        in_market_portfolios, unassigned_customers = self.create_in_market_portfolios(
            branches, customers
        )
        
        # Create centralized portfolios
        centralized_portfolios = self.create_centralized_portfolios(unassigned_customers)
        
        # Combine all portfolios
        all_portfolios = in_market_portfolios + centralized_portfolios
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(all_portfolios, customers, branches)
        
        return all_portfolios, summary_stats
    
    def _generate_summary_stats(self, portfolios: List[Portfolio], 
                               customers: List[Customer], branches: List[Branch]) -> Dict:
        """Generate summary statistics"""
        in_market_portfolios = [p for p in portfolios if p.portfolio_type == "in-market"]
        centralized_portfolios = [p for p in portfolios if p.portfolio_type == "centralized"]
        
        total_customers_assigned = sum(len(p.customers) for p in portfolios)
        
        portfolio_sizes = [len(p.customers) for p in portfolios]
        
        stats = {
            'total_customers': len(customers),
            'total_branches': len(branches),
            'total_portfolios': len(portfolios),
            'in_market_portfolios': len(in_market_portfolios),
            'centralized_portfolios': len(centralized_portfolios),
            'customers_assigned': total_customers_assigned,
            'customers_unassigned': len(customers) - total_customers_assigned,
            'avg_portfolio_size': np.mean(portfolio_sizes) if portfolio_sizes else 0,
            'min_portfolio_size': min(portfolio_sizes) if portfolio_sizes else 0,
            'max_portfolio_size': max(portfolio_sizes) if portfolio_sizes else 0,
            'portfolio_size_std': np.std(portfolio_sizes) if portfolio_sizes else 0
        }
        
        return stats
    
    def export_results_to_dataframe(self, portfolios: List[Portfolio]) -> pd.DataFrame:
        """Export portfolio results to a DataFrame"""
        results = []
        
        for portfolio in portfolios:
            for customer in portfolio.customers:
                results.append({
                    'Portfolio_ID': portfolio.portfolio_id,
                    'Portfolio_Type': portfolio.portfolio_type,
                    'Branch_AU': portfolio.branch_au,
                    'ECN': customer.ecn,
                    'Customer_LAT': customer.lat,
                    'Customer_LON': customer.lon,
                    'Billing_Street': customer.billing_street,
                    'Billing_City': customer.billing_city,
                    'Billing_State': customer.billing_state,
                    'Portfolio_Size': len(portfolio.customers)
                })
        
        return pd.DataFrame(results)

# Example usage function
def run_portfolio_optimization(customer_df: pd.DataFrame, branch_df: pd.DataFrame):
    """
    Main function to run the portfolio optimization
    
    Args:
        customer_df: DataFrame with columns ['ECN', 'LAT_NUM', 'LON_NUM', 'BILLINGSTREET', 'BILLINGCITY', 'BILLINGSTATE']
        branch_df: DataFrame with columns ['BRANCH_AU', 'BRANCH_LAT_NUM', 'BRANCH_LON_NUM']
    
    Returns:
        results_df: DataFrame with portfolio assignments
        summary: Dictionary with summary statistics
    """
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        min_portfolio_size=220,
        max_portfolio_size=250,
        proximity_miles=20.0
    )
    
    # Run optimization
    portfolios, summary = optimizer.optimize_portfolios(customer_df, branch_df)
    
    # Export results
    results_df = optimizer.export_results_to_dataframe(portfolios)
    
    # Print summary
    print("Portfolio Optimization Summary:")
    print("=" * 40)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return results_df, summary

# Example of how to use:
"""
# Load your data
customer_df = pd.read_csv('customer_data.csv')
branch_df = pd.read_csv('branch_data.csv')

# Run optimization
results_df, summary = run_portfolio_optimization(customer_df, branch_df)

# Save results
results_df.to_csv('portfolio_assignments.csv', index=False)

# View portfolio distribution
portfolio_summary = results_df.groupby(['Portfolio_ID', 'Portfolio_Type', 'Branch_AU']).agg({
    'ECN': 'count'
}).rename(columns={'ECN': 'Customer_Count'}).reset_index()

print(portfolio_summary)
"""
