from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import math

class SparkPortfolioOptimizer:
    def __init__(self, spark: SparkSession, min_portfolio_size: int = 220, 
                 max_portfolio_size: int = 250, proximity_miles: float = 20.0):
        self.spark = spark
        self.min_portfolio_size = min_portfolio_size
        self.max_portfolio_size = max_portfolio_size
        self.proximity_miles = proximity_miles
        
    def haversine_distance_udf(self):
        """Create UDF for haversine distance calculation"""
        def haversine(lat1, lon1, lat2, lon2):
            if any(x is None for x in [lat1, lon1, lat2, lon2]):
                return None
                
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            # Radius of earth in miles
            r = 3956
            return c * r
            
        return udf(haversine, DoubleType())
    
    def create_customer_branch_distances(self, customer_df: DataFrame, branch_df: DataFrame) -> DataFrame:
        """Create a DataFrame with distances between all customers and branches"""
        haversine_udf = self.haversine_distance_udf()
        
        # Cross join customers with branches
        customer_branch = customer_df.crossJoin(branch_df.select(
            col("BRANCH_AU"),
            col("BRANCH_LAT_NUM").alias("BRANCH_LAT"),
            col("BRANCH_LON_NUM").alias("BRANCH_LON")
        ))
        
        # Calculate distances
        distances_df = customer_branch.withColumn(
            "distance_miles",
            haversine_udf(
                col("LAT_NUM"), col("LON_NUM"),
                col("BRANCH_LAT"), col("BRANCH_LON")
            )
        ).filter(
            col("distance_miles") <= self.proximity_miles
        )
        
        return distances_df
    
    def create_in_market_portfolios(self, customer_df: DataFrame, branch_df: DataFrame) -> DataFrame:
        """Create in-market portfolios"""
        # Get customer-branch distances within proximity
        distances_df = self.create_customer_branch_distances(customer_df, branch_df)
        
        # Count customers per branch
        branch_customer_counts = distances_df.groupBy("BRANCH_AU").agg(
            count("ECN").alias("customer_count")
        ).filter(
            col("customer_count") >= self.min_portfolio_size
        )
        
        # Get eligible branches
        eligible_branches = branch_customer_counts.select("BRANCH_AU").collect()
        eligible_branch_list = [row["BRANCH_AU"] for row in eligible_branches]
        
        # Create portfolios for eligible branches
        in_market_customers = distances_df.filter(
            col("BRANCH_AU").isin(eligible_branch_list)
        )
        
        # Add row numbers to create portfolio assignments
        window_spec = Window.partitionBy("BRANCH_AU").orderBy("ECN")
        
        portfolios_df = in_market_customers.withColumn(
            "row_num", row_number().over(window_spec)
        ).withColumn(
            "portfolio_num", 
            ((col("row_num") - 1) / self.max_portfolio_size).cast("int") + 1
        ).withColumn(
            "portfolio_id", 
            concat(col("BRANCH_AU"), lit("_P"), col("portfolio_num"))
        ).withColumn(
            "portfolio_type", lit("in-market")
        ).select(
            "ECN", "LAT_NUM", "LON_NUM", "BILLINGSTREET", "BILLINGCITY", "BILLINGSTATE",
            "BRANCH_AU", "portfolio_id", "portfolio_type"
        )
        
        return portfolios_df
    
    def create_centralized_portfolios(self, customer_df: DataFrame, 
                                    assigned_customers_df: DataFrame) -> DataFrame:
        """Create centralized portfolios from unassigned customers"""
        # Get unassigned customers
        unassigned_customers = customer_df.join(
            assigned_customers_df.select("ECN"), 
            on="ECN", 
            how="left_anti"
        )
        
        # Add row numbers for portfolio assignment
        window_spec = Window.orderBy("ECN")
        
        centralized_df = unassigned_customers.withColumn(
            "row_num", row_number().over(window_spec)
        ).withColumn(
            "portfolio_num", 
            ((col("row_num") - 1) / self.max_portfolio_size).cast("int") + 1
        ).withColumn(
            "portfolio_id", 
            concat(lit("CENTRALIZED_P"), col("portfolio_num"))
        ).withColumn(
            "portfolio_type", lit("centralized")
        ).withColumn(
            "BRANCH_AU", lit(None).cast("string")
        ).select(
            "ECN", "LAT_NUM", "LON_NUM", "BILLINGSTREET", "BILLINGCITY", "BILLINGSTATE",
            "BRANCH_AU", "portfolio_id", "portfolio_type"
        )
        
        return centralized_df
    
    def optimize_portfolios(self, customer_df: DataFrame, branch_df: DataFrame) -> DataFrame:
        """Main optimization function"""
        # Create in-market portfolios
        in_market_portfolios = self.create_in_market_portfolios(customer_df, branch_df)
        
        # Create centralized portfolios
        centralized_portfolios = self.create_centralized_portfolios(
            customer_df, in_market_portfolios
        )
        
        # Combine all portfolios
        all_portfolios = in_market_portfolios.union(centralized_portfolios)
        
        return all_portfolios
    
    def get_portfolio_summary(self, portfolios_df: DataFrame) -> DataFrame:
        """Generate portfolio summary statistics"""
        summary = portfolios_df.groupBy("portfolio_id", "portfolio_type", "BRANCH_AU").agg(
            count("ECN").alias("customer_count")
        ).orderBy("portfolio_type", "portfolio_id")
        
        return summary

# Usage function
def run_spark_portfolio_optimization(spark: SparkSession, customer_df: DataFrame, branch_df: DataFrame):
    """
    Run portfolio optimization using Spark DataFrames
    
    Args:
        spark: SparkSession
        customer_df: Spark DataFrame with customer data
        branch_df: Spark DataFrame with branch data
    
    Returns:
        portfolios_df: DataFrame with portfolio assignments
        summary_df: DataFrame with portfolio summaries
    """
    
    # Initialize optimizer
    optimizer = SparkPortfolioOptimizer(
        spark=spark,
        min_portfolio_size=220,
        max_portfolio_size=250,
        proximity_miles=20.0
    )
    
    # Run optimization
    portfolios_df = optimizer.optimize_portfolios(customer_df, branch_df)
    
    # Get summary
    summary_df = optimizer.get_portfolio_summary(portfolios_df)
    
    # Show summary
    print("Portfolio Summary:")
    summary_df.show(100, truncate=False)
    
    # Overall statistics
    total_stats = portfolios_df.agg(
        count("ECN").alias("total_customers"),
        countDistinct("portfolio_id").alias("total_portfolios")
    )
    
    type_stats = portfolios_df.groupBy("portfolio_type").agg(
        countDistinct("portfolio_id").alias("portfolio_count"),
        count("ECN").alias("customer_count")
    )
    
    print("\nOverall Statistics:")
    total_stats.show()
    
    print("\nBy Portfolio Type:")
    type_stats.show()
    
    return portfolios_df, summary_df

# Example usage:
"""
# Initialize Spark
spark = SparkSession.builder.appName("PortfolioOptimization").getOrCreate()

# Load data (assuming you have the DataFrames)
customer_df = spark.read.table("your_customer_table")
branch_df = spark.read.table("your_branch_table")

# Run optimization
portfolios_df, summary_df = run_spark_portfolio_optimization(spark, customer_df, branch_df)

# Save results
portfolios_df.write.mode("overwrite").saveAsTable("portfolio_assignments")
summary_df.write.mode("overwrite").saveAsTable("portfolio_summary")
"""
