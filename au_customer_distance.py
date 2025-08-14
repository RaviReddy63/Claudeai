import pandas as pd
import numpy as np

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation in miles"""
    R = 3959  # Earth's radius in miles
    
    # Convert to numpy arrays if not already and ensure proper dtype
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical errors
    
    distance = R * c
    
    # Handle scalar case
    if np.isscalar(distance):
        return float(distance)
    return distance

def calculate_distances_au_to_ecns(au_number, ecn_list, customer_csv_path='customer_data.csv', branch_csv_path='branch_data.csv'):
    """
    Calculate haversine distance between a specific AU and list of ECNs
    
    Parameters:
    au_number (str): The AU number to calculate distances from
    ecn_list (list): List of ECN numbers
    customer_csv_path (str): Path to customer data CSV file
    branch_csv_path (str): Path to branch data CSV file
    
    Returns:
    pandas.DataFrame: DataFrame with ECN, customer details, and distance to AU
    """
    
    try:
        # Read the CSV files
        print("Reading customer data...")
        customer_df = pd.read_csv(customer_csv_path)
        
        print("Reading branch data...")
        branch_df = pd.read_csv(branch_csv_path)
        
        # Find the AU coordinates
        au_data = branch_df[branch_df['BRANCH_AU'] == au_number]
        
        if len(au_data) == 0:
            print(f"Error: AU {au_number} not found in branch data")
            return None
        
        au_lat = float(au_data.iloc[0]['BRANCH_LAT_NUM'])
        au_lon = float(au_data.iloc[0]['BRANCH_LON_NUM'])
        
        print(f"AU {au_number} coordinates: ({au_lat}, {au_lon})")
        
        # Filter customers for the given ECNs
        customer_subset = customer_df[customer_df['ECN'].isin(ecn_list)].copy()
        
        if len(customer_subset) == 0:
            print("Error: No customers found for the given ECNs")
            return None
        
        print(f"Found {len(customer_subset)} customers out of {len(ecn_list)} ECNs provided")
        
        # Check for customers without coordinates
        customers_with_coords = customer_subset.dropna(subset=['LAT_NUM', 'LON_NUM'])
        customers_without_coords = customer_subset[
            customer_subset['LAT_NUM'].isna() | customer_subset['LON_NUM'].isna()
        ]
        
        print(f"Customers with coordinates: {len(customers_with_coords)}")
        print(f"Customers without coordinates: {len(customers_without_coords)}")
        
        results = []
        
        # Calculate distances for customers with coordinates
        if len(customers_with_coords) > 0:
            for idx, customer in customers_with_coords.iterrows():
                customer_lat = float(customer['LAT_NUM'])
                customer_lon = float(customer['LON_NUM'])
                
                # Calculate haversine distance
                distance = haversine_distance_vectorized(
                    customer_lat, customer_lon, au_lat, au_lon
                )
                
                results.append({
                    'ECN': customer['ECN'],
                    'BILLINGCITY': customer.get('BILLINGCITY', 'N/A'),
                    'BILLINGSTATE': customer.get('BILLINGSTATE', 'N/A'),
                    'CUSTOMER_LAT': customer_lat,
                    'CUSTOMER_LON': customer_lon,
                    'AU_NUMBER': au_number,
                    'AU_LAT': au_lat,
                    'AU_LON': au_lon,
                    'DISTANCE_MILES': round(distance, 2),
                    'HAS_COORDINATES': True
                })
        
        # Add customers without coordinates
        if len(customers_without_coords) > 0:
            for idx, customer in customers_without_coords.iterrows():
                results.append({
                    'ECN': customer['ECN'],
                    'BILLINGCITY': customer.get('BILLINGCITY', 'N/A'),
                    'BILLINGSTATE': customer.get('BILLINGSTATE', 'N/A'),
                    'CUSTOMER_LAT': None,
                    'CUSTOMER_LON': None,
                    'AU_NUMBER': au_number,
                    'AU_LAT': au_lat,
                    'AU_LON': au_lon,
                    'DISTANCE_MILES': None,
                    'HAS_COORDINATES': False
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by distance (customers with coordinates first, then by distance)
        results_df = results_df.sort_values(['HAS_COORDINATES', 'DISTANCE_MILES'], 
                                          ascending=[False, True])
        
        return results_df
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def print_distance_summary(results_df):
    """Print a summary of the distance calculations"""
    if results_df is None or len(results_df) == 0:
        print("No results to summarize")
        return
    
    print(f"\n=== DISTANCE CALCULATION SUMMARY ===")
    print(f"Total ECNs processed: {len(results_df)}")
    
    with_coords = results_df[results_df['HAS_COORDINATES'] == True]
    without_coords = results_df[results_df['HAS_COORDINATES'] == False]
    
    print(f"ECNs with coordinates: {len(with_coords)}")
    print(f"ECNs without coordinates: {len(without_coords)}")
    
    if len(with_coords) > 0:
        print(f"\nDistance Statistics (for ECNs with coordinates):")
        print(f"  Minimum distance: {with_coords['DISTANCE_MILES'].min():.2f} miles")
        print(f"  Maximum distance: {with_coords['DISTANCE_MILES'].max():.2f} miles")
        print(f"  Average distance: {with_coords['DISTANCE_MILES'].mean():.2f} miles")
        print(f"  Median distance: {with_coords['DISTANCE_MILES'].median():.2f} miles")

# Example usage:
def main():
    """
    Example usage of the distance calculator
    Replace with your actual AU number and ECN list
    """
    
    # Example inputs - replace these with your actual values
    au_number = "YOUR_AU_NUMBER"  # Replace with actual AU number
    ecn_list = ["ECN1", "ECN2", "ECN3"]  # Replace with actual ECN list
    
    # Calculate distances
    results = calculate_distances_au_to_ecns(au_number, ecn_list)
    
    if results is not None:
        # Print summary
        print_distance_summary(results)
        
        # Display first few results
        print(f"\nFirst 10 results:")
        print(results.head(10).to_string(index=False))
        
        # Save results to CSV
        output_filename = f"distances_AU_{au_number}.csv"
        results.to_csv(output_filename, index=False)
        print(f"\nResults saved to: {output_filename}")
        
        return results
    else:
        print("Failed to calculate distances")
        return None

if __name__ == "__main__":
    # To use this script, update the AU number and ECN list in the main() function
    # and run: python distance_calculator.py
    print("Distance Calculator Ready!")
    print("Update the main() function with your AU number and ECN list, then run the script.")
    
    # Uncomment the line below to run with example data
    # main()
