import pandas as pd

# Assuming your dataframe is called 'df'
# Calculate centroids for each portfolio
centroids = df.groupby('CG_PORTFOLIO_CD')[['LAT_NUM', 'LON_NUM']].mean().reset_index()
centroids.columns = ['CG_PORTFOLIO_CD', 'centroid_lat', 'centroid_lon']

# Merge back to original dataframe
df_with_centroids = df.merge(centroids, on='CG_PORTFOLIO_CD', how='left')

print(df_with_centroids.head())
