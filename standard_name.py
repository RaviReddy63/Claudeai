import pandas as pd

def standardize_names(df, manager_col='MANAGER_NAME', director_col='DIRECTOR_NAME'):
    """
    Standardize names by matching first or last names
    """
    df_copy = df.copy()
    
    def get_name_parts(name):
        if pd.isna(name):
            return "", ""
        parts = str(name).strip().title().split()
        if len(parts) >= 2:
            return parts[0], parts[-1]
        return parts[0] if parts else "", ""
    
    def find_matches(names):
        name_map = {}
        name_list = list(set([n for n in names if pd.notna(n)]))
        
        for i, name1 in enumerate(name_list):
            if name1 in name_map:
                continue
            first1, last1 = get_name_parts(name1)
            matches = [name1]
            
            for name2 in name_list[i+1:]:
                if name2 in name_map:
                    continue
                first2, last2 = get_name_parts(name2)
                
                if (first1 and first1 == first2) or (last1 and last1 == last2):
                    matches.append(name2)
            
            if len(matches) > 1:
                standard = max(matches, key=lambda x: len(x.split()))
                for match in matches:
                    name_map[match] = standard
        
        return name_map
    
    # Standardize directors
    director_map = find_matches(df_copy[director_col])
    for old, new in director_map.items():
        df_copy[director_col] = df_copy[director_col].replace(old, new)
    
    # Standardize managers within each director group
    for director in df_copy[director_col].dropna().unique():
        director_df = df_copy[df_copy[director_col] == director]
        manager_map = find_matches(director_df[manager_col])
        for old, new in manager_map.items():
            mask = df_copy[director_col] == director
            df_copy.loc[mask, manager_col] = df_copy.loc[mask, manager_col].replace(old, new)
    
    return df_copy

# Example usage:
# df_standardized = standardize_names(df)
