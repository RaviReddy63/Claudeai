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
                # Find the name with most characters as standard (longest name)
                standard = matches[0]
                max_length = len(matches[0])
                for match in matches:
                    if len(match) > max_length:
                        standard = match
                        max_length = len(match)
                
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
            mask = (df_copy[director_col] == director) & (df_copy[manager_col] == old)
            df_copy.loc[mask, manager_col] = new
    
    # Handle null director cases - standardize managers globally where director is null
    null_director_df = df_copy[df_copy[director_col].isna()]
    if not null_director_df.empty:
        manager_map = find_matches(null_director_df[manager_col])
        for old, new in manager_map.items():
            mask = df_copy[director_col].isna() & (df_copy[manager_col] == old)
            df_copy.loc[mask, manager_col] = new
    
    # LAST STEP: Handle cases where director name appears in manager name - make them same
    for idx, row in df_copy.iterrows():
        if pd.notna(row[director_col]) and pd.notna(row[manager_col]):
            dir_first, dir_last = get_name_parts(row[director_col])
            mgr_first, mgr_last = get_name_parts(row[manager_col])
            
            # Check if full director name matches full manager name (both first AND last name)
            if (dir_first and dir_last and mgr_first and mgr_last and 
                dir_first == mgr_first and dir_last == mgr_last):
                # Use the longer version as standard
                if len(row[director_col]) >= len(row[manager_col]):
                    df_copy.loc[idx, manager_col] = row[director_col]
                else:
                    df_copy.loc[idx, director_col] = row[manager_col]
    
    return df_copy

# Example usage:
# df_standardized = standardize_names(df)
