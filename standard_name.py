import pandas as pd

def standardize_names(df):
    """
    Standardize names in dataframe with DIRECTOR_NAME and MANAGER_NAME columns.
    Matches names by first or last name, uses longest name as standard.
    """
    df_copy = df.copy()
    
    # Clean names
    df_copy['DIRECTOR_NAME'] = df_copy['DIRECTOR_NAME'].apply(lambda x: ' '.join(str(x).strip().split()).title() if pd.notna(x) else x)
    df_copy['MANAGER_NAME'] = df_copy['MANAGER_NAME'].apply(lambda x: ' '.join(str(x).strip().split()).title() if pd.notna(x) else x)
    
    def names_match(name1, name2):
        if pd.isna(name1) or pd.isna(name2):
            return False
        parts1, parts2 = name1.split(), name2.split()
        if not parts1 or not parts2:
            return False
        return (parts1[0].lower() == parts2[0].lower()) or (parts1[-1].lower() == parts2[-1].lower())
    
    # Step 1: Standardize Director Names
    unique_directors = df_copy['DIRECTOR_NAME'].dropna().unique()
    director_mapping = {}
    used_directors = set()
    
    for director in unique_directors:
        if director in used_directors:
            continue
        matching_directors = [director]
        used_directors.add(director)
        
        for other_director in unique_directors:
            if other_director != director and other_director not in used_directors:
                if names_match(director, other_director):
                    matching_directors.append(other_director)
                    used_directors.add(other_director)
        
        if len(matching_directors) > 1:
            longest_name = max(matching_directors, key=lambda x: len(x))
            for name in matching_directors:
                if name != longest_name:
                    director_mapping[name] = longest_name
    
    df_copy['DIRECTOR_NAME'] = df_copy['DIRECTOR_NAME'].map(director_mapping).fillna(df_copy['DIRECTOR_NAME'])
    
    # Step 2: Standardize Manager Names (with director matching)
    unique_combinations = df_copy[['MANAGER_NAME', 'DIRECTOR_NAME']].drop_duplicates()
    manager_mapping = {}
    used_combinations = set()
    
    for _, row in unique_combinations.iterrows():
        manager, director = row['MANAGER_NAME'], row['DIRECTOR_NAME']
        if pd.isna(manager) or (manager, director) in used_combinations:
            continue
        
        matching_managers = [manager]
        used_combinations.add((manager, director))
        
        for _, other_row in unique_combinations.iterrows():
            other_manager, other_director = other_row['MANAGER_NAME'], other_row['DIRECTOR_NAME']
            if (pd.isna(other_manager) or (other_manager, other_director) in used_combinations or 
                other_manager == manager or director != other_director):
                continue
            
            if names_match(manager, other_manager):
                matching_managers.append(other_manager)
                used_combinations.add((other_manager, other_director))
        
        if len(matching_managers) > 1:
            longest_name = max(matching_managers, key=lambda x: len(x))
            for name in matching_managers:
                if name != longest_name:
                    manager_mapping[name] = longest_name
    
    df_copy['MANAGER_NAME'] = df_copy['MANAGER_NAME'].map(manager_mapping).fillna(df_copy['MANAGER_NAME'])
    
    # Step 3: Check if directors appear as managers
    final_unique_directors = df_copy['DIRECTOR_NAME'].dropna().unique()
    matching_managers = df_copy['MANAGER_NAME'].dropna().unique()
    director_manager_mapping = {}
    
    for director in final_unique_directors:
        for manager in matching_managers:
            if names_match(director, manager):
                director_manager_mapping[director] = manager
                break
    
    df_copy['DIRECTOR_NAME'] = df_copy['DIRECTOR_NAME'].map(director_manager_mapping).fillna(df_copy['DIRECTOR_NAME'])
    
    return df_copy
