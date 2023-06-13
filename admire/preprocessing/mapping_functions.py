import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

def not_dict_or_list(object: Any) -> bool:
    '''
    Return True if `object` is not dict or list.
    '''
    return type(object) is not list and type(object) is not dict

def check_if_terminal(object: Any) -> bool:
    ''' 
    Return True if `object` seems like its terminal variable. 

    *Assumption*: terminals are any objects that aren't of type 
    list or dict and their length is 0.
    '''
    if type(object) is list and len(object) > 0:
        return False
    
    if type(object) is dict and len(object.keys()) > 0:
        return False
    return True
    
def DFS_unravel_list(lst: List[Any], prefix_name: str) -> List[str]:
    '''
    DFS search for populating col_names global list. 
    '''
    colnames = []
    # if list is terminal then do not expand further. e.g. list of nodes ['e123', 'e324']
    if all([check_if_terminal(x) for x in lst]):
        colnames.append(prefix_name) 
    else:
        for index, value in enumerate(lst):
            # Create next name in standardized format
            new_name = prefix_name + '-#' + str(index)
            # If list then unravel as list
            if type(value) is list:
                colnames += DFS_unravel_list(value, new_name)
            # If dict unravel as dict
            elif type(value) is dict:
                colnames += DFS_unravel_dict(value, new_name)
            # Should not end up here, but for just in case will print error
            else:
                print('LIST???')
    return colnames
                 
def DFS_unravel_dict(dic: Dict[str, Any], prefix_name: str) -> List[str]:
    ''' 
    DFS search for populating col_names global list.  

    Pass dictionary to be mapped from nested form 
    into standatd dash and hashtag separated format.
    '''
    colnames = []
    for index, value in dic.items():
        new_name = prefix_name + '-' + str(index)
        if type(value) is list:
            colnames += DFS_unravel_list(value, new_name)
        elif type(value) is dict:
            colnames += DFS_unravel_dict(value, new_name)
        else:
            if not_dict_or_list(value):
                colnames.append(new_name)
            else:
                print('DICT???')
    return colnames

def get_mapping_for_dict(dictionary: Dict[str, Any], col_name: str) -> List[str]:
    '''
    Obtain list of map_strings that map the dictionary into its specific fields.\n
    Format of mapping is simple -> [col_name]-[dict_key]-[dict_key]-#[list_id]-[dict_key].\n
    e.g. steps-nodes-list, steps-tres-requested-max-#0-type.
    '''
    colnames = DFS_unravel_dict(dictionary, col_name)
    return colnames

def extract_by_map_str(dictionary: Dict[str, Any], map_string: str) -> Any:
    ''' 
    Extract value from dictionary by following path specified by mapping string.  \n
    If path cannot be reached then returned value should be np.nan.
    '''
    map_steps = map_string.split('-')
    # extract value, first name in split is the overall colname
    value = dictionary
    for step in map_steps[1:]:
        if type(value) is list and step[0] == '#': # hashtag means that we map to the list index 
            index = int(step[1:]) 
            # If index is not in list return np.nan
            if len(value) <= index:
                print(f'Index do not match length of {value}')
                return np.nan
            # Else proceed further with the list
            else:
                value = value[index]
        else:
            if type(value) is not dict or step not in value.keys():
                print(f'{step} not in dict or. Value is not dict, but should be. Value is of type {type(value)}: {value}')
                return np.nan
            else:
                value = value[step] # go into dictionary field

    return value

def flatten_series_by_mapping(series: pd.Series, mapping: List[str]) -> pd.DataFrame:
    '''
    Flatten specific dataframe column (series) into DataFrame of unnested values according to mapping.  

    For example: \n
    ```txt
        pass df['steps'] (stripped previously to one dimension depth like df = df['steps].apply(lambda x: x[0]))  
        and for mapping list of map strings like ["steps-nodes-list", "steps-tres-requested-max-#0-type"].   
        Return value for this example should be DataFrame with map_strings as columns and extracted values in body. 
    ```
    '''
    flat = list()
    # Iterative approach - probably slow, but should be sufficent for now
    for row_idx, dictionary in series.items():
        row = dict()
        for map_string in mapping:
            row[map_string] = extract_by_map_str(dictionary, map_string)
        flat.append(row)
    return pd.DataFrame(flat)

def extract_nodes_from_node(node_string: str) -> List:
    '''Extract specific nodes from node_string. E.g. e[123-125] -> [e123,e124,e125]'''
    return_nodes = list()

    if '[' in node_string:
        comma_split = node_string[2:len(node_string)-1].split(',')
        for node in comma_split:
            if '-' in node:
                range_low, range_up = node.split('-')
                for i in range(int(range_low), int(range_up) + 1, 1):
                    return_nodes.append(f'e{i}')
            else:
                return_nodes.append(f'e{node}')
    else:
        return_nodes.append(node_string)
    
    return return_nodes

def transform_all_nodes(col_data: pd.Series) -> Tuple[List, List]:
    '''
    Extract exact nodes from the raw nodes column containing slurm 
    node-string representation.

    Return: (`split_nodes`, `all_count`)
    `split_nodes`: nodes converted from slurm string representation  
    into specific nodes separated by comma. The order of rows is untouched.
    `all_count`: one big list of all nodes extracted from the data column. 
    '''
    all_count = list()
    split_nodes = list()

    for node_String in col_data.to_list():
        nodes = extract_nodes_from_node(node_String)
        all_count += nodes
        split_nodes.append(nodes)
    return split_nodes, all_count

def merge_type_and_counts(column_core: str, df: pd.DataFrame) -> pd.DataFrame:
    '''
    Merge types and counts into one column where colname 
    is of specific type and values are corresponding counts
    '''    
    unique_values = df[f'{column_core}-type'].unique()
    name_after = f'{column_core}-{unique_values[0]}'

    if unique_values.shape[0] == 1:
        counts = df[f'{column_core}-count'].to_numpy()
    else:
        print(f'More than one value. Be careful: {column_core}-type, {unique_values}')

    # Some of the records have -node field which we want to keep for now
    if f'{column_core}-node' in df.columns:
        nodes = df[f'{column_core}-node'].to_numpy()
        return pd.DataFrame({name_after: counts, f'{name_after}-node': nodes})

    #print(df_flat[f'{column_core}-{unique_values[0]}'].value_counts())
    return pd.DataFrame(data=counts, columns=[name_after])

def merge_all_tres_possible(df: pd.DataFrame) -> pd.DataFrame:
    '''Extract types and counts from tres fields and use them as new columns'''
    dataframes_to_merge = []
    columns_to_drop = []

    column_cores_to_go_over = []

    # Add steps-tres requested and consumed paths as they share the same colpath type
    for what in ['requested', 'consumed']:
        for aggregation_type in ['max', 'min', 'average', 'total']:
            for i in range(6):
                to_merge = f'steps-tres-{what}-{aggregation_type}-#{i}'
                if f'{to_merge}-type' in df.columns:
                    column_cores_to_go_over.append(to_merge)
                else:
                    print(f'Colpath does not exist! Make sure that it is okay: "{to_merge}-type"')

    # Add step-tres-allocated paths
    for i in range(6):
        to_merge = f'steps-tres-allocated-#{i}'
        if f'{to_merge}-type' in df.columns:
            column_cores_to_go_over.append(to_merge)
        else:
            print(f'Colpath does not exist! Make sure that it is okay: "{to_merge}-type"')

    # Add tres requested and consumed paths as they share the same colpath type
    for what in ['allocated', 'requested']:
        for i in range(6):
            to_merge = f'tres-{what}-#{i}'
            if f'{to_merge}-type' in df.columns:
                column_cores_to_go_over.append(to_merge)
            else:
                print(f'Colpath does not exist! Make sure that it is okay: "{to_merge}-type"')

    print(column_cores_to_go_over)
    # Perform merge operations on designated columns
    for col in column_cores_to_go_over:
        coldf = merge_type_and_counts(col, df)
        columns_to_drop.append(f'{col}-type')
        columns_to_drop.append(f'{col}-name')
        columns_to_drop.append(f'{col}-id')
        columns_to_drop.append(f'{col}-count')
        if 'min' in col or 'max' in col:
            columns_to_drop.append(f'{col}-task')
            columns_to_drop.append(f'{col}-node')
        dataframes_to_merge.append(coldf)

    df = pd.concat([df] + dataframes_to_merge, axis='columns')
    df = df.drop(columns=columns_to_drop)

    #print(df.columns.to_list())

    return df 

def remove_index_element_from_column_names(_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove "#id" part from column names. 
    Should be used only after flattening and merging type-counts operations.
    '''
    rename_map = {}
    for old in _df.columns.to_list():
        new = '-'.join(filter(lambda x: '#' not in x, old.split('-')))
        rename_map[old] = new

    _df = _df.rename(columns=rename_map)
    return _df
    
