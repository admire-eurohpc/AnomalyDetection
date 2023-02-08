import numpy as np
import pandas as pd
from typing import List, Dict, Any

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

