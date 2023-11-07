import pytest 
import pandas as pd
import numpy as np
from ..metrics import threshold_data_by_value, calculate_z_score, get_nth_percentile

def test_calculate_z_score_CheckCorrectShape():
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z_score = calculate_z_score(data.values)
    assert z_score.shape == (3, 3), "z_score shape is not correct"
    
def test_calculate_z_score_CorrectValue():
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z_score = calculate_z_score(data.values)
    assert pytest.approx(z_score[0][0], 0.0001) == 1.2247, "z_score value is not correct, our function takes absolute value of the z-score, but negative was returned"
    
def test_get_nth_percentile_CheckCorrectShape():
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    percentile = get_nth_percentile(data.values, 99)
    assert percentile.shape == (3,), "percentile shape is not correct"
    
def test_get_nth_percentile_CorrectValue():
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    percentile = get_nth_percentile(data.values, 99)
    assert pytest.approx(percentile[0], 0.0001) == 2.98, "percentile value is not correct"
    
def test_threshold_data_by_value_CheckCorrectShape():
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    thresholded_data = threshold_data_by_value(data.values, 2)
    assert thresholded_data.shape == (3, 3), "thresholded_data shape is not correct"
    
def test_threshold_data_by_value_CorrectValue():
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    thresholded_data = threshold_data_by_value(data.values, 2)
    assert thresholded_data[0][0] == 0, "thresholded_data value is not correct"
    