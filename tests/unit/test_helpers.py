import pytest
import pandas as pd
from src.utils.helpers import check_missing_values, balance_dataset

def test_check_missing_values():
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [4, None, 6]
    })
    result = check_missing_values(df)
    
    assert result.loc['A', 'Missing Values'] == 1
    assert result.loc['B', 'Percentage'] == pytest.approx(33.333, 0.001)

def test_balance_dataset():
    X = pd.DataFrame({'feature': range(100)})
    y = pd.Series([0]*90 + [1]*10)
    
    X_bal, y_bal = balance_dataset(X, y, method='smote')
    
    assert sum(y_bal == 1) > 10
    assert len(X_bal) == len(y_bal)