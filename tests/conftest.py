import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'subscription-time': [12, 24, 6],
        'use-service': [200, 150, 300],
        'call_sac': [2, 0, 5],
        'client-satisfaction': [3, 4, 2],
        'complaints': [1, 0, 3],
        'plan-type': ['premium', 'basic', 'basic'],
        'churn': [0, 1, 0]
    })

@pytest.fixture
def preprocessed_data(sample_data):
    from src.data.preprocessing import preprocess_data
    return preprocess_data(sample_data.copy())