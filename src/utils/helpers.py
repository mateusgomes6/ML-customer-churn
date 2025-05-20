import pandas as pd
import numpy as np
from typing import Union, List

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing values ​​and returns a DataFrame with the count and percentage

    Arguments:
    df: DataFrame to be analyzed

    Returns:
    DataFrame with columns ['Missing Values', 'Percentage']
    """
    missing = df.isnull().sum()
    percent = (missing / df.shape[0]) * 100
    return pd.concat([missing, percent], axis=1, 
                    keys=['Missing Values', 'Percentage'])

def balance_dataset(X: Union[pd.DataFrame, np.ndarray], 
                   y: Union[pd.Series, np.ndarray],
                   method: str = 'smote') -> tuple:
    """
    Balance the dataset using SMOTE or undersampling
    
    Args:
        X: Features
        y: Target
        method: 'smote' or 'undersample'
        
    Returns:
        Tuple with (X_balanced, y_balanced)
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    if method == 'smote':
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)
    elif method == 'undersample':
        undersampler = RandomUnderSampler(random_state=42)
        return undersampler.fit_resample(X, y)
    else:
        raise ValueError("Método deve ser 'smote' ou 'undersample'")

def save_model(model, filepath: str):
    """
    Balance the dataset using SMOTE or undersampling
    
    Args:
        X: Features
        y:Target
        method: 'smote' or 'undersample'
        
    Returns:
        Tuple with (X_balanced, y_balanced)
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str):
    """
    Balance the dataset using SMOTE or undersampling
    
    Args:
        X: Features
        y:Target
        method: 'smote' or 'undersample'
        
    Returns:
        Tuple with (X_balanced, y_balanced)
    """
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)