import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Pre processing the data for model training.
    1. Remove unnecessary columns
    2. Convert churn to binary (0, 1)
    3. Create dummy variables for plan-type
    4. Select features and handle missing values
    """
    df['churn'] = df['churn'].map({'yes': 1, 'no': 0})
    df['churn'] = df['churn'].fillna(0)
    
    df = pd.get_dummies(df, columns=['plan-type'], drop_first=True)
    
    features = get_features(df)
    df[features] = df[features].fillna(df[features].mean())
    
    return df

def get_features(df):
    """Return the list of features to be used for model training."""
    base_features = ['subscription-time', 'use-service', 'call_sac', 
                   'client-satisfaction', 'complaints']
    plan_type_features = [col for col in df.columns if 'plan-type_' in col]
    return base_features + plan_type_features