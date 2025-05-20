from src.data.preprocessing import get_features

def test_preprocess_data_churn_conversion(preprocessed_data):
    assert set(preprocessed_data['churn'].unique()).issubset({0, 1})
    assert preprocessed_data['churn'].isna().sum() == 0

def test_preprocess_data_dummies_creation(preprocessed_data):
    assert 'plan-type_premium' in preprocessed_data.columns
    assert preprocessed_data['plan-type_premium'].dtype == 'uint8'

def test_get_features(preprocessed_data):
    features = get_features(preprocessed_data)
    assert 'subscription-time' in features
    assert 'plan-type_premium' in features
    assert 'churn' not in features