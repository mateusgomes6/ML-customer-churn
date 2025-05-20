import pytest
from src.models.clustering import perform_clustering
from src.models.classification import train_random_forest

@pytest.mark.integration
def test_full_pipeline(preprocessed_data):
    from src.data.preprocessing import get_features
    
    features = get_features(preprocessed_data)
    X = preprocessed_data[features]
    y = preprocessed_data['churn']
    
    clusters, kmeans, scaler = perform_clustering(X)
    preprocessed_data['cluster'] = clusters
    
    X_model = preprocessed_data[features + ['cluster']]
    model, X_test, y_test = train_random_forest(X_model, y)
    
    assert hasattr(model, 'predict')
    assert X_test.shape[0] > 0
    assert y_test.shape[0] == X_test.shape[0]