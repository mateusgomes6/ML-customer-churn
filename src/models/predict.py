import numpy as np
import pandas as pd

def predict_churn(sample_data, model, kmeans, scaler, feature_names):
    """
    Predict churn probability and class for a given sample data.
    """
    sample_data = sample_data.copy()
    sample_data = pd.get_dummies(sample_data, columns=['plan-type'], drop_first=True)  

    for col in feature_names:
        if col not in sample_data.columns and not col.startswith('plan-type_'):
            raise ValueError(f"Missing required feature: {col}")
        elif col.startswith('plan-type_') and col not in sample_data.columns:
            sample_data[col] = 0
    
    # Scale features
    sample_data = sample_data[feature_names]
    sample_scaled = scaler.transform(sample_data)

    # Predict cluster
    clusters = kmeans.predict(sample_scaled)
    sample_with_cluster = np.hstack([sample_scaled, clusters.reshape(-1, 1)])

    try:
        probabilities = model.predict_proba(sample_with_cluster)[:, 1]
        predictions = model.predict(sample_with_cluster)
    except IndexError:
        predictions = model.predict(sample_with_cluster)
        probabilities = np.where(predictions == 1, 0.99, 0.01)
    
    return probabilities, predictions



