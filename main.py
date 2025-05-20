import pandas as pd
from src.data.preprocessing import load_data, preprocess_data, get_features
from src.data.visualization import plot_churn_distribution, plot_correlation_matrix
from src.models.clustering import perform_clustering, plot_elbow_method, plot_clusters_pca
from src.models.classification import train_random_forest, evaluate_model, plot_feature_importance
from src.models.predict import predict_churn
from dataframe import Data 

def main():
    # 1. Load and preprocess data
    df = Data
    df = preprocess_data(df)
    features = get_features(df)
    
    # 2. Visualization
    plot_churn_distribution(df)
    plot_correlation_matrix(df)
    
    # 3. Clustering
    X_cluster = df[features]
    clusters, kmeans, scaler = perform_clustering(X_cluster)
    df['cluster'] = clusters
    plot_clusters_pca(scaler.transform(X_cluster), clusters)
    
    # 4. Model training
    X = df[features + ['cluster']]
    y = df['churn'].astype(int)
    
    rf, X_test, y_test = train_random_forest(X, y)
    evaluate_model(rf, X_test, y_test)
    feature_importance = plot_feature_importance(rf, X.columns)
    
    # 5. Prediction Sample
    sample_data = pd.DataFrame({
        'subscription-time': [12, 24, 6],
        'use-service': [200, 150, 300],
        'call_sac': [2, 0, 5],
        'client-satisfaction': [3, 4, 2],
        'complaints': [1, 0, 3],
        'plan-type': ['premium', 'basic', 'basic']
    })
    
    probabilities, predictions = predict_churn(sample_data, rf, kmeans, scaler, features)
    
    sample_data['churn_probability'] = probabilities
    sample_data['churn_prediction'] = predictions
    sample_data['churn_prediction'] = sample_data['churn_prediction'].map({1: 'yes', 0: 'no'})
    
    print("\nPrevisões de Churn para Dados de Exemplo:")
    print(sample_data[['subscription-time', 'use-service', 'churn_probability', 'churn_prediction']])

if __name__ == "__main__":
    main()