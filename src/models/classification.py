import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_random_forest(X, y):
    """
    Train a Random Forest Classifier.
    1. Split the data into training and testing sets
    2. Train the model
    3. Evaluate the model
    """
    X_train, X_test, y_train, y_test = RandomForestClassifier(X, y, test_size=0.3, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return rf, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """ 
    Evalute model and returns metrics
    """
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    return {
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred)
    }

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance of the model
    """
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance in Random Forest')
    plt.show()

    return feature_importance