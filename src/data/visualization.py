import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    """Plot the distribution of churn for Call SAC"""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='churn', hue='call_sac', data=df)
    plt.title('Relationship between Call SAC and Churn')
    plt.show()

def plot_correlation_matrix(df):
    """Plota correlation matrix for the features"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()