import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

np.random.seed(42)
n_clients = 1000

data = {
    'client_id': range(1, n_clients + 1),
    'subscription-time': np.random.randint(1, 37, n_clients),
    # 1 month to 3 years
    'plan-type': np.random.choice(['Basic', 'Premium', 'Gold'], n_clients),
    'use-service': np.random.normal(50, 15, n_clients).round(1),
    'call_sac': np.random.poisson(5, n_clients),
    'client-satisfaction': np.random.randint(1, 6, n_clients),
    # 1 = very dissatisfied, 5 = very satisfied
    'complaints': np.random.randint(0, 10, n_clients),
    'churn': np.random.choice([0, 1], n_clients, p=[0.7, 0.3])
    # 0 = no churn, 1 = churn
}

df = pd.DataFrame(data)
df['subscription-time'] = df['subscription-time'].apply(lambda x: f"{x} months" if x > 1 else f"{x} month")