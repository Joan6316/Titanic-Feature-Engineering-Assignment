import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/train_features.csv')

# Define features and target
X = df.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1, errors='ignore')
y = df['Survived']

# --- 1. Correlation Analysis ---
corr_matrix = X.corr()
# Drop features with correlation > 0.9 (multicollinearity)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X.drop(to_drop, axis=1, inplace=True)

# --- 2. Feature Importance using Random Forest ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Print top features
print("Top 10 Important Features:")
print(importances.head(10))

# Select top features
selected_features = importances[importances > 0.02].index.tolist()
print("\nSelected Features for Model:")
print(selected_features)