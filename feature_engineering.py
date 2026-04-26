import pandas as pd

df = pd.read_csv('data/train_cleaned.csv')

# --- 1. Create Derived Features ---

# Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Is Alone
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

# Title Extraction
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Group rare titles
rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Sir', 'Capt', 'Countess', 'Don', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Age Groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Senior', 'Elderly'])

# Fare per Person
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# --- 2. Encoding Categorical Variables ---

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup'], drop_first=True)

# Ordinal encoding for Pclass (already numerical, but we can keep as is)

# --- 3. Feature Transformation ---

# Log transform Fare (since it is skewed)
df['Fare_log'] = np.log1p(df['Fare'])

# Standardization (Optional)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Save final features
df.to_csv('data/train_features.csv', index=False)