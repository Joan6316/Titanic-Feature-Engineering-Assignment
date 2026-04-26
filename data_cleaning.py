import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/train.csv')

# --- 1. Missing Value Handling ---
print(df.isnull().sum())

# Age: Impute with Median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Embarked: Impute with Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Cabin: Too many missing, create Deck feature or drop
df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U') # U for Unknown
df.drop('Cabin', axis=1, inplace=True)

# --- 2. Outlier Handling ---
# Cap Fare outliers using IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Fare'] = np.where(df['Fare'] < lower_bound, lower_bound, df['Fare'])
df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, df['Fare'])

# --- 3. Data Consistency ---
# Check for duplicates
df.drop_duplicates(inplace=True)

# Check unique values in Sex
df['Sex'] = df['Sex'].str.lower() # Ensure consistency

# Save cleaned data
df.to_csv('data/train_cleaned.csv', index=False)