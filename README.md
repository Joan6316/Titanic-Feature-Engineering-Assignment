# Titanic Survival Prediction

## Project Overview

This project analyzes the Titanic dataset to predict passenger survival. We performed data cleaning, feature engineering, and feature selection to build an effective model.

## Project Structure

- `data/`: Contains raw and cleaned datasets.
- `notebooks/`: Jupyter notebook with exploration and visualization.
- `scripts/`: Python scripts for each stage of the pipeline.

## Approach

### 1. Data Cleaning

- **Missing Values:** Age imputed with median, Embarked with mode.
- **Cabin:** Extracted Deck information.
- **Outliers:** Fare values were capped using IQR method.

### 2. Feature Engineering

- Created `FamilySize`, `IsAlone`, `Title`, `AgeGroup`, `FarePerPerson`.
- Extracted social status from names (Mr, Mrs, Miss, etc.).
- Applied log transformation to skewed features.
- One-hot encoded categorical variables.

### 3. Feature Selection

- Used Correlation Matrix to remove multicollinearity.
- Used Random Forest to measure feature importance.
- Selected only significant features to prevent overfitting.

## Key Findings

- **Gender** was the strongest predictor: females had higher survival rate.
- **Class** mattered: First class passengers survived more.
- **Title** reflected social status and age, helping predictions.
- Being alone or with large family affected survival chances.

## How to Run

```bash
pip install -r requirements.txt
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```
