import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # for better plots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('student_scores.csv')

# Data cleaning and understanding
print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# --------- Step 1: Basic Visualization (EDA) -------------

# Histogram of math score
plt.figure(figsize=(8,5))
plt.hist(df['math score'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.show()

# Histogram of reading score
plt.figure(figsize=(8,5))
plt.hist(df['reading score'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribution of Reading Scores')
plt.xlabel('Reading Score')
plt.ylabel('Frequency')
plt.show()

# Histogram of writing score
plt.figure(figsize=(8,5))
plt.hist(df['writing score'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Distribution of Writing Scores')
plt.xlabel('Writing Score')
plt.ylabel('Frequency')
plt.show()

# Boxplot: Math score by Gender
plt.figure(figsize=(6,5))
sns.boxplot(x='gender', y='math score', data=df)
plt.title('Math Score by Gender')
plt.show()

# Boxplot: Math score by Lunch
plt.figure(figsize=(6,5))
sns.boxplot(x='lunch', y='math score', data=df)
plt.title('Math Score by Lunch Type')
plt.show()

# Boxplot: Math score by Test Preparation Course
plt.figure(figsize=(6,5))
sns.boxplot(x='test preparation course', y='math score', data=df)
plt.title('Math Score by Test Preparation Course')
plt.show()

# Correlation heatmap of scores
plt.figure(figsize=(6,5))
sns.heatmap(df[['math score', 'reading score', 'writing score']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Scores')
plt.show()

# --------- Step 2: Experiment with Different Feature Combinations -------------

def run_model(feature_columns, target_column='math score'):
    print(f"\nRunning model with features: {feature_columns} and target: {target_column}")
    X = df[feature_columns]
    y = df[target_column]

    # Encode categorical features if any
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Predict
    y_pred = lin_reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Plot Actual vs Predicted
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual ' + target_column)
    plt.ylabel('Predicted ' + target_column)
    plt.title(f'Linear Regression: Actual vs Predicted ({target_column})\nFeatures: {feature_columns}')
    plt.show()

# feature combinations:
# Using only gender
run_model(['gender'])

# Using gender + test preparation course
run_model(['gender', 'test preparation course'])

# Using all categorical features (except scores)
all_features = df.drop(['math score', 'reading score', 'writing score'], axis=1).columns.tolist()
run_model(all_features)

# --------- Step 3: Train Model to Estimate Final Score -------------

# Create final_score column as average of three scores
df['final_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Use all categorical features for prediction of final_score
X_final = df.drop(['math score', 'reading score', 'writing score', 'final_score'], axis=1)
y_final = df['final_score']

# Encode categorical features
X_final_encoded = pd.get_dummies(X_final, drop_first=True)

# Split data
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_final_encoded, y_final, test_size=0.2, random_state=42)

# Train Linear Regression on final_score
lin_reg_final = LinearRegression()
lin_reg_final.fit(X_train_f, y_train_f)

# Predict
y_pred_f = lin_reg_final.predict(X_test_f)

# Evaluate
print("\nLinear Regression Performance for Final Score:")
print(f"Mean Squared Error: {mean_squared_error(y_test_f, y_pred_f):.2f}")
print(f"R^2 Score: {r2_score(y_test_f, y_pred_f):.2f}")

# Plot Actual vs Predicted final_score
plt.scatter(y_test_f, y_pred_f, color='blue', alpha=0.6)
plt.plot([y_final.min(), y_final.max()], [y_final.min(), y_final.max()], 'r--')
plt.xlabel('Actual Final Score')
plt.ylabel('Predicted Final Score')
plt.title('Linear Regression: Actual vs Predicted Final Scores')
plt.show()
