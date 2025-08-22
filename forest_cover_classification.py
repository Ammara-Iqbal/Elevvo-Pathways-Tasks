#!/usr/bin/env python3
"""
Forest Cover Type Classification
Dataset: Covertype (UCI)
Predict forest cover type based on cartographic and environmental features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("=== Forest Cover Type Classification ===\n")

def load_and_preprocess_data():
    """Load and preprocess the Covertype dataset"""
    print("Loading and preprocessing data...")
    
    # Column names as per dataset description
    column_names = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    
    # Add wilderness area columns (4 binary features)
    wilderness_areas = [f'Wilderness_Area{i}' for i in range(1, 5)]
    column_names.extend(wilderness_areas)
    
    # Add soil type columns (40 binary features)
    soil_types = [f'Soil_Type{i}' for i in range(1, 41)]
    column_names.extend(soil_types)
    
    # Add target variable
    column_names.append('Cover_Type')
    
    try:
        # Try to load from local file
        data = pd.read_csv('covtype.data', header=None, names=column_names)
        print("Loaded data from local file 'covtype.data'")
    except FileNotFoundError:
        # If local file not found, download from URL
        print("Local file not found. Downloading from UCI repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        data = pd.read_csv(url, header=None, names=column_names, compression='gzip')
        data.to_csv('covtype.data', index=False)
        print("Dataset downloaded and saved as 'covtype.data'")
    
    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution:\n{data['Cover_Type'].value_counts().sort_index()}")
    
    return data

def exploratory_data_analysis(data):
    """Perform exploratory data analysis"""
    print("\n=== Exploratory Data Analysis ===\n")
    
    # Basic information
    print("Dataset Info:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    print("\nMissing values:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        print("No missing values found!")
    
    print("\nDescriptive statistics for numerical features:")
    numerical_features = data.columns[:10]  # First 10 are numerical
    print(data[numerical_features].describe())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Target distribution
    cover_type_counts = data['Cover_Type'].value_counts().sort_index()
    axes[0, 0].bar(cover_type_counts.index, cover_type_counts.values, color='skyblue')
    axes[0, 0].set_title('Cover Type Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Cover Type')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(cover_type_counts.values):
        axes[0, 0].text(i + 1, v + 1000, str(v), ha='center', va='bottom')
    
    # Correlation heatmap (first 10 numerical features)
    corr_matrix = data[numerical_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
    axes[0, 1].set_title('Correlation Matrix (Numerical Features)', fontsize=14, fontweight='bold')
    
    # Distribution of elevation
    axes[1, 0].hist(data['Elevation'], bins=50, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Elevation Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Elevation (feet)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Cover type by elevation
    box_data = [data[data['Cover_Type'] == i]['Elevation'] for i in range(1, 8)]
    axes[1, 1].boxplot(box_data)
    axes[1, 1].set_title('Elevation by Cover Type', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Cover Type')
    axes[1, 1].set_ylabel('Elevation (feet)')
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Aspect distribution (circular data)
    axes[0].hist(data['Aspect'], bins=36, alpha=0.7, color='orange')
    axes[0].set_title('Aspect Distribution (0-360 degrees)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Aspect (degrees)')
    axes[0].set_ylabel('Frequency')
    
    # Slope distribution
    axes[1].hist(data['Slope'], bins=30, alpha=0.7, color='purple')
    axes[1].set_title('Slope Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Slope (degrees)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('additional_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return data

def prepare_features_target(data):
    """Prepare features and target variable"""
    X = data.drop('Cover_Type', axis=1)
    y = data['Cover_Type']
    
    # Convert target from 1-7 to 0-6 for compatibility
    y = y - 1
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Unique target classes: {np.unique(y)}")
    
    return X, y

def train_models(X_train, y_train, X_test, y_test):
    """Train and compare different models"""
    print("\n=== Model Training and Evaluation ===\n")
    
    results = {}
    
    # Random Forest
    print("Training Random Forest...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted'),
        'recall': recall_score(y_test, rf_pred, average='weighted'),
        'f1': f1_score(y_test, rf_pred, average='weighted'),
        'training_time': rf_time,
        'predictions': rf_pred,
        'probabilities': rf_proba
    }
    
    # XGBoost
    print("Training XGBoost...")
    start_time = time.time()
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    
    results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, average='weighted'),
        'recall': recall_score(y_test, xgb_pred, average='weighted'),
        'f1': f1_score(y_test, xgb_pred, average='weighted'),
        'training_time': xgb_time,
        'predictions': xgb_pred,
        'probabilities': xgb_proba
    }
    
    # Print results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Time:      {metrics['training_time']:.2f}s")
    
    return results

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest"""
    print("\n=== Hyperparameter Tuning (Random Forest) ===\n")
    
    # Reduced parameter grid for faster tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print("Performing RandomizedSearchCV...")
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"Tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model"""
    print(f"\n=== {model_name} Evaluation ===\n")
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, 8), yticklabels=range(1, 8))
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Cover Type')
    plt.ylabel('True Cover Type')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'probabilities': probabilities
    }

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    print(f"\n=== {model_name} Feature Importance ===\n")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} - Top 20 Feature Importances', fontsize=16, fontweight='bold')
        plt.bar(range(20), importances[indices][:20], align='center', color='lightblue')
        plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top 10 features
        print("Top 10 most important features:")
        for i in range(10):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print("Model doesn't have feature_importances_ attribute")

def main():
    """Main function to run the complete analysis"""
    try:
        # Step 1: Load and preprocess data
        data = load_and_preprocess_data()
        
        # Step 2: Exploratory Data Analysis
        data = exploratory_data_analysis(data)
        
        # Step 3: Prepare features and target
        X, y = prepare_features_target(data)
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Step 5: Train and compare models
        results = train_models(X_train, y_train, X_test, y_test)
        
        # Step 6: Hyperparameter tuning
        best_rf_model = hyperparameter_tuning(X_train, y_train)
        
        # Step 7: Evaluate tuned model
        tuned_results = evaluate_model(best_rf_model, X_test, y_test, "Tuned Random Forest")
        
        # Step 8: Feature importance analysis
        for model_name, metrics in results.items():
            plot_feature_importance(metrics['model'], X.columns, model_name)
        
        plot_feature_importance(best_rf_model, X.columns, "Tuned Random Forest")
        
        # Final comparison
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON")
        print("="*60)
        
        models_to_compare = list(results.keys()) + ["Tuned Random Forest"]
        all_results = {**results, "Tuned Random Forest": {'accuracy': tuned_results['accuracy']}}
        
        comparison_df = pd.DataFrame({
            'Model': models_to_compare,
            'Accuracy': [all_results[model]['accuracy'] for model in models_to_compare]
        }).sort_values('Accuracy', ascending=False)
        
        print("\nModels ranked by accuracy:")
        print(comparison_df.to_string(index=False))
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color=['skyblue', 'lightgreen', 'orange'])
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0.9, 1.0)  # Forest cover datasets typically have high accuracy
        for i, v in enumerate(comparison_df['Accuracy']):
            plt.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== Analysis Complete ===")
        print("Visualizations saved as PNG files:")
        print("- eda_visualizations.png")
        print("- additional_visualizations.png")
        print("- Model-specific confusion matrices")
        print("- Model-specific feature importance plots")
        print("- model_comparison.png")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

