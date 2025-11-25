import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# train_model function
def train_model(X, y, model, param_grid, scale=True):
    """
    Train model with pipeline and hyperparameter tuning
    - scale: whether to use StandardScaler
    - GridSearchCV: tries different hyperparameters to find the best
    - cross_val_score: validates performance across 5 folds
    """
    if scale:
        steps = [("scaler", StandardScaler()), ("model", model)]
    else:
        steps = [("model", model)]
    
    pipe = Pipeline(steps)
    
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)
    
    cv_scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
    
    return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "cv_mean_accuracy": np.mean(cv_scores)
    }

def get_clean_data():
    """Load and preprocess the dataset"""
    # Load data
    data = pd.read_csv('data/student_performance.csv')
    print("Dataset loaded successfully.")
    print(f"Shape: {data.shape}")
    
    # Drop any missing values just in case
    data.dropna(inplace=True)
    
    # CREATE TARGET VARIABLE: College Ready = average score >= 75
    data['avg_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
    data['college_ready'] = (data['avg_score'] >= 75).astype(int)
    
    print(f"College Ready: {data['college_ready'].sum()} students ({data['college_ready'].mean()*100:.1f}%)")
    print(f"Not Ready: {(1-data['college_ready']).sum()} students ({(1-data['college_ready']).mean()*100:.1f}%)")
    
    return data

def prepare_features(data):
    """
    Prepare features (X) and target (y)
    - One-hot encode categorical variables
    - Remove test scores from features (to avoid data leakage)
    """
    # One-hot encode all categorical columns
    data_encoded = pd.get_dummies(data, columns=[
        'gender', 
        'race/ethnicity', 
        'parental level of education', 
        'lunch', 
        'test preparation course'
    ], drop_first=False)
    
    # Select features: everything EXCEPT scores and target
    feature_cols = [col for col in data_encoded.columns 
                    if col not in ['math score', 'reading score', 'writing score', 
                                   'avg_score', 'college_ready']]
    
    X = data_encoded[feature_cols]
    y = data_encoded['college_ready']
    
    print(f"\nFeatures: {len(feature_cols)} columns")
    print(f"Feature names: {feature_cols[:5]}... (showing first 5)")
    
    return X, y, feature_cols

# training both models
def train_models(X, y):
    """Train both Logistic Regression and Random Forest"""
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # ===== LOGISTIC REGRESSION =====
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)
    
    lr_params = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'model__penalty': ['l2'],
        'model__max_iter': [1000]
    }
    
    lr_results = train_model(X_train, y_train, LogisticRegression(random_state=42), 
                            lr_params, scale=True)
    
    # Evaluate on test set
    lr_pred = lr_results['best_model'].predict(X_test)
    lr_test_acc = accuracy_score(y_test, lr_pred)
    
    print(f"Best params: {lr_results['best_params']}")
    print(f"CV Accuracy: {lr_results['cv_mean_accuracy']:.4f}")
    print(f"Test Accuracy: {lr_test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_pred, target_names=['Not Ready', 'College Ready']))
    
    # ===== RANDOM FOREST =====
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    
    rf_params = {
        'model__n_estimators': [50, 100, 200],  # Number of trees
        'model__max_depth': [5, 10, 15, None],  # Tree depth
        'model__min_samples_split': [2, 5, 10]  # Min samples to split
    }
    
    rf_results = train_model(X_train, y_train, RandomForestClassifier(random_state=42), 
                            rf_params, scale=False)
    
    # Evaluate on test set
    rf_pred = rf_results['best_model'].predict(X_test)
    rf_test_acc = accuracy_score(y_test, rf_pred)
    
    print(f"Best params: {rf_results['best_params']}")
    print(f"CV Accuracy: {rf_results['cv_mean_accuracy']:.4f}")
    print(f"Test Accuracy: {rf_test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=['Not Ready', 'College Ready']))
    
    # ===== COMPARE MODELS =====
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Logistic Regression - CV: {lr_results['cv_mean_accuracy']:.4f}, Test: {lr_test_acc:.4f}")
    print(f"Random Forest       - CV: {rf_results['cv_mean_accuracy']:.4f}, Test: {rf_test_acc:.4f}")
    
    # Choose best model based on test accuracy
    if lr_test_acc > rf_test_acc:
        print("\n✅ Best Model: Logistic Regression")
        best_model = lr_results['best_model']
        model_name = "logistic_regression"
    else:
        print("\n✅ Best Model: Random Forest")
        best_model = rf_results['best_model']
        model_name = "random_forest"
    
    return best_model, model_name, {
        'lr': lr_results,
        'rf': rf_results,
        'X_test': X_test,
        'y_test': y_test
    }

def save_model(model, model_name, feature_cols):
    """Save the trained model and feature columns"""
    # Save model
    with open(f'model/{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved: model/{model_name}_model.pkl")
    
    # Save feature columns (needed for prediction)
    with open('model/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✅ Feature columns saved: model/feature_columns.pkl")


#main Ml pipeline
def main():
    """Main pipeline"""
    print("="*50)
    print("COLLEGE READINESS PREDICTION PIPELINE")
    print("="*50)
    
    # Step 1: Load and clean data
    data = get_clean_data()
    
    # Step 2: Prepare features
    X, y, feature_cols = prepare_features(data)
    
    # Step 3: Train models
    best_model, model_name, results = train_models(X, y)
    
    # Step 4: Save model
    save_model(best_model, model_name, feature_cols)
    
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETE!")
    print("="*50) 

if __name__ == "__main__":
    main()