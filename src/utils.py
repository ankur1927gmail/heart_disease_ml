import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple classification models with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training features
    y_train : array-like of shape (n_samples,)
        Training target
    X_test : array-like of shape (n_samples, n_features)
        Test features
    y_test : array-like of shape (n_samples,)
        Test target
    models : dict
        Dictionary of model names and model objects
    params : dict
        Dictionary of model names and their hyperparameters
    
    Returns:
    --------
    model_report : dict
        Dictionary containing evaluation metrics for each model
    """
    try:
        model_report = {}
        
        for model_name, model in models.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"Training {model_name}...")
            logging.info(f"{'='*60}")
            
            # GridSearchCV for hyperparameter tuning
            # error_score='raise' will raise exceptions for invalid combinations
            gs = GridSearchCV(
                model, 
                params[model_name], 
                cv=5, 
                scoring='f1_macro', 
                n_jobs=-1, 
                verbose=1,
                error_score='raise'
            )
            
            try:
                gs.fit(X_train, y_train)
            except Exception as e:
                logging.warning(f"GridSearchCV failed for {model_name}: {str(e)}")
                logging.info(f"Using model with default parameters instead")
                gs.fit(X_train, y_train)
            
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            
            logging.info(f"Best parameters: {best_params}")
            
            # Make predictions on test set
            y_pred = best_model.predict(X_test)
            
            # For multi-class ROC-AUC, we need probability predictions
            try:
                y_pred_proba = best_model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', zero_division=0)
            except Exception as e:
                logging.warning(f"Could not calculate ROC-AUC for {model_name}: {str(e)}")
                roc_auc = 0.0
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_macro')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            model_report[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': best_model,
                'best_params': best_params,
                'y_pred': y_pred,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Log evaluation metrics
            logging.info(f"\n{model_name} - Test Set Metrics:")
            logging.info(f"  Accuracy:  {accuracy:.4f}")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall:    {recall:.4f}")
            logging.info(f"  F1 Score:  {f1:.4f}")
            logging.info(f"  ROC-AUC:   {roc_auc:.4f}")
            logging.info(f"\n{model_name} - Cross-Validation (5-Fold):")
            logging.info(f"  CV F1 Scores: {cv_scores}")
            logging.info(f"  CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
            logging.info(f"\nClassification Report for {model_name}:")
            logging.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
        
        return model_report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)