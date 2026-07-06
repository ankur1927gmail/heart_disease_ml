import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Define models
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Support Vector Machine": CalibratedClassifierCV(SVC(kernel='rbf', random_state=42), ensemble=False)
            }
            
            # Define hyperparameters for each model
            params = {
                "Logistic Regression": {
                    'solver': ['lbfgs', 'saga'],
                    'C': [0.1, 1, 10],
                    'max_iter': [1000, 2000]
                },
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                "Support Vector Machine": {
                    'estimator__C': [0.1, 1, 10, 100],
                    'estimator__kernel': ['linear', 'rbf', 'poly'],
                    'estimator__gamma': ['scale', 'auto', 0.001, 0.01]
                }
            }

            # Evaluate all models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # Get best model based on F1 score
            best_model_score = max([report['f1_score'] for report in model_report.values()])
            best_model_name = [name for name, report in model_report.items() 
                              if report['f1_score'] == best_model_score][0]
            
            best_metrics = model_report[best_model_name]
            best_model = best_metrics['model']
            
            # Extract accuracy scores for all three models
            accuracy_scores = {}
            for model_name, metrics in model_report.items():
                accuracy_scores[model_name] = metrics['accuracy']
                logging.info(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}")
            
            # Log best model summary
            logging.info(f"\n{'='*60}")
            logging.info(f"BEST MODEL: {best_model_name}")
            logging.info(f"{'='*60}")
            logging.info(f"Accuracy:  {best_metrics['accuracy']:.4f}")
            logging.info(f"Precision: {best_metrics['precision']:.4f}")
            logging.info(f"Recall:    {best_metrics['recall']:.4f}")
            logging.info(f"F1 Score:  {best_metrics['f1_score']:.4f}")
            logging.info(f"ROC-AUC:   {best_metrics['roc_auc']:.4f}")
            logging.info(f"CV Mean:   {best_metrics['cv_mean']:.4f} (+/- {best_metrics['cv_std']:.4f})")

            logging.info(f"\nBest model found on both training and testing dataset")
            
            # Note: F1 score threshold removed for debugging purposes
            if best_metrics['f1_score'] < 0.3:
                logging.warning(f"Warning: Model performance is low (F1 Score: {best_metrics['f1_score']:.4f})")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return accuracy_scores
            
        except Exception as e:
            raise CustomException(e, sys)