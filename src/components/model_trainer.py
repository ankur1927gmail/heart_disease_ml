import os
import sys
from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


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

            models = {
                "Logistic Regression": LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Support Vector Machine": CalibratedClassifierCV(
                    SVC(kernel='rbf', random_state=42),
                    ensemble=False
                )
            }

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

            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                params
            )

            best_model_score = max(
                report['f1_score'] for report in model_report.values()
            )
            best_model_name = next(
                name for name, report in model_report.items()
                if report['f1_score'] == best_model_score
            )

            best_metrics = model_report[best_model_name]
            best_model = best_metrics['model']

            accuracy_scores = {}
            for model_name, metrics in model_report.items():
                accuracy_scores[model_name] = metrics['accuracy']
                logging.info(
                    "%s: Accuracy = %.4f",
                    model_name,
                    metrics['accuracy']
                )

            logging.info("\n%s", '=' * 60)
            logging.info("BEST MODEL: %s", best_model_name)
            logging.info("%s", '=' * 60)
            logging.info("Accuracy:  %.4f", best_metrics['accuracy'])
            logging.info("Precision: %.4f", best_metrics['precision'])
            logging.info("Recall:    %.4f", best_metrics['recall'])
            logging.info("F1 Score:  %.4f", best_metrics['f1_score'])
            logging.info("ROC-AUC:   %.4f", best_metrics['roc_auc'])
            logging.info(
                "CV Mean:   %.4f (+/- %.4f)",
                best_metrics['cv_mean'],
                best_metrics['cv_std']
            )

            logging.info(
                "\nBest model found on both training and testing dataset"
            )

            if best_metrics['f1_score'] < 0.3:
                logging.warning(
                    "Warning: Model performance is low "
                    "(F1 Score: %.4f)",
                    best_metrics['f1_score']
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return accuracy_scores

        except Exception as e:
            raise CustomException(e, sys) from e
