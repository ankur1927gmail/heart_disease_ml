import os
import pickle
import sys

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
        raise CustomException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Evaluate multiple classification models with hyperparameter tuning."""
    try:
        model_report = {}

        for model_name, model in models.items():
            logging.info("\n%s", '=' * 60)
            logging.info("Training %s...", model_name)
            logging.info("%s", '=' * 60)

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
                logging.warning(
                    "GridSearchCV failed for %s: %s",
                    model_name,
                    str(e)
                )
                logging.info("Using model with default parameters instead")
                gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_params = gs.best_params_

            logging.info("Best parameters: %s", best_params)

            y_pred = best_model.predict(X_test)

            try:
                y_pred_proba = best_model.predict_proba(X_test)
                roc_auc = roc_auc_score(
                    y_test,
                    y_pred_proba,
                    multi_class='ovr',
                    zero_division=0
                )
            except Exception as e:
                logging.warning(
                    "Could not calculate ROC-AUC for %s: %s",
                    model_name,
                    str(e)
                )
                roc_auc = 0.0

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test,
                y_pred,
                average='macro',
                zero_division=0
            )
            recall = recall_score(
                y_test,
                y_pred,
                average='macro',
                zero_division=0
            )
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            cv_scores = cross_val_score(
                best_model,
                X_train,
                y_train,
                cv=5,
                scoring='f1_macro'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

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

            logging.info("\n%s - Test Set Metrics:", model_name)
            logging.info("  Accuracy:  %.4f", accuracy)
            logging.info("  Precision: %.4f", precision)
            logging.info("  Recall:    %.4f", recall)
            logging.info("  F1 Score:  %.4f", f1)
            logging.info("  ROC-AUC:   %.4f", roc_auc)
            logging.info("\n%s - Cross-Validation (5-Fold):", model_name)
            logging.info("  CV F1 Scores: %s", cv_scores)
            logging.info("  CV Mean: %.4f (+/- %.4f)", cv_mean, cv_std)
            logging.info("\nClassification Report for %s:", model_name)
            logging.info(
                "\n%s",
                classification_report(y_test, y_pred, zero_division=0)
            )

        return model_report

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e
