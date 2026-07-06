import sys
import os

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            # assume train/test CSVs are present in artifacts/
            train_path = os.path.join("artifacts", "train.csv")
            test_path = os.path.join("artifacts", "test.csv")

            data_transform = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transform.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )

            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            model_trainer = ModelTrainer()
            accuracy_scores = model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)

            logging.info(f"Training finished. Model accuracies: {accuracy_scores}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run_pipeline()
