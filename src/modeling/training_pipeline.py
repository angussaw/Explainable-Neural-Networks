import tensorflow as tf
import pandas as pd
from src.modeling.model import build_model
from src.modeling.evaluation import Evaluator
from src.utils import (
    timer,
    setup_logging,
    init_mlflow,
)
import os
from pathlib import Path
import joblib
import hydra
import mlflow
import mlflow.keras
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="../../conf", config_name="train_config.yaml"
)
def training_pipeline(config: DictConfig):
    """_summary_

    Args:
        config (dict): _description_
    """
    data_dir = config["files"]["data_dir"]
    validation_split = config["training_params"]["validation_split"]
    img_height = config["model"]["rescale_params"]["img_height"]
    img_width = config["model"]["rescale_params"]["img_width"]
    batch_size = config["training_params"]["batch_size"]

    logger.info("Generating image datasets...")
    training_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{data_dir}/train',
                                                                    labels="inferred",
                                                                    color_mode="rgb",
                                                                    validation_split=validation_split,
                                                                    subset="training",
                                                                    seed=123,
                                                                    image_size=(img_height, img_width),
                                                                    batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{data_dir}/train',
                                                                    labels="inferred",
                                                                    color_mode="rgb",
                                                                    validation_split=validation_split,
                                                                    subset="validation",
                                                                    seed=123,
                                                                    image_size=(img_height, img_width),
                                                                    batch_size=batch_size)

    test_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{data_dir}/test',
                                                                labels="inferred",
                                                                color_mode="rgb",
                                                                image_size=(img_height, img_width),
                                                                batch_size = batch_size)

    logger.info("Building Model...")
    model = build_model(**config["model"])

    logger.info("Training Model...")
    history = model.fit(training_dataset,
                        epochs=config["training_params"]["n_epochs"],
                        validation_data=validation_dataset)
    
    logger.info("Evaluating Model...")
    evaluator = Evaluator(model=model)
    final_metrics, test_metrics, visualizations_save_dir = evaluator.evaluate_model(metrics=config["model"]["metrics"],
                                                            history=pd.DataFrame(history.history),
                                                            test_data=test_dataset)

    logger.info("Intialise MLFlow...")
    artifact_name, description_str = init_mlflow(config["mlflow"])
    with mlflow.start_run(
        run_name="Model Training", description=description_str) as run:
    
        logger.info("Starting MLFlow Run...")

        logger.info("Saving model and params...")
        print(visualizations_save_dir)
        save_dir = os.path.dirname(visualizations_save_dir)
        print(save_dir)
        model_dir = Path(os.sep.join([save_dir, "model"]))
        model_dir.mkdir()

        model_file_name = config["mlflow"]["model_name"]
        # model_save_path = Path(os.sep.join([model_dir, model_file_name]))
        model_save_path = f'{model_dir}\{model_file_name}'
        model.save(model_save_path)
        print(model_dir)
        print(model_save_path)
        # mlflow.keras.save_model(model, model_save_path)
        # mlflow.keras.log_model(model, model_save_path)
        mlflow.log_artifact(model_save_path)

        mlflow.log_params(config['model']["model_params"])
        mlflow.log_params(config["training_params"])
        # joblib.dump(model, model_save_path)
        # mlflow.log_artifact(model_save_path, "model")
        # mlflow.register_model(model_uri=model_uri, name=artifact_name)
        # model_uri = f"runs:/{run.info.run_id}/model/{model_file_name}"
        # os.environ["MODEL_URI"] = model_uri
        # logger.info(f"Model available at {model_uri}")

        logger.info("Saving metrics and model performance...")
        mlflow.log_metrics(final_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifacts(visualizations_save_dir)
        graph_uri = f"runs:/{run.info.run_id}/graph"
        os.environ["GRAPH_URI"] = graph_uri
        logger.info(f"Model performance visualisation available at {graph_uri}")
        mlflow.end_run()

        logger.info("Model training has completed!!!")

if __name__ == "__main__":
    with timer("Model training"):
        setup_logging()
        training_pipeline()

