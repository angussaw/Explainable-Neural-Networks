import hydra
import logging
import mlflow
from omegaconf import DictConfig
import os
import pandas as pd
from pathlib import Path
from src.modeling.evaluation import Evaluator
from src.modeling.model import build_model, retrieve_model
from src.utils import (
    timer,
    setup_logging,
    init_mlflow,
)
import tensorflow as tf

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="../../conf", config_name="train_config.yaml"
)
def training_pipeline(config: DictConfig, mode: str):
    """_summary_

    Args:
        config (DictConfig): _description_
        mode (str): _description_
    """
    data_dir = config["files"]["data_dir"]
    model_params = config["model"]["model_params"]
    training_params = config["model"]["training_params"]
    compile_params = config["model"]["compile_params"]
    fine_tuning_params = config["fine_tuning"]

    validation_split = training_params["validation_split"]
    img_height = model_params["rescale_params"]["img_height"]
    img_width = model_params["rescale_params"]["img_width"]
    batch_size = training_params["batch_size"]


    logger.info("Generating image datasets...")
    training_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{data_dir}/train',
                                                                    labels="inferred",
                                                                    label_mode="categorical",
                                                                    color_mode="rgb",
                                                                    validation_split=validation_split,
                                                                    subset="training",
                                                                    seed=123,
                                                                    image_size=(img_height, img_width),
                                                                    batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{data_dir}/train',
                                                                    labels="inferred",
                                                                    label_mode="categorical",
                                                                    color_mode="rgb",
                                                                    validation_split=validation_split,
                                                                    subset="validation",
                                                                    seed=123,
                                                                    image_size=(img_height, img_width),
                                                                    batch_size=batch_size)

    test_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{data_dir}/test',
                                                                labels="inferred",
                                                                label_mode="categorical",
                                                                color_mode="rgb",
                                                                image_size=(img_height, img_width),
                                                                batch_size = batch_size)

    logger.info("Intialise MLFlow...")
    artifact_name, description_str = init_mlflow(config["mlflow"])

    if mode == "training":
        logger.info("Building Model...")
        model = build_model(**model_params)

    elif mode == "fine-tuning":
        logger.info("Retrieving Model...")
        model = retrieve_model(**fine_tuning_params["retrieve_params"])
        model.trainable=True
        for layer in model.layers[:fine_tuning_params["fine_tune_at"]]:
            layer.trainable = False
        bn_layers = list(filter(lambda x: isinstance(x, tf.keras.layers.BatchNormalization), model.layers))
        for layer in bn_layers:
            layer.trainable = False

    model.compile(loss=compile_params["loss_function"],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=compile_params["learning_rate"]),
                  metrics=config["metrics"])
    

    logger.info(f'Model {mode}...')
    history = model.fit(training_dataset,
                        epochs=training_params["n_epochs"],
                        validation_data=validation_dataset)
    
    logger.info("Evaluating Model...")
    evaluator = Evaluator(model=model)
    final_metrics, test_metrics, visualizations_save_dir = evaluator.evaluate_model(metrics=config["metrics"],
                                                                                    test_data = test_dataset,
                                                                                    history=pd.DataFrame(history.history))

    with mlflow.start_run(
        run_name=config["mlflow"]["run_name"], description=description_str) as run:
        logger.info("Starting MLFlow Run...")

        logger.info("Saving model and params...")
        save_dir = os.path.dirname(visualizations_save_dir)
        model_dir = Path(os.sep.join([save_dir, "model"]))
        model_dir.mkdir()

        model_file_name = config["mlflow"]["model_name"]
        model_save_path = f'{model_dir}\{model_file_name}'
        model.save(model_save_path)
        mlflow.log_artifact(model_save_path)
        mlflow.log_params(compile_params)
        mlflow.log_params(training_params)

        logger.info("Saving metrics and model performance...")
        mlflow.log_metrics(final_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifacts(visualizations_save_dir)
        graph_uri = f"runs:/{run.info.run_id}/graph"
        logger.info(f"Model performance visualisation available at {graph_uri}")
        mlflow.end_run()

        logger.info(f'Model {mode} has completed!!')

if __name__ == "__main__":
    mode = os.getenv("MODE").lower()
    with timer(f'Model {mode}'):
        setup_logging()
        training_pipeline()

