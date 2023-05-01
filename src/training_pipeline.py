import hydra
import logging
from omegaconf import DictConfig
import os
import tensorflow as tf

import xray_classifier as xray_clf

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="../conf", config_name="train_config.yaml"
)
def training_pipeline(config: DictConfig):
    """
    Loads the training, validation and test data from directory into batches
    and performs training of a new model or fine-tuning of an existing model

    Args:
        config (DictConfig): training configurations as specified in a yaml file
    """

    logger.info("Generating image datasets...")
    training_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{config["files"]["data_dir"]}/train',
                                                                    labels="inferred",
                                                                    label_mode="categorical",
                                                                    color_mode="rgb",
                                                                    validation_split=config["model"]["training_params"]["validation_split"],
                                                                    subset="training",
                                                                    seed=123,
                                                                    image_size=(config["model"]["model_params"]["rescale_params"]["img_height"],
                                                                                config["model"]["model_params"]["rescale_params"]["img_width"]),
                                                                    batch_size=config["model"]["training_params"]["batch_size"])

    validation_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{config["files"]["data_dir"]}/train',
                                                                    labels="inferred",
                                                                    label_mode="categorical",
                                                                    color_mode="rgb",
                                                                    validation_split=config["model"]["training_params"]["validation_split"],
                                                                    subset="validation",
                                                                    seed=123,
                                                                    image_size=(config["model"]["model_params"]["rescale_params"]["img_height"],
                                                                                config["model"]["model_params"]["rescale_params"]["img_width"]),
                                                                    batch_size=config["model"]["training_params"]["batch_size"])

    test_dataset = tf.keras.utils.image_dataset_from_directory(directory=f'{config["files"]["data_dir"]}/test',
                                                                labels="inferred",
                                                                label_mode="categorical",
                                                                color_mode="rgb",
                                                                image_size=(config["model"]["model_params"]["rescale_params"]["img_height"],
                                                                            config["model"]["model_params"]["rescale_params"]["img_width"]),
                                                                batch_size = config["model"]["training_params"]["batch_size"])
    
    logger.info(f"Loading image datasets for {mode}...")
    xray_clf.modeling.training.train_pipeline(config = config,
                                              mode = os.getenv("MODE").lower(),
                                              training_dataset = training_dataset,
                                              validation_dataset = validation_dataset,
                                              test_dataset = test_dataset)


    logger.info(f'Model {mode} has completed!!')

if __name__ == "__main__":
    mode = os.getenv("MODE").lower()
    with xray_clf.utils.timer(f'Model {mode}'):
        xray_clf.utils.setup_logging()
        training_pipeline()

