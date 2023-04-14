from PIL import Image
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
import mlflow
import os


def build_model(model_architecture_params: dict, 
                rescale_params: dict,
                data_augmentation: dict):
    """_summary_

    Args:
        model_architecture_params (dict): _description_
        rescale_params (dict): _description_
        data_augmentation (dict): _description_

    Returns:
        _type_: _description_
    """
    
    inputs = tf.keras.layers.Input(shape=(rescale_params["img_height"], rescale_params["img_height"], 3))
    # Note: Data augmentation is inactive at test time so input images will only be augmented during calls to Model.fit (not Model.evaluate or Model.predict).
    augmented = tf.keras.layers.RandomFlip(data_augmentation["random_flip"])(inputs)
    augmented = tf.keras.layers.RandomRotation(data_augmentation["random_rotation"])(augmented)
    rescale = tf.keras.layers.Rescaling(1./255)(augmented)

    base_model = tf.keras.applications.MobileNetV2(input_tensor=rescale,
                                                include_top=False,
                                                weights='imagenet')
    base_model.trainable = False

    pooling = tf.keras.layers.GlobalMaxPooling2D()(base_model.layers[-1].output)
    dropout = tf.keras.layers.Dropout(model_architecture_params["dropout"])(pooling)

    dense_output = tf.keras.layers.Dense(model_architecture_params["n_nodes"], activation="relu")(dropout)
    dense_output = tf.keras.layers.Dense(model_architecture_params["n_nodes"], activation="relu")(dense_output)
    dense_output = tf.keras.layers.Dense(model_architecture_params["n_nodes"], activation="relu")(dense_output)
    dense_output = tf.keras.layers.Dense(model_architecture_params["n_nodes"], activation="relu")(dense_output)
    dense_output = tf.keras.layers.Dense(model_architecture_params["n_nodes"], activation="relu")(dense_output)
    dense_output = tf.keras.layers.Dense(model_architecture_params["n_nodes"], activation="relu")(dense_output)

    final_output = tf.keras.layers.Dense(3, activation="softmax")(dense_output)
    model = tf.keras.models.Model(inputs=inputs, outputs=final_output)

    return model


def retrieve_model(run_id: str,
                   model_uri: str,
                   model_name: str,
                   destination_path: str):
    """_summary_

    Args:
        run_id (str): _description_
        model_uri (str): _description_
        model_name (str): _description_
        destination_path (str): _description_

    Raises:
        mlflow_error: _description_

    Returns:
        _type_: _description_
    """

    artifact_uri = f'mlflow-artifacts:/{run_id}/{model_uri}/artifacts/{model_name}'
    try:
        mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri, dst_path=destination_path
        )
    except Exception as mlflow_error:
        raise mlflow_error
    
    model_file = os.path.split(artifact_uri)[1]
    model = tf.keras.models.load_model(f'{destination_path}/{model_file}')

    return model

