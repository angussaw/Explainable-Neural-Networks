from PIL import Image
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List


def build_model(model_params: dict,
                model_architecture_params: dict, 
                rescale_params: dict,
                data_augmentation: dict,
                metrics: List):
    """_summary_

    Args:
        model_params (dict): _description_
        rescale_params (dict): _description_
        data_augmentation (dict): _description_
        metrics (List): _description_

    Returns:
        _type_: _description_
    """
    
    inputs = tf.keras.layers.Input(shape=(rescale_params["img_height"], rescale_params["img_height"], 3))
    augmented = tf.keras.layers.RandomFlip(data_augmentation["random_flip"])(inputs)
    augmented = tf.keras.layers.RandomRotation(data_augmentation["random_rotation"])(augmented)
    rescale = tf.keras.layers.Rescaling(1./255)(augmented)

    base_model = tf.keras.applications.MobileNetV2(input_tensor=rescale,
                                                include_top=False,
                                                weights='imagenet')
    base_model.trainable = False

    pooling = tf.keras.layers.GlobalMaxPooling2D()(base_model.layers[-1].output)
    dropout = tf.keras.layers.Dropout(model_params["dropout"])(pooling)

    # dense_output =  tf.keras.layers.Dense(model_architecture_params["n_nodes"], activation="relu")(dropout)

    final_output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(inputs=inputs, outputs=final_output)

    model.compile(loss=model_params["loss_function"],
                    optimizer=model_params["optimizer"],
                    metrics=metrics)
    
    return model
