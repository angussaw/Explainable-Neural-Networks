from PIL import Image
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List


def build_model(model_params: dict,
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

    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip(data_augmentation["random_flip"]),
                                             tf.keras.layers.RandomRotation(data_augmentation["random_rotation"])
                                            ])
    
    rescale = tf.keras.layers.Rescaling(1./255)

    base_model = tf.keras.applications.MobileNetV2(input_shape=(rescale_params["img_height"], 
                                                                rescale_params["img_width"], 3),
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(rescale_params["img_height"], rescale_params["img_width"], 3)),
                                data_augmentation,
                                rescale,
                                base_model,
                                tf.keras.layers.GlobalMaxPooling2D(),
                                tf.keras.layers.Dropout(model_params["dropout"]),
                                tf.keras.layers.Dense(1, activation="sigmoid")])

    model.compile(loss=model_params["loss_function"],
                    optimizer=model_params["optimizer"],
                    metrics=metrics)
    
    return model
