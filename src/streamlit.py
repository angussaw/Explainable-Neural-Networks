import os
import logging
import hydra
import hydra.core.global_hydra
import streamlit as st
from modeling import model, evaluation
import logging
import mlflow
from PIL import Image
from torchvision.transforms import transforms
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

@st.cache_resource
def retrieve_model(run_id, model_uri, model_name, destination_path):
    return model.retrieve_model(run_id, model_uri, model_name, destination_path)

def main():
    """This main function does the following:
    - load logging config
    - loads trained model on cache
    - gets string input from user to be loaded for inferencing
    - conducts inferencing on string
    - outputs prediction results on the dashboard
    """
    
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up logging configuration.")

    logger.info("Intialise MLFlow...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    logger.info("Loading the model...")
    pred_model = retrieve_model(run_id="434536936096303142",
                                model_uri="96eb2ef9dd4d4c2db56e348cfd6e9cef",
                                model_name="finalized_model",
                                destination_path="./models")

    pred_model_input_shape = pred_model.input_shape
    image_width = pred_model_input_shape[1]
    image_height = pred_model_input_shape[2]
    conv_layers = list(filter(lambda x: isinstance(x, tf.keras.layers.Conv2D), pred_model.layers))

    logger.info("Loading dashboard...")
    title = st.title('X-ray pneumonia detector')

    image_upload = st.file_uploader('Insert x-ray image for classification', type=['png','jpg','jpeg'])
    last_n_layers = st.slider('Select last n convolutional layers of model to visualize Grad-CAM heatmap', 2,len(conv_layers),2)
    classes = {0:"BACTERIA", 1:"NORMAL", 2:"VIRUS"}

    if st.button("Generate prediction"):
        logger.info("Conducting inferencing on image input...")
        img = tf.keras.preprocessing.image.load_img(image_upload, target_size=(image_width, image_height), color_mode="rgb")
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_tensor = tf.image.convert_image_dtype(img, dtype=tf.float32)

        predicted_probas = pred_model.predict(np.expand_dims(img_tensor, axis=0))
        predicted_proba = max(predicted_probas[0])
        predicted_class = classes[predicted_probas.argmax(axis=1)[0]]
        logger.info("Inferencing has completed")
        st.write(f"{predicted_class} is detected from x-ray image")
        st.write(f"Generating heatmaps per convolutional layer........")

        layer_heatmaps = generate_heatmaps_per_layer(model = pred_model,
                                                     img_tensor = img_tensor,
                                                     predicted_class = predicted_class,
                                                     predicted_proba = predicted_proba,
                                                     last_n = last_n_layers,
                                                     conv_layers = conv_layers)
        
        layer_heatmaps_image = Image.open(layer_heatmaps)
        st.image(layer_heatmaps_image)

    else:
        st.write("Awaiting an image...")

def generate_heatmaps_per_layer(model,
                                img_tensor,
                                predicted_class,
                                predicted_proba,
                                last_n,
                                conv_layers):

    evaluator = evaluation.Evaluator(model=model)
    last_n_conv_layers = conv_layers[-last_n:]
    last_n_conv_layers = list(reversed(last_n_conv_layers))
    n_plots = math.ceil(np.sqrt(len(last_n_conv_layers)))

    fig, axes = plt.subplots(n_plots,n_plots, figsize=(18,18))
    plot_name = "Grad_CAM_Heatmap"
    title = "Grad-CAM Heatmap of last {} convolutional layers\nClass: {}\nProbability: {}".format(len(last_n_conv_layers), predicted_class, predicted_proba)
    fig.suptitle(title)

    for i in range(len(last_n_conv_layers)):

        conv_layer = last_n_conv_layers[i]
        conv_layer_name = conv_layer.name

        heatmap = evaluator.make_gradcam_heatmap(img_array = np.expand_dims(img_tensor, axis=0), conv_layer_name = conv_layer_name)
        heatmap = np.uint8(255 * heatmap)
        superimposed_img = evaluator.generate_superimposed_image(img_array = img_tensor, heatmap = heatmap)

        axes[i//n_plots, i%n_plots].imshow(superimposed_img)
        axes[i//n_plots, i%n_plots].set_title(f"Layer: {conv_layer_name}")
        axes[i//n_plots, i%n_plots].axis('off')

    tmp_dir = tempfile.mkdtemp()
    save_dir = Path(tmp_dir)
    file_name = f"{plot_name}.png"
    file_save_path = f"{save_dir}/{file_name}"
    plt.savefig(file_save_path, bbox_inches="tight")

    return file_save_path

if __name__ == "__main__":
    main()
    
