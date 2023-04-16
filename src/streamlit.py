
import logging
import math
import matplotlib.pyplot as plt
import mlflow
from modeling import model, evaluation
import numpy as np
import os
from pathlib import Path
from PIL import Image
import streamlit as st
import tempfile
import tensorflow as tf

logger = logging.getLogger(__name__)

@st.cache_resource
def retrieve_model(run_id, model_uri, model_name):
    return model.retrieve_model(run_id, model_uri, model_name)

def main():
    """This main function does the following:
    - Loads trained model on cache
    - Gets image input from user to be loaded for inferencing
    - Conducts inferencing on image
    - Outputs prediction results on the dashboard
    - Generate heatmaps for each convolutional layer superimposed on image
    """
    
    logger = logging.getLogger(__name__)
    
    logger.info("Intialise MLFlow...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    logger.info("Loading the model...")
    pred_model = retrieve_model(run_id=os.getenv("RUN_ID"),
                                model_uri=os.getenv("MODEL_URI"),
                                model_name=os.getenv("MODEL_NAME"))

    pred_model_input_shape = pred_model.input_shape
    image_width = pred_model_input_shape[1]
    image_height = pred_model_input_shape[2]
    conv_layers = list(filter(lambda x: isinstance(x, tf.keras.layers.Conv2D), pred_model.layers))

    logger.info("Loading dashboard...")
    title = st.title('X-ray pneumonia detector')

    image_upload = st.file_uploader('Insert x-ray image to detect any bacterial/viral pneumonia', type=['png','jpg','jpeg'])
    last_n_layers = st.slider('Select last n convolutional layers of model to visualize Grad-CAM heatmap', 2,len(conv_layers),2)
    classes = {0:"BACTERIA", 1:"NORMAL", 2:"VIRUS"}

    if st.button("Generate prediction"):
        logger.info("Conducting inferencing on image input...")
        img = tf.keras.preprocessing.image.load_img(image_upload, target_size=(image_width, image_height), color_mode="rgb")
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.image.convert_image_dtype(img, dtype=tf.float32)

        predicted_probas = pred_model.predict(np.expand_dims(img_array, axis=0))
        predicted_class = classes[predicted_probas.argmax(axis=1)[0]]
        class_probabilities = dict(zip(list(classes.values()), predicted_probas[0]))
        
        logger.info("Inferencing has completed")
        st.write(f"{predicted_class} is detected from x-ray image")
        st.write(f"Generating heatmaps per convolutional layer........")

        layer_heatmaps = generate_heatmaps_per_layer(model = pred_model,
                                                     img_array = img_array,
                                                     predicted_class = predicted_class,
                                                     class_probabilities = class_probabilities,
                                                     last_n = last_n_layers,
                                                     conv_layers = conv_layers)
        
        layer_heatmaps_image = Image.open(layer_heatmaps)
        st.image(layer_heatmaps_image)

    else:
        st.write("Awaiting an image...")

def generate_heatmaps_per_layer(model,
                                img_array,
                                predicted_class,
                                class_probabilities,
                                last_n,
                                conv_layers):

    evaluator = evaluation.Evaluator(model=model)
    last_n_conv_layers = conv_layers[-last_n:]
    last_n_conv_layers = list(reversed(last_n_conv_layers))
    n_plots = math.ceil(np.sqrt(len(last_n_conv_layers)))

    fig, axes = plt.subplots(n_plots,n_plots, figsize=(18,18))
    plot_name = "Grad_CAM_Heatmap"
    title = "Grad-CAM Heatmap of final {} convolutional layers\n \
             Predicted Class: {}\n \
             Class Probabilities: {}" \
            .format(len(last_n_conv_layers), predicted_class, class_probabilities)
    fig.suptitle(title, fontsize=20)

    for i in range(len(last_n_conv_layers)):

        conv_layer = last_n_conv_layers[i]
        conv_layer_name = conv_layer.name

        heatmap = evaluator.make_gradcam_heatmap(img_array = np.expand_dims(img_array, axis=0), conv_layer_name = conv_layer_name)
        heatmap = np.uint8(255 * heatmap)
        superimposed_img = evaluator.generate_superimposed_image(img_array = img_array, heatmap = heatmap)

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
    
