import os
from pathlib import Path
import tempfile
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import numpy as np
import tensorflow as tf

class Evaluator:
    """Evaluation class will calculate metrics and generate visualizations
    """

    def __init__(
        self,
        model
        ) -> None:
        
        self.model = model
        self.last_conv_layer_name = list(filter(lambda x: isinstance(x, tf.keras.layers.Conv2D), self.model.layers))[-1].name


    def evaluate_model(
        self,
        metrics,
        history: pd.DataFrame,
        test_data
        ):
    
        """

        Returns:

        """

        tmp_dir = tempfile.mkdtemp()
        visualizations_save_dir = Path(os.sep.join([tmp_dir, "graph"]))
        visualizations_save_dir.mkdir()
    
        for metric in metrics:
            metric = metric.lower()
            val_metric = f'val_{metric}'
            plt.figure(figsize=(10,5))
            plt.plot(history[metric], label=metric)
            plt.plot(history[val_metric], label=val_metric)
            plt.legend([metric, val_metric], loc="upper left", prop={"size": 8})
            plot_name = f"Training and validation accuracy"
            plt.title(plot_name)
            file_name = f"{plot_name}.png"
            file_save_path = f"{visualizations_save_dir}/{file_name}"
            plt.savefig(file_save_path, bbox_inches="tight")

        
        final_metrics = dict(history.iloc[-1,:])

        test_img_arrays = []
        test_predictions = np.array([])
        test_labels =  np.array([])
        for x, y in test_data:
            test_img_arrays.append(x)
            prediction_prob = self.model.predict(x).flatten()
            test_predictions = np.concatenate([test_predictions, tf.where(prediction_prob < 0.5, 0, 1)])
            test_labels = np.concatenate([test_labels, y.numpy()])
        
        test_metrics = {
            "test_accuracy": accuracy_score(test_labels, test_predictions),
            "test_precision": precision_score(test_labels, test_predictions),
            "test_recall": recall_score(test_labels, test_predictions),
            "test_auc": roc_auc_score(test_labels, test_predictions)}
        
        ConfusionMatrixDisplay.from_predictions(test_labels, test_predictions, display_labels=test_data.class_names)
        plot_name = f"Confusion_Matrix (test data)"
        plt.title(plot_name)
        file_name = f"{plot_name}.png"
        file_save_path = f"{visualizations_save_dir}/{file_name}"
        plt.savefig(file_save_path, bbox_inches="tight")

        plot_name = self.generate_gradcam_visualizations(img_arrays = test_img_arrays, labels = test_labels)
        file_name = f"{plot_name}.png"
        file_save_path = f"{visualizations_save_dir}/{file_name}"
        plt.savefig(file_save_path)

        return final_metrics, test_metrics, visualizations_save_dir
    

    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """_summary_

        Args:
            img_array (_type_): _description_
            model (_type_): _description_
            last_conv_layer_name (_type_): _description_
            pred_index (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        self.model.layers[-1].activation = None
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape(persistent=True) as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()


    def generate_superimposed_image(self, img_array, heatmap, alpha=0.4):
        """_summary_

        Args:
            img_array (_type_): _description_
            heatmap (_type_): _description_
            alpha (float, optional): _description_. Defaults to 0.4.
        """
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + np.squeeze(img_array)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        return superimposed_img

    
    def generate_gradcam_visualizations(self, img_arrays, labels):
        """_summary_

        Args:
            img_arrays (_type_): _description_
            labels (_type_): _description_
            last_conv_layer_name (_type_): _description_
        """

        random_img_arrays = np.random.randint(99, size=9)
        plot_name = "Sample Grad-CAM visualizations on test data"
        fig, axes = plt.subplots(3,3, figsize=(18,18))
        fig.suptitle(plot_name)

        for i in range(len(random_img_arrays)):

            img_array = np.expand_dims(img_arrays[0][i], axis=0)
            prediction_prob = self.model.predict(img_array)
            prediction = np.where(prediction_prob[0] < 0.5, "NORMAL", "PNEUMONIA")[0]
            label = "PNEUMONIA" if labels[i] == 1.0 else "NORMAL"

            heatmap = self.make_gradcam_heatmap(img_array = img_array)
            heatmap = np.uint8(255 * heatmap)
            superimposed_img = self.generate_superimposed_image(img_array = img_array, heatmap = heatmap)

            axes[i//3, i%3].imshow(superimposed_img)
            axes[i//3, i%3].set_title(f"Actual: {label}, Predicted: {prediction}\nProbability: {round(prediction_prob[0][0],4)}")
            axes[i//3, i%3].axis('off')

        return plot_name

        





            
    

    



        

