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
            test_img_arrays = test_img_arrays + tf.unstack(x)
            prediction_prob = self.model.predict(x).flatten()
            test_predictions = np.concatenate([test_predictions, tf.where(prediction_prob < 0.5, 0, 1)])
            test_labels = np.concatenate([test_labels, y.numpy()])

        test_true_positives = np.where((test_labels == 1) & (test_predictions == 1))[0]
        test_false_positives = np.where((test_labels == 0) & (test_predictions == 1))[0]
        test_true_negatives = np.where((test_labels == 0) & (test_predictions == 0))[0]
        test_false_negatives = np.where((test_labels == 1) & (test_predictions == 0))[0]

        test_outcomes_dict = {"True Positives":test_true_positives,
                              "False Positives": test_false_positives,
                              "True Negatives": test_true_negatives, 
                              "False Negatives": test_false_negatives}
        
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

        for outcome in test_outcomes_dict:
            sample_img_indexes = list(np.random.choice(test_outcomes_dict[outcome], size=9))
            sample_test_img_arrays = [test_img_arrays[i] for i in sample_img_indexes]
            sample_test_prediction_probabilities = [self.model.predict(np.expand_dims(img_array, axis=0))[0][0] for img_array in sample_test_img_arrays]
            sample_test_predictions = test_predictions[sample_img_indexes]
            sample_test_labels = test_labels[sample_img_indexes]
            plot_name = f"Sample Grad-CAM visualizations on test data ({outcome})"

            self.generate_gradcam_visualizations(plot_name = plot_name,
                                                 sample_test_img_arrays = sample_test_img_arrays,
                                                 sample_test_prediction_probabilities = sample_test_prediction_probabilities,
                                                 sample_test_predictions = sample_test_predictions,
                                                 sample_test_labels = sample_test_labels)
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
        # self.model.layers[-1].activation = None
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape(persistent=True) as tape:
            last_conv_layer_output, preds = grad_model(img_array, training=False)
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

    
    def generate_gradcam_visualizations(self, plot_name, sample_test_img_arrays, sample_test_prediction_probabilities, sample_test_predictions, sample_test_labels):
        """_summary_

        Args:
            sample_test_img_arrays (_type_): _description_
            sample_test_prediction_probabilities (_type_): _description_
            sample_test_predictions (_type_): _description_
            sample_test_labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        fig, axes = plt.subplots(3,3, figsize=(18,18))
        fig.suptitle(plot_name)
        sample_test_predictions = np.where(sample_test_predictions == 1.0, "PNEUMONIA", "NORMAL")
        sample_test_labels = np.where(sample_test_labels == 1.0, "PNEUMONIA", "NORMAL")

        for i in range(len(sample_test_img_arrays)):

            img_array = np.expand_dims(sample_test_img_arrays[i], axis=0)
            prediction_prob = sample_test_prediction_probabilities[i]
            prediction = sample_test_predictions[i]
            label = sample_test_labels[i]

            heatmap = self.make_gradcam_heatmap(img_array = img_array)
            heatmap = np.uint8(255 * heatmap)
            superimposed_img = self.generate_superimposed_image(img_array = img_array, heatmap = heatmap)

            axes[i//3, i%3].imshow(superimposed_img)
            axes[i//3, i%3].set_title(f"Actual: {label}, Predicted: {prediction}\nProbability: {prediction_prob}")
            axes[i//3, i%3].axis('off')


        

