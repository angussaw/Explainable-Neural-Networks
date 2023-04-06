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
from sklearn.preprocessing import label_binarize
import numpy as np
import tensorflow as tf

class Evaluator:
    """Evaluation class will calculate metrics and generate visualizations
    """

    def __init__(
        self,
        model,
        test_data,
        history: pd.DataFrame,
        ) -> None:
        
        self.model = model
        self.test_data = test_data
        self.history = history
        self.last_conv_layer_name = list(filter(lambda x: isinstance(x, tf.keras.layers.Conv2D), self.model.layers))[-1].name
        tmp_dir = tempfile.mkdtemp()
        self.visualizations_save_dir = Path(os.sep.join([tmp_dir, "graph"]))
        self.visualizations_save_dir.mkdir()

    def evaluate_model(
        self,
        metrics
        ):
    
        """

        Returns:

        """

        class_names = self.test_data.class_names
        test_img_arrays = []
        test_predictions = np.array([])
        test_labels =  np.array([])
        for x, y in self.test_data:
            test_img_arrays = test_img_arrays + tf.unstack(x)
            test_prediction_probs = self.model.predict(x)
            test_predictions = np.concatenate([test_predictions, test_prediction_probs.argmax(axis=1)])
            test_labels = np.concatenate([test_labels, y.numpy().argmax(axis=1)])

        all_test_outcomes_dict = {}
        class_test_metrics = []
        for class_index in range(len(class_names)):
            test_class_outcomes, test_class_metrics = self.calculate_class_outcomes_and_metrics(class_index = class_index,
                                                                                                labels = test_labels,
                                                                                                predictions = test_predictions)
            all_test_outcomes_dict[class_names[class_index]] = test_class_outcomes
            test_class_metrics["Class Name"] = class_names[class_index]
            class_test_metrics.append(test_class_metrics)
        
        self.generate_class_metrics_visualization(class_metrics = class_test_metrics)

        self.generate_gradcam_visualizations(outcomes_dict = all_test_outcomes_dict,
                                             img_arrays = test_img_arrays,
                                             labels = test_labels,
                                             predictions = test_predictions,
                                             class_names = class_names)
        
        self.generate_confusion_matrix(class_names = class_names,
                                       labels = test_labels,
                                       predictions = test_predictions)
        
        for metric in metrics:
            self.generate_epochs_visualization(history = self.history, metric = metric)

        final_metrics = dict(self.history.iloc[-1,:])

        test_metrics = self.calculate_test_metrics(class_names = class_names,
                                                   labels = test_labels,
                                                   predictions = test_predictions
                                                   )

        return final_metrics, test_metrics, self.visualizations_save_dir
    
    def generate_epochs_visualization(self, history, metric):
        """_summary_

        Args:
            metric (_type_): _description_
        """

        metric = metric.lower()
        val_metric = f'val_{metric}'
        plt.figure(figsize=(10,5))
        plt.plot(history[metric], label=metric)
        plt.plot(history[val_metric], label=val_metric)
        plt.legend([metric, val_metric], loc="upper left", prop={"size": 8})
        plot_name = f"Training and validation {metric}"
        plt.title(plot_name)
        self.save_visualization(plot_name = plot_name)


    def calculate_test_metrics(self, class_names, labels, predictions):
        """
        
        """
        labels_binarized = label_binarize(labels, classes=list(range(len(class_names))))
        predictions_binarized = label_binarize(predictions, classes=list(range(len(class_names))))
        test_metrics = {
            "test_accuracy": accuracy_score(labels, predictions),
            "test_precision": precision_score(labels, predictions, average = 'weighted'),
            "test_recall": recall_score(labels, predictions, average = 'weighted'),
            "test_auc": roc_auc_score(labels_binarized, predictions_binarized, average = 'weighted', multi_class='ovr')}
        
        return test_metrics
    
    def calculate_class_outcomes_and_metrics(self, class_index, labels, predictions):
        """_summary_

        Args:
            class_index (_type_): _description_
            labels (_type_): _description_
            predictions (_type_): _description_
        """

        class_true_positives = np.where((labels == class_index) & (predictions == class_index))[0]
        class_false_positives = np.where((labels != class_index) & (predictions == class_index))[0]
        class_true_negatives = np.where((labels != class_index) & (predictions != class_index))[0]
        class_false_negatives = np.where((labels == class_index) & (predictions != class_index))[0]

        class_outcomes = {"True Positives":class_true_positives,
                        "False Positives": class_false_positives,
                        "True Negatives": class_true_negatives, 
                        "False Negatives": class_false_negatives}
        
        class_accuracy = (len(class_true_positives) + len(class_true_negatives)) / len(labels)
        class_precision = (len(class_true_positives)) / (len(class_true_positives) + len(class_false_positives))
        class_recall = (len(class_true_positives)) / (len(class_true_positives) + len(class_false_negatives))
        
        class_metrics = {"Test Accuracy": class_accuracy,
                        "Test Precision": class_precision,
                        "Test Recall": class_recall}
        
        return class_outcomes, class_metrics

    def generate_confusion_matrix(self, class_names, labels, predictions):
        """_summary_

        Args:
            class_names (_type_): _description_
            labels (_type_): _description_
            predictions (_type_): _description_
        """
        ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=class_names)
        plot_name = f"Confusion_Matrix (test data)"
        plt.title(plot_name)
        self.save_visualization(plot_name = plot_name)


    def generate_class_metrics_visualization(self, class_metrics):
        """_summary_

        Args:
            class_metrics (_type_): _description_
        """
        class_metrics_df = pd.DataFrame(class_metrics).set_index("Class Name").transpose()
        plot_name = "Test metrics for each class"
        class_metrics_df.plot(kind="bar", figsize=(10,5))
        plt.title(plot_name)
        self.save_visualization(plot_name = plot_name)

    def generate_gradcam_visualizations(self,
                                        outcomes_dict,
                                        img_arrays,
                                        labels,
                                        predictions,
                                        class_names):
        """_summary_

        Args:
            outcomes_dict (_type_): _description_
            img_arrays (_type_): _description_
            labels (_type_): _description_
            predictions (_type_): _description_
            class_names (_type_): _description_
        """
        for class_name in outcomes_dict:
            class_outcomes_dict = outcomes_dict[class_name]
            for outcome in class_outcomes_dict:
                samples = {}
                sample_img_indexes = list(np.random.choice(class_outcomes_dict[outcome], size=9))
                samples["sample_test_img_arrays"] = [img_arrays[i] for i in sample_img_indexes]
                samples["sample_test_prediction_probabilities"] = [max(self.model.predict(np.expand_dims(img_array, axis=0))[0]) for img_array in samples["sample_test_img_arrays"]]
                samples["sample_test_predictions"] = [class_names[int(i)] for i in predictions[sample_img_indexes]]
                samples["sample_test_labels"] = [class_names[int(i)] for i in labels[sample_img_indexes]]
                
                plot_name = f"Sample Grad-CAM visualizations on test data ({class_name}-{outcome})"

                self.generate_sample_outcome_visualizations(plot_name = plot_name,
                                                            **samples)
                self.save_visualization(plot_name = plot_name)

    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """_summary_

        Args:
            img_array (_type_): _description_
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
        jet_heatmap = jet_heatmap.resize((img_array.shape[0], img_array.shape[1]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img_array
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        return superimposed_img

    
    def generate_sample_outcome_visualizations(self,
                                               plot_name,
                                               sample_test_img_arrays,
                                               sample_test_prediction_probabilities,
                                               sample_test_predictions,
                                               sample_test_labels):
        """_summary_

        Args:
            plot_name (_type_): _description_
            sample_test_img_arrays (_type_): _description_
            sample_test_prediction_probabilities (_type_): _description_
            sample_test_predictions (_type_): _description_
            sample_test_labels (_type_): _description_
        """
        
        fig, axes = plt.subplots(3,3, figsize=(18,18))
        fig.suptitle(plot_name)

        for i in range(len(sample_test_img_arrays)):

            img_array = sample_test_img_arrays[i]
            prediction_prob = sample_test_prediction_probabilities[i]
            prediction = sample_test_predictions[i]
            label = sample_test_labels[i]

            heatmap = self.make_gradcam_heatmap(img_array = np.expand_dims(img_array, axis=0))
            heatmap = np.uint8(255 * heatmap)
            img_array_resized = tf.keras.preprocessing.image.array_to_img(img_array)
            img_array_resized = img_array_resized.resize((224, 224))
            img_array_resized = tf.keras.preprocessing.image.img_to_array(img_array_resized)
            superimposed_img = self.generate_superimposed_image(img_array = img_array_resized, heatmap = heatmap)

            axes[i//3, i%3].imshow(superimposed_img)
            axes[i//3, i%3].set_title(f"Actual: {label}, Predicted: {prediction}\nProbability: {prediction_prob}")
            axes[i//3, i%3].axis('off')

    def save_visualization(self, plot_name):
        """_summary_

        Args:
            plot_name (_type_): _description_
        """

        file_name = f"{plot_name}.png"
        file_save_path = f"{self.visualizations_save_dir}/{file_name}"
        plt.savefig(file_save_path, bbox_inches="tight")


        

