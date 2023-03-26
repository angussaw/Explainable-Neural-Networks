import os
from pathlib import Path
import tempfile
import numpy.typing as npt
import matplotlib.pyplot as plt
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

        test_predictions = np.array([])
        test_labels =  np.array([])
        for x, y in test_data:
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


        return final_metrics, test_metrics, visualizations_save_dir



            
    

    



        

