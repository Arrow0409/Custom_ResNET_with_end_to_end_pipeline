from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import logging
import tensorflow as tf
import json
import yaml
from dvclive import Live

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("model-evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'model-evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(config_path: str) -> dict:
    """
    Load parameters from a YAML configuration file.
    :param config_path: Path to the YAML configuration file
    :return: Dictionary of parameters
    """
    try:
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded from {config_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters from {config_path}: {e}")
        raise

def load_saved_dataset(load_path: str) ->tf.data.Dataset:
    """Load a saved TensorFlow dataset from disk
    
    :param load_path: Path to the saved dataset
    :return: Loaded tf.data.Dataset
    
    """
    try:
        dataset = tf.data.Dataset.load(load_path)
        logger.info(f"Dataset loaded from {load_path}")
        return dataset
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def load_model(model_path: str) ->tf.keras.Model:
    """
    Load a saved TensorFlow/Keras model from disk.
    
    :param model_path: Path to the saved model.
    :return: Loaded Keras Model.
    
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def extract_labels_and_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) ->tuple[np.ndarray, np.ndarray]:
    """
    Extract true labels and model predictions from a dataset.
    
    :param model: Keras Model used for predictions.
    :param dataset: tf.data.Dataset containing images and labels.
    :return: Tuple of numpy arrays (true_labels, predicted_labels).
    
    """
    try:
        test_images, test_labels = [], []
        for images, labels in dataset.unbatch():
            test_images.append(images.numpy())
            test_labels.append(labels.numpy())

        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        test_predictions = model.predict(test_images, verbose=0)
        predicted_labels = np.argmax(test_predictions, axis=1)
        logger.info("Labels and predictions extracted successfully.")
        return test_labels, predicted_labels
    except Exception as e:
        logger.error(f"Error during label and prediction extraction: {e}")
        raise


def calculate_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) ->dict:
    """
    Calculate evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrix.
    
    :param true_labels: Numpy array of true labels.
    :param predicted_labels: Numpy array of predicted labels.
    :return: Dictionary containing evaluation metrics.
    
    """
    try:
        # Accuracy
        model_accuracy = accuracy_score(true_labels, predicted_labels) * 100
        # Precision, Recall, F1-Score
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
        model_results ={"accuracy": model_accuracy,
                        "precision": model_precision,
                        "recall": model_recall,
                        "f1_score": model_f1}
        logger.info("Evaluation metrics calculated successfully.")
        return model_results
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


def save_metrics(metrics: dict, file_path: str) ->None:
    """
    Save evaluation metrics to a JSON file.
    :param metrics: Dictionary containing evaluation metrics.
    :param file_path: Path to save the JSON file.
    
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

def main() ->None:
    try:
        # Paths
        params = load_params(config_path="params.yaml")
        #best = params["model_evaluation"]["best"]
        
        #best = True

        model_path = "models/resnet18_best_model.keras"
        """
        if best:
            model_path = "models/resnet18_best_model.keras"
        else:
            model_path = "models/resnet18_final_model.keras"

        """
        test_dataset_path = "data/processed/test_dataset"
        
        # Load model and dataset
        model = load_model(model_path)
        test_dataset = load_saved_dataset(test_dataset_path)
        
        # Extract labels and predictions
        true_labels, predicted_labels = extract_labels_and_predictions(model, test_dataset)
        
        # Calculate metrics
        metrics = calculate_metrics(true_labels, predicted_labels)

        #Experiment tracking with DVC Live
        with Live(save_dvc_exp=True) as live:
            live_metrics = calculate_metrics(true_labels, predicted_labels)
            live.log_metric("accuracy", live_metrics["accuracy"])
            live.log_metric("precision", live_metrics["precision"])
            live.log_metric("recall", live_metrics["recall"])
            live.log_metric("f1_score", live_metrics["f1_score"])


            live.log_params(params)

        save_metrics(metrics, file_path="reports/metrics.json")
    except Exception as e:
        logger.error(f"Error in main evaluation process: {e}")
        raise

if __name__ == "__main__":
    main()