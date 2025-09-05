import tensorflow as tf
import os
import logging
from tensorflow.keras import Model
from model_building import ResNet18
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import yaml

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("model-training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'model-training.log')
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


def train_model(model: Model, train_data: tf.data.Dataset, test_data: tf.data.Dataset, params:dict, model_save_path: str) ->Model:
    """
    Train the ResNet-18 model with callbacks for checkpointing, learning rate reduction, and early stopping.
    
    :param model: Keras Model to be trained.
    :param train_data: Training dataset.
    :paramtest_data: Validation dataset.
    :param params: Dictionary containing training parameters like 'epochs' and 'learning_rate'.
    :param model_save_path: Path to save the best model.
    """
    os.makedirs(model_save_path, exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'resnet18_best_model.keras'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',  
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )

    try:
        # Compile model
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            metrics=['accuracy']
        )
        logger.info("model compiled successfully.")
    
        logger.info("Starting model training...")
        # Train model
        model.fit(
            train_data,
            epochs=params['epochs'],
            validation_data=test_data,
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )
        logger.info(f"Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_trained_model(model, model_save_path:str)->None:
    """Save the trained model to disk
    :param model: Trained Keras Model
    :param model_save_path: Path to save the trained model"""
    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        model.save(model_save_path)
        logger.info(f"Trained model saved at {model_save_path}")
    except Exception as e:
        logger.error(f"Error saving trained model: {e}")
        raise


def main() -> None:
    try:
        params = load_params(config_path="params.yaml")['model_training']
        NUM_CLASSES = params['num_classes']

        """
            INPUT_SHAPE = (64, 64, 3)
            NUM_CLASSES = 10
            epochs = 20
            learning_rate = 0.001

        """
        # Load datasets
        train_dataset = load_saved_dataset(load_path="data/processed/train_dataset")
        val_dataset = load_saved_dataset(load_path="data/processed/validation_dataset")
        
        # Build model
        model = ResNet18(input_shape=(64, 64, 3), num_classes=params['num_classes'])
        logger.info("ResNet-18 model built successfully.")
        
        # Train model
        trained_model = train_model(model, train_dataset, val_dataset, params, model_save_path="models")
        
        # Save trained model
        save_trained_model(trained_model, model_save_path="models/resnet18_final_model.keras")
        logger.info("Model training and saving completed successfully.")
    except Exception as e:
        logger.error(f"Error in main training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()