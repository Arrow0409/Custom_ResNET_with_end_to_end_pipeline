import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("data-preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'data-preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_and_preprocess_image(filepath: str, label: str, img_size: Tuple[int, int] = (64, 64)) -> Tuple[tf.Tensor, str]:
    """
    Load and preprocess an image from a file path
    :param filepath: Path to the image file
    :param label: Corresponding label for the image
    :param img_size: Desired image size (height, width)
    :return: Preprocessed image tensor and label

    """
    try:
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0  # Normalize to [0, 1]
        return image, label
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        raise

def data_augmentation(image: tf.Tensor, label: str, flip: bool = True, rotate: bool = True, brightness: bool = True, contrast: bool = True) -> Tuple[tf.Tensor, str]:
    """
    Apply data augmentation to an image
    
    :param image: Input image tensor
    :param label: Corresponding label for the image
    :param flip: Whether to apply random flipping
    :param rotate: Whether to apply random rotation
    :param brightness: Whether to apply random brightness adjustment
    :param contrast: Whether to apply random contrast adjustment
    :return: Augmented image tensor and label
    
    """
    if flip: #flip the image horizontally and vertically
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    if rotate:    
        if tf.random.uniform([]) > 0.5:  # Rotate 50% of the time
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    if brightness: # Randomly adjust brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
    if contrast: # Randomly adjust contrast    
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)    
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
    
    
def create_tf_dataset_and_optimize(
    df: pd.DataFrame, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    buffer_size: int = 1000, 
    img_size: Tuple[int, int] = (64, 64), 
    augment: bool = True, 
    flip: bool = True, 
    rotate: bool = True, 
    brightness: bool = True, 
    contrast: bool = True
) -> tf.data.Dataset:
    """
    Create and optimize a TensorFlow dataset from a DataFrame.

    :param df: The pandas DataFrame containing 'file_path' and 'Label' columns.
    :param batch_size: The number of elements in each batch.
    :param shuffle: A boolean to enable or disable shuffling.
    :param buffer_size: The size of the shuffle buffer.
    :param img_size: The target image size.
    :param augment: A boolean to enable or disable data augmentation.
    :param flip: A boolean for random horizontal and vertical flipping.
    :param rotate: A boolean for random rotation.
    :param brightness: A boolean for random brightness adjustment.
    :param contrast: A boolean for random contrast adjustment.
    :return: A TensorFlow tf.data.Dataset object.
    """
    try:
        file_paths = df['file_path'].values
        labels = df['Label'].values
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(lambda fp, lb: load_and_preprocess_image(fp, lb, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            dataset = dataset.map(lambda img, lb: data_augmentation(img, lb, flip=flip, rotate=rotate, brightness=brightness, contrast=contrast), num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        logger.debug("TensorFlow dataset created and optimized.")
        return dataset
    except Exception as e:
        logger.error(f"Error creating TensorFlow dataset: {e}")
        raise


def save_dataset(dataset: tf.data.Dataset, save_path: str = "data/train_dataset") -> None:
    """
    Save a prefetch TensorFlow dataset to disk.

    :param dataset: The TensorFlow dataset to save.
    :param save_path: The directory path to save the dataset to.
   
     """
    try:
        os.makedirs(save_path, exist_ok=True)
        dataset.save(save_path)
        logger.info(f"Dataset saved at {save_path}")
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        raise


    
def main() -> None:
        try:
            IMG_SIZE=(64, 64)
            BATCH_SIZE=32
            SHUFFLE=True
            BUFFER_SIZE=1000
            AUGMENT=True
            FLIP=True
            ROTATE=True
            BRIGHTNESS=True
            CONTRAST=True


            #Loading dataframes
            train_df = pd.read_csv("data/raw/train_data.csv")
            test_df = pd.read_csv("data/raw/test_data.csv")
            validation_df = pd.read_csv("data/raw/validation_data.csv")
            labels = pd.read_json("data/raw/label_map.json", typ='series').to_dict()


            #Creating datasets
            train_dataset = create_tf_dataset_and_optimize(train_df, 
                                                           batch_size=BATCH_SIZE, 
                                                           shuffle=SHUFFLE, 
                                                           buffer_size=BUFFER_SIZE, 
                                                           img_size=IMG_SIZE, 
                                                           augment=AUGMENT, 
                                                           flip=FLIP, 
                                                           rotate=ROTATE, 
                                                           brightness=BRIGHTNESS, 
                                                           contrast=CONTRAST)
            
            validation_dataset = create_tf_dataset_and_optimize(validation_df, 
                                                                batch_size=BATCH_SIZE, 
                                                                shuffle=False, 
                                                                buffer_size=BUFFER_SIZE, 
                                                                img_size=IMG_SIZE, 
                                                                augment=False)
            
            test_dataset = create_tf_dataset_and_optimize(test_df, 
                                                          batch_size=BATCH_SIZE, 
                                                          shuffle=False, 
                                                          buffer_size=BUFFER_SIZE, 
                                                          img_size=IMG_SIZE, 
                                                          augment=False)
            # Saving datasets
            save_dataset(train_dataset, save_path="data/processed/train_dataset")
            save_dataset(validation_dataset, save_path="data/processed/validation_dataset")
            save_dataset(test_dataset, save_path="data/processed/test_dataset")

            logger.debug("Datasets created and optimized successfully.")
        except Exception as e:
            logger.error(f"Error in main preprocessing function: {e}")
            raise

if __name__=="__main__":
    main()