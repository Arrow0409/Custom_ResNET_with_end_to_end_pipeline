import os
import json
import pandas as pd
import shutil
import logging

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("data-ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'data-ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def download_eurosat() -> None:
    """
    Download and unzip the EuroSAT dataset from kaggle if not already present.
    Requires Kaggle API to be configured.

    """
    dataset_dir = "EuroSAT"
    zip_file = "eurosat-dataset.zip"
    if os.path.exists(dataset_dir):
        logger.debug("EuroSAT dataset already exists. Skipping download and extraction.")
        return
    try:
        logger.debug("EuroSAT dataset not found. Downloading...")
        exit_code = os.system("kaggle datasets download apollo2506/eurosat-dataset")
        if exit_code != 0 or not os.path.exists(zip_file):
            raise RuntimeError("Failed to download EuroSAT dataset from Kaggle.")
        # Add -q flag to suppress unzip output
        exit_code = os.system(f"unzip -q -o {zip_file} -d {dataset_dir}")
        if exit_code != 0:
            raise RuntimeError("Failed to unzip EuroSAT dataset.")
        logger.debug("EuroSAT dataset downloaded and extracted successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found during download: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error during download: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise


def assign_eurosat_paths(base_path: str) -> tuple[str, str, str, str, str]:
    """
    Assign file paths for EuroSAT dataset.
    
    :param base_path: Base directory where EuroSAT is located.
    :return: Tuple containing paths for data directory, label map, train CSV, test CSV
    
    """
    try:
        data_dir = os.path.join(base_path, "EuroSAT")
        label_map_path = os.path.join(data_dir, "label_map.json")
        train_csv_path = os.path.join(data_dir, "train.csv")
        test_csv_path = os.path.join(data_dir, "test.csv")
        validation_csv_path = os.path.join(data_dir, "validation.csv")
        logger.debug("File paths assigned.")
        return data_dir, label_map_path, train_csv_path, test_csv_path, validation_csv_path
    except Exception as e:
        logger.error(f"Error during path assignment: {e}")
        raise


def load_eurosat_dataframes(label_map_path: str, train_csv_path: str, test_csv_path: str, validation_csv_path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load EuroSAT dataset CSV files into pandas DataFrames.

    :param label_map_path: Path to label map JSON file.
    :param train_csv_path: Path to training CSV file.
    :param test_csv_path: Path to testing CSV file.
    :param validation_csv_path: Path to validation CSV file.
    :return: Tuple containing label map dict, training DataFrame, testing DataFrame, validation
    
    """
    try:
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        validation_df = pd.read_csv(validation_csv_path)
        logger.debug("DataFrames loaded successfully.")
        return label_map, train_df, test_df, validation_df
    except FileNotFoundError as e:
        logger.error(f"File not found during DataFrame loading: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during DataFrame loading: {e}")
        raise


def save_image_folders(src_dir: str, dst_dir: str) -> None:
    """
    Copy only image folders from source to destination, skipping files like CSV and JSON.
    
    :param src_dir: Source directory containing image folders.
    :param dst_dir: Destination directory to copy image folders to.
    
    """
    try:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)
        # Copy only directories inside src_dir
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dst_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
        logger.debug(f"Copied image folders from {src_dir} to {dst_dir}")
    except FileNotFoundError as e:
        logger.error(f"Source folder not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during copy: {e}")
        raise


def get_full_path(row: pd.Series, data_dir: str) -> str:
    """
    Get full file path for a given row in the DataFrame.
    :param row: Row from DataFrame containing 'ClassName' and 'Filename'.
    :param data_dir: Base directory where images are stored.
    :return: Full file path as a string.

    """
    try:
        class_name = row['ClassName']
        filename = row['Filename']
        file_name_only = filename.split('/')[-1]
        return os.path.join(data_dir, class_name, file_name_only)
    except Exception as e:
        logger.error(f"Error constructing full path: {e}")
        raise

def save_data(df: pd.DataFrame | dict, filename: str, output_dir: str) -> None:
    """Save a DataFrame to a CSV file, or a dict to JSON."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        if isinstance(df, pd.DataFrame):
            df.to_csv(file_path, index=False)
            logger.debug(f"Data saved to {file_path}")
        elif isinstance(df, dict):
            with open(file_path, 'w') as f:
                json.dump(df, f)
            logger.debug(f"JSON saved to {file_path}")
        else:
            raise ValueError("Unsupported data type for saving.")
    except FileNotFoundError as e:
        logger.error(f"File not found error while saving: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}")


def cleanup_eurosat_files(zip_file: str, dataset_dir: str) -> None:
    """
    Delete the EuroSAT zip file and extracted folder.
    param zip_file: Path to the zip file.
    param dataset_dir: Path to the extracted dataset directory.
    
    """
    try:
        if os.path.exists(zip_file):
            os.remove(zip_file)
            logger.debug(f"Deleted zip file: {zip_file}")
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
            logger.debug(f"Deleted dataset folder: {dataset_dir}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise

def main():
    try:
        os.makedirs("data", exist_ok=True)
        download_eurosat()
        data_dir, label_map_path, train_csv_path, test_csv_path, validation_csv_path = assign_eurosat_paths()
        label_map, train_df, test_df, validation_df = load_eurosat_dataframes(label_map_path, train_csv_path, test_csv_path, validation_csv_path)
        
        save_image_folders(src_dir=data_dir, dst_dir="data/raw/images")
        data_dir = "data/raw/images"
        
        train_df['file_path'] = train_df.apply(lambda row: get_full_path(row, data_dir), axis=1)
        test_df['file_path'] = test_df.apply(lambda row: get_full_path(row, data_dir), axis=1)
        validation_df['file_path'] = validation_df.apply(lambda row: get_full_path(row, data_dir), axis=1)

        if train_df['Label'].dtype == object:
            train_df['Label'] = train_df['Label'].map(label_map)
            test_df['Label'] = test_df['Label'].map(label_map)
            validation_df['Label'] = validation_df['Label'].map(label_map)
        
        save_data(train_df, "train_data.csv", output_dir="data/raw")
        save_data(test_df, "test_data.csv", output_dir="data/raw")
        save_data(validation_df, "validation_data.csv", output_dir="data/raw")
        save_data(label_map, "label_map.json", output_dir="data/raw")

        cleanup_eurosat_files(zip_file="eurosat-dataset.zip", dataset_dir="EuroSAT")
        
    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()