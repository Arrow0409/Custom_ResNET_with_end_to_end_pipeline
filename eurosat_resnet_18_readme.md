# EuroSAT ResNet-18: End-to-End DVC Pipeline

A reproducible image classification pipeline for EuroSAT using a custom ResNet-18 built with TensorFlow/Keras and managed via DVC.

---

## 📊 Dataset
EuroSAT (RGB) satellite imagery dataset for land-use classification across **10 classes**.  

- **Source**: [Kaggle - EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

---

## 🔎 Visual Exploration
For detailed visualizations, plots, training curves, and data exploration:  
- `experiments/EuroSAT_custom_resnet.ipynb`

---

## ✨ Key Features
- **Custom ResNet-18**: Implemented from scratch with residual blocks and 1×1 projection shortcuts  
- **Automated Data Pipeline**: Handles Kaggle ingestion, CSV assembly, and label mapping  
- **Efficient Preprocessing**: `tf.data` pipelines for decode → resize → normalize → augment  
- **Robust Training**: Includes `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`  
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score with DVCLive logging  

---

## 🗂️ Repository Structure
```
├── data/
│   ├── raw/          # CSVs, label map, raw images
│   └── processed/    # Preprocessed tf.data datasets
├── models/           # Trained Keras models (.keras)
├── reports/          # Evaluation metrics (metrics.json)
├── dvclive/          # DVCLive experiment logs
├── experiments/      # Jupyter notebooks
├── src/              # Pipeline scripts
│   ├── data_ingestion.py
│   ├── Pre-processing.py
│   ├── model_building.py
│   ├── model_training.py
│   └── model_evaluation.py
├── logs/             # Logs for each stage
├── dvc.yaml          # DVC pipeline configuration
└── params.yaml       # Hyperparameters and settings
```

---

## ⚡ Quick Start

### Prerequisites
```bash
pip install tensorflow scikit-learn pandas pyyaml dvc[all] dvclive kaggle
```

### Setup
1. **Configure Kaggle API**: Place `kaggle.json` in `~/.kaggle/`  
2. **Initialize DVC** (if not done):  
   ```bash
   dvc init
   ```

### Run Complete Pipeline
```bash
dvc repro
```

### Run Individual Stages
```bash
python src/data_ingestion.py
python src/Pre-processing.py
python src/model_building.py
python src/model_training.py
python src/model_evaluation.py
```

---

## ⚙️ Configuration

### Parameters (params.yaml)
**Preprocessing**
- `batch_size`: 32  
- `shuffle`: True  
- `buffer`: 1000  
- `augment`: True  
- `flip`: True  
- `rotate`: True  
- `brightness`: True  
- `contrast`: True  

**Training**
- `num_classes`: 10  
- `epochs`: 20  
- `learning_rate`: 0.001  

---

## 🏗️ Model Architecture
**ResNet-18**:
1. 7×7 Conv → BatchNorm → ReLU → 3×3 MaxPool  
2. Residual stages with filters: 64, 128, 256, 512 (2 blocks each)  
3. Downsampling via 1×1 projections  
4. GlobalAveragePooling → Dense(10, softmax)  

- **Input**: 64×64×3 RGB images  
- **Output**: 10-class probabilities  

---

## 🔄 Data Flow
1. **Ingestion** → Downloads EuroSAT → builds CSVs with file paths  
2. **Preprocessing** → Creates `tf.data` pipelines with augmentation → saves datasets  
3. **Training** → Loads datasets → trains ResNet-18 → saves checkpoints  
4. **Evaluation** → Loads best model → computes metrics → logs with DVCLive  

---

## 📈 Results & Artifacts
- **Metrics**: `reports/metrics.json` → test accuracy, precision, recall, F1-score  
- **Models**:
  - `models/resnet18_best_model.keras` → best checkpoint  
  - `models/resnet18_final_model.keras` → final trained model  
- **Experiment Tracking**: step-wise logs in `dvclive/`  
- **Logs**: execution logs in `logs/`  

---

## 🔧 DVC Pipeline Stages
| Stage            | Command                          | Dependencies         | Outputs            |
|------------------|----------------------------------|----------------------|--------------------|
| data-ingestion   | `python src/data_ingestion.py`   | src/data_ingestion.py| data/raw           |
| Pre-processing   | `python src/Pre-processing.py`   | data/raw             | data/processed     |
| model_building   | `python src/model_building.py`   | src/model_building.py| -                  |
| model_training   | `python src/model_training.py`   | data/processed       | models/            |
| model_evaluation | `python src/model_evaluation.py` | models, data/processed| reports/metrics.json |

---

## 🛠️ Troubleshooting
- **Kaggle Issues**: Verify API credentials (`kaggle datasets list`)  
- **Dataset Load Errors**: Ensure `data/processed/` exists after preprocessing  
- **Training Issues**: Confirm `num_classes=10` in `params.yaml`  
- **DVC Issues**:  
  ```bash
  dvc status
  dvc repro --force
  ```

---

## 🧑‍💻 Technical Details
- **Framework**: TensorFlow 2.x + Keras  
- **Optimizer**: Adam + `ReduceLROnPlateau`  
- **Regularization**: BatchNorm, EarlyStopping  
- **Data Pipeline**: `tf.data` with AUTOTUNE  
- **Reproducibility**: Hyperparameters tracked in `params.yaml`  
- **Experiment Tracking**: DVCLive integration  

---

## 📜 License
MIT License - see [LICENSE](LICENSE) for details.

---

## 📖 Citation
```bibtex
@misc{eurosat-resnet18-dvc,
  title={EuroSAT ResNet-18 DVC Pipeline},
  author={Your Name},
  year={2025},
  howpublished={https://github.com/your-username/your-repo}
}
```

---

## 📚 References
- [EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)  
- [DVC Documentation](https://dvc.org/doc)  
- [DVCLive](https://dvc.org/doc/dvclive)  
- [ResNet Paper](https://arxiv.org/abs/1512.03385)  

---

