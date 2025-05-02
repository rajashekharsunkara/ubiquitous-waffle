# Lung Disease Prediction System

A deep learning system for predicting lung diseases using respiratory sound analysis. This project uses both the Coswara and ICBHI datasets to classify various respiratory conditions including COVID-19, pneumonia, and COPD.

## Features
- Audio preprocessing with high-pass filtering and frame segmentation
- Feature extraction including MFCCs, spectrograms, and zero-crossing rate
- CNN-RNN hybrid architecture for audio classification
- Support for multiple respiratory disease classifications
- 5-fold cross-validation for robust model evaluation

## Datasets
The system uses two primary datasets:

### Coswara Dataset
- Origin: Indian Institute of Technology, Hyderabad
- Contents: Respiratory sounds (coughs and deep breaths) from COVID-19 positive and negative subjects
- Purpose: COVID-19 diagnosis using respiratory sounds

### ICBHI Dataset
- Origin: University of Liège, Belgium
- Contents: >5,000 lung sound recordings with various respiratory conditions
- Purpose: Machine learning analysis of lung sounds

## Installation

```bash
# Clone the repository
git clone https://github.com/satyasree-kpnv/ldp.git
cd ldp

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation
1. Download the Coswara dataset from [IISc Bangalore](https://github.com/iiscleap/Coswara-Data)
2. Download the ICBHI dataset (requires registration) from [ICBHI Challenge](https://bhichallenge.med.auth.gr/)
3. Place the datasets in their respective directories:
   - `data/coswara/`
   - `data/icbhi/`
4. Run the data preparation script:
```bash
python src/download_data.py
```

## Model Training
```bash
python src/train.py --num_classes 4 --epochs 50 --batch_size 32
```

## Project Structure
```
├── data/
│   ├── coswara/       # Coswara dataset
│   └── icbhi/         # ICBHI dataset
├── models/            # Saved model checkpoints
├── notebooks/         # Jupyter notebooks
├── src/
│   ├── download_data.py   # Dataset preparation
│   ├── preprocessing.py   # Audio preprocessing
│   ├── model.py          # Model architecture
│   └── train.py          # Training script
└── requirements.txt   # Project dependencies
```

## Implementation Details
- Sample rate: 22050 Hz
- Frame length: 2.5 seconds
- Frame overlap: 50%
- Features: MFCCs, STFT spectrograms, zero-crossing rate, chroma features
- Model: CNN-RNN hybrid with attention mechanism
- Training: 5-fold cross-validation
- Optimizer: Adam
- Loss function: Cross-entropy

## License
MIT License