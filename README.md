# VS Code Setup

Create virtual environment  
`python -m venv venv`

Activate virtual environment  
`.\venv\Scripts\Activate`

Make sure that pip installation points to virtual environment  
`pip --version`

Install dependencies  
`pip install -r requirements.txt`

Install recommended extensions specified in  
`extensions.json`

# Scripts

### **download.py**

This script downloads raw datasets related to motor imagery. It accepts dataset name as a starting param.  
If none is provided, it will download all datasets. Supported dataset names are:  
`bci3a` - for BCI Competition III 3a  
`bci2a` - for BCI Competition IV 2a  
`bci2b` - for BCI Competition IV 2b  
`physionet` - for Physionet

### **preprocess.py**

This script extracts expochs for selected dataset and saves them in predefined directory. It accepts dataset name as a starting param. If none is provided, it will not do anything. Supported dataset names are:  
`bci3a` - for BCI Competition III 3a  
`bci2a` - for BCI Competition IV 2a  
`bci2b` - for BCI Competition IV 2b  
`physionet` - for Physionet

### **train.py**

This scrips performs 5 fold cross validation for selected model. The final model accuracy is mean from all 5 fold accuracies. Model name is accepted as a starting param. If none is provided, it will not do anything. Supported model names are:  
`spatial` - for SpatialTransformer  
`temporal` - for TemporalTransformer  
`spatialcnn` - for SpatialCNNTransformer  
`temporalcnn` - for TemporalCNNTransformer  
`fusion` - for FusionCNNTransformer

***
# Results:


| Model | Feature Method | Learning Rate | Average Accuracy(%) |
| :---- | :---- | :---- | :---- |
| FusionCNNTransformer | raw | 0.0007 | 74.06 |
| SpatialCNNTransformer | raw | 0.0007 | 73.42 |
| SpatialTransformer | raw | 0.0001 | 68.57 |
| SpatialTransformer | raw | 0.0007 | 62.38 |
| TemporalCNNTransformer | raw | 0.0007 | 73.76 |
| TemporalTransformer | raw | 0.0001 | 74.42 |
| TemporalTransformer | raw | 0.0007 | 74.85 |

| Model | Feature Method | LearningRate | Average Accuracy(%) | nfft | hop |
| :---- | :---- | :---- | :---- | ----- | ----- |
| FusionCNNTransformer | stft | 0.0001 | 69.92 | 128 | 32 |
| FusionCNNTransformer | stft | 0.0007 | 70.75 | 128 | 32 |
| SpatialCNNTransformer | stft | 0.0007 | 69.41 | 128 | 32 |
| SpatialCNNTransformer | stft | 0.0001 | 70.21 | 128 | 32 |
| SpatialTransformer | stft | 0.0007 | 54.44 | 128 | 32 |
| SpatialTransformer | stft | 0.0007 | 54.15 | 32 | 16 |
| SpatialTransformer | stft | 0.001 | 50.82 | 512 | 64 |
| TemporalCNNTransformer | stft | 0.0001 | 74.4 | 128 | 32 |
| TemporalTransformer | stft | 0.0007 | 66.24 | 128 | 32 |
| TemporalTransformer | stft | 0.0001 | 63.68 | 128 | 32 |
| TemporalTransformer | stft | 0.001 | 61.92 | 128 | 32 |
| TemporalTransformer | stft | 0.001 | 58.29 | 256 | 64 |
| TemporalTransformer | stft | 0.001 | 53.71 | 512 | 64 |

| Model | Feature Method | Learning Rate | Average Accuracy(%) |
| :---- | :---- | :---- | :---- |
| FusionCNNTransformer | cnn | 0.0001 | 71.48 |
| SpatialCNNTransformer | cnn | 0.0001 | 70.47 |
| SpatialCNNTransformer | cnn | 0.0007 | 72.05 |
| TemporalCNNTransformer | cnn | 0.0001 | 75.23 |
| TemporalTransformer | cnn | 0.0001 | 73.48 |

| Model | Feature Method | Learning Rate | Average Accuracy(%) | Num Comp | Patch Size |
| :---- | :---- | :---- | :---- | ----- | ----- |
| FusionCNNTransformer | csp | 0.0001 | 70.5 | 6 | metoda bez patch |
| FusionCNNTransformer | csp | 0.007 | 69.55% | 6 | 16 |
| SpatialCNNTransformer | csp | 0.0001 | 69.27 | 4 | 32 |
| SpatialTransformer | csp | 0.001 | 65.69 | 8 | 16 |
| SpatialTransformer | csp | 0.0001 | 64.67 | 8 | 16 |
| SpatialTransformer | csp | 0.001 | 64.42 | 8 | 32 |
| SpatialTransformer | csp | 0.0001 | 64.4 | 8 | 32 |
| SpatialTransformer | csp | 0.01 | 61.72 | 4 | metoda bez patch |
| SpatialTransformer | csp | 0.0007 | 61.41 | 8 | metoda bez patch |
| SpatialTransformer | csp | 0.0001 | 61.25 | 6 | metoda bez patch |
| SpatialTransformer | csp | 0.0001 | 61.2 | 4 | metoda bez patch |
| SpatialTransformer | csp | 0.0001 | 60.93 | 8 | metoda bez patch |
| TemporalCNNTransformer | csp | 0.0007 | 69.37 | 4 | 32 |
| TemporalCNNTransformer | csp | 0.0007 | 69.0 | 6 | 16 |
| TemporalCNNTransformer | csp | 0.001 | 68.98 | 4 | 16 |
| TemporalCNNTransformer | csp | 0.0001 | 68.66 | 6 | 16 |
| TemporalCNNTransformer | csp | 0.0007 | 68.59 | 4 | 16 |
| TemporalCNNTransformer | csp | 0.007 | 68.55 | 6 | 16 |
| TemporalCNNTransformer | csp | 0.0001 | 68.44 | 8 | 16 |
| TemporalCNNTransformer | csp | 0.0001 | 68.23 | 4 | 32 |
| TemporalTransformer | csp | 0.0001 | 67.78 | 4 | metoda bez patch |
| TemporalTransformer | csp | 0.0001 | 67.35 | 8 | metoda bez patch |
| TemporalTransformer | csp | 0.0001 | 67.12 | 6 | metoda bez patch |
| TemporalTransformer | csp | 0.0007 | 66.8 | 6 | 16 |
| TemporalTransformer | csp | 0.001 | 65.94 | 8 | metoda bez patch |
| TemporalTransformer | csp | 0.007 | 64.9 | 6 | 16 |

| Model | Feature Method | Learning Rate | Average  Accuracy(%) | Wavelet Type |
| :---- | :---- | :---- | :---- | :---- |
| FusionCNNTransformer | wavelet | 0.0001 | 71.45 | db6 |
| FusionCNNTransformer | wavelet | 0.001 | 71.32 | db4 |
| SpatialCNNTransformer | wavelet | 0.001 | 73.49 | db4 |
| SpatialCNNTransformer | wavelet | 0.0001 | 72.49 | db4 |
| SpatialCNNTransformer | wavelet | 0.001 | 70.36 | coif3 |
| SpatialCNNTransformer | wavelet | 0.007 | 69.86 | db6 |
| SpatialCNNTransformer | wavelet | 0.0007 | 68.82 | coif3 |
| SpatialCNNTransformer | wavelet | 0.0001 | 68.57 | db6 |
| SpatialCNNTransformer | wavelet | 0.001 | 68.41 | db4 |
| SpatialTransformer | wavelet | 0.001 | 65.94 | db4 |
| SpatialTransformer | wavelet | 0.0001 | 65.28 | db4 |
| TemporalCNNTransformer | wavelet | 0.0001 | 75.41 | coif3 |
| TemporalCNNTransformer | wavelet | 0.001 | 74.15 | db4 |
| TemporalCNNTransformer | wavelet | 0.0001 | 65.71 | db6 |
| TemporalTransformer | wavelet | 0.0001 | 73.51 | coif3 |
| TemporalTransformer | wavelet | 0.0001 | 73.42 | db4 |
| TemporalTransformer | wavelet | 0.001 | 73.36 | db4 |


