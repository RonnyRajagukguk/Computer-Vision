# EMNIST Handwritten Character Classification using HOG and SVM

A machine learning project for handwritten character recognition using Histogram of Oriented Gradients (HOG) feature extraction and Support Vector Machine (SVM) classification on the EMNIST Balanced dataset.

## Table of Contents
- Project Overview
- Dataset
- Methodology
- System Architecture
- Implementation Steps
- Evaluation
- Results
- File Structure
- Libraries Used
- Author

# Project Overview
This project implements a complete handwritten character classification pipeline using classical computer vision and machine learning methods.

The workflow combines:

- Image preprocessing
- HOG feature extraction
- Feature normalization
- SVM classification
- Hyperparameter tuning
- LOOCV evaluation
- Train-test validation

The objective is to recognize handwritten characters from the EMNIST Balanced dataset efficiently and accurately.

# Dataset
Dataset used:
EMNIST Balanced
Dataset characteristics:

- Image size: 28×28 pixels
- Grayscale images
- Multi-class classification
- CSV format
- First column = label
- Remaining 784 columns = image pixels

Configuration:
```python
DATASET_PATH='./Data Sheet/emnist-balanced-test.csv'
SAMPLE_TOTAL=940
```

Balanced sampling is performed so every class contributes similar data proportions.
# Methodology
The system follows several processing stages:

### 1. Data Loading
Dataset is loaded using Pandas:
```python
df=pd.read_csv(DATASET_PATH)
```
System verifies dataset existence before loading.

### 2. Balanced Sampling
Function:
```python
pick_balanced_samples()
```
Purpose:
- Reduce class imbalance
- Improve classifier stability
- Ensure fair training

Total samples: 940 samples

### 3. Image Preprocessing
EMNIST images require orientation correction.
Processing:
- transpose image
- horizontal flip

Function:
```python
to_image()
```
Output:28×28 corrected image.

### 4. HOG Feature Extraction
Histogram of Oriented Gradients (HOG) extracts shape and edge information from characters.
Parameters:
```python
orientations=9
pixels_per_cell=(8,8)
cells_per_block=(2,2)
```
Advantages:

- captures edge information
- robust feature representation
- suitable for character recognition

### 5. Feature Scaling
Features are normalized using:
```python
StandardScaler()
```
Purpose:

- equalize feature ranges
- improve SVM performance

### 6. Hyperparameter Tuning
GridSearchCV is used to search optimal parameters.
Parameter combinations:
```python
C=[1,10]
kernel=['linear','rbf']
gamma=['scale','auto']
```
Cross validation:
```python
cv=5
```

### 7. Classification
Classifier: Support Vector Machine

Configuration:

```python
kernel='rbf'
C=10
gamma='scale'
```
Reason:SVM performs well for high-dimensional feature spaces.

# System Architecture
```text
Input Image (28x28)
Image Preprocessing
HOG Feature Extraction
Feature Vector
StandardScaler
SVM Classification
Predicted Character
```
# Evaluation Method
Two evaluation methods are implemented.

## Leave One Out Cross Validation (LOOCV)
Method:
- One sample becomes testing data
- Remaining samples become training data
- Process repeated for all samples

Advantages:
- maximum use of dataset
- detailed evaluation

Implementation:
```python
LeaveOneOut()
```

## Train-Test Split
Dataset split:
Training = 80%
Testing = 20%
Implementation:
```python
train_test_split(
test_size=0.20,
stratify=y
)
```

Purpose:
Evaluate generalization capability.

# Performance Metrics
Performance metrics used:

### Accuracy
Measures prediction correctness.

### Precision
Measures prediction reliability.

### Recall
Measures ability to identify true classes.

### F1 Score
Balanced metric between precision and recall.

# Visualization
Program generates:
- Class distribution graph
- Sample image visualization
- HOG visualization
- Confusion matrix
- Evaluation metric graphs
- Train vs Test comparison


# Saved Models
The trained model is stored as:
```python
svm_emnist_model.pkl
```
Scaler:
```python
scaler_emnist.pkl
```
Purpose:
Reuse model without retraining.

# Libraries Used
Required libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scikit-image
- joblib
Installation:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image joblib
```

# Final Result
Outputs:
✔ Character visualization
✔ HOG visualization
✔ Confusion Matrix
✔ Accuracy
✔ Precision
✔ Recall
✔ F1 Score
✔ Saved model


# Author

Ronny Rajagukguk
Computer Vision — Semester 6
Politeknik Negeri Batam