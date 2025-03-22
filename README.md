# **Dermatology AI Classification Challenge**

## **Project Overview**

This project aims to build an inclusive machine learning model for dermatology, addressing the underperformance of AI tools on darker skin tones due to limited diverse training data. The model classifies 21 different skin conditions across a range of skin tones using a dataset derived from the FitzPatrick17k dataset.

## **Dataset**

The dataset contains approximately 4500 images categorized into 21 skin conditions. It includes metadata such as the FitzPatrick skin tone scale, diagnostic labels, and quality control annotations.

### **Files:**

* `images.zip`: Contains the dataset images divided into `train` and `test` directories.  
* `train.csv`: Metadata for training images, including labels.  
* `test.csv`: Metadata for test images without labels (to be predicted).  
* `sample_submission.csv`: Sample format for submission.

## **Model Implementation**

### **Dependencies**

The project utilizes the following Python libraries:

* `numpy`, `pandas`  
* `tensorflow`, `keras`  
* `sklearn`  
* `PIL`

### **Preprocessing**

* Loads training and test metadata from CSV files.  
* Prepares image paths based on labels and md5hash values.  
* Encodes labels using `LabelEncoder`.  
* Splits the training set into training and validation subsets.

### **Model Architecture**

* Convolutional Neural Network (CNN) with layers:  
  * Convolutional layers with ReLU activation  
  * MaxPooling layers for dimensionality reduction  
  * Flatten layer for feature extraction  
  * Dense layers for classification  
  * Dropout layers for regularization  
* Optimized using Adam optimizer and early stopping to prevent overfitting.

### **Training**

* Uses `train_test_split` to create validation sets.  
* Model trained with categorical cross-entropy loss and accuracy metrics.

## **Evaluation**

* Models are evaluated using a weighted average F1-score.

## **Submission Format**

Predictions are stored in a CSV file with the following format:

md5hash,label  
16d1e6b4143c88cb158a50ea8bc3a595,acne-vulgaris  
aceebbdcfd419fa960ebe3933d721550,folliculitis  
85bfb7325d93bac71fcbc08ae0a9ba23,dermatomyositis

## **Fairness Considerations**

* Uses AI fairness and explainability tools to assess model biases.  
* Evaluates impact on underrepresented populations.  
* Implements data augmentation to ensure balanced representation.

## **How to Run**

Install dependencies:  
pip install numpy pandas tensorflow scikit-learn pillow

Run the training script:    
python train\_test\_skin.ipynb