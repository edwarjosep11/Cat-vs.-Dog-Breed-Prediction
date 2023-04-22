# Cat vs. Dog Prediction: Image Classification with Keras, Tensorflow, CNNs 

## Overview
This project aims to classify images of cats and dogs using machine learning techniques. The project includes data preprocessing, model building, and evaluation steps. The project is saved in a GitHub repository with the following files:

## Files 

* README.md: A markdown file that contains information about the project.
* [Data Preprocessing](https://nbviewer.org/github/edwarjosep11/Cat-vs.-Dog-Prediction/blob/main/Data%20Preprocessing.ipynb): A Jupyter Notebook that contains the code for preprocessing the image data for the first model (simple CNN).
* [First Model: Simple CNN](https://nbviewer.org/github/edwarjosep11/Cat-vs.-Dog-Prediction/blob/main/First%20Model.ipynb): A Jupyter Notebook that contains the code for building and training the first version of the model. **Accuracy: 66%.**
* [Second Model: Tuned CNN](https://nbviewer.org/github/edwarjosep11/Cat-vs.-Dog-Prediction/blob/main/Second_model.ipynb): A Jupyter Notebook that contains the code for building and training the second version of the model, using Pipeline Preprocessing and Data Augmentation. **Accuracy: 93%**. 
* [Third Model: Transfer Learning](https://nbviewer.org/github/edwarjosep11/Cat-vs.-Dog-Prediction/blob/main/cat-vs-dog-transfer-learning.ipynb): A Jupyter Notebook that includes building a model based on the head of VGG16. **Accuracy: 98%**.
* [Requirements.txt](https://github.com/edwarjosep11/Cat-vs.-Dog-Prediction/blob/main/requirements.txt): The textfile which contains the necessary libraries to execute the code smoothly.

## Setup
Run the following code to setup your environment:
```
pip install requirements.txt
```


## Data Preprocessing
The image data used in this project was obtained from the [Kaggle Cats vs. Dogs dataset](https://www.kaggle.com/datasets/arpitjain007/dog-vs-cat-fastai) and contains photos of cats and dogs (more than 8000 entries). The data was preprocessed using techniques such as data augmentation, normalization, and resizing. The preprocessed data was split into training, validation, and test sets.

## Model Building
Three different versions of the model were built and trained. The first model was a basic convolutional neural network (CNN) with two convolutional layers and two dense layers. The second model was an improved version of the first model with additional convolutional and dense layers. The final model was built using transfer learning with the VGG16 architecture.

## Model Evaluation 
The performance of each model was evaluated using metrics such as accuracy, precision, and recall. The final model achieved an accuracy of 97.95%, precision of 98.29%, and recall of 97.60% on the validation set. The model was also evaluated on a test set and achieved an accuracy of 97.20%.

## Next Steps
Possible next steps for this project include:

* Experimenting with different model architectures and hyperparameters.
* Increasing the size of the training data to improve the model's performance on unseen data.
