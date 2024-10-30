# Comment Toxicity Prediction using Deep Learning

This repository contains a project for predicting toxic comments using a Deep Learning model. The project focuses on identifying various types of toxic comments and classifying them into categories like toxic, severe toxic, obscene, threat, insult, and identity hate.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Comment Toxicity Prediction is a crucial task for maintaining a healthy online community by filtering harmful or offensive comments. This project aims to build a deep learning model using Bidirectional LSTM to classify comments based on their toxicity levels. It provides a reliable method for detecting and managing toxic comments on online platforms.

## Dataset

The dataset used in this project is from Kaggle's Jigsaw Toxic Comment Classification Challenge. It contains user comments with labels indicating the presence of different types of toxicity. The dataset is split into training and testing sets to evaluate the model's performance.

- [Jigsaw Toxic Comment Classification Challenge Dataset on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install numpy pandas tensorflow gradio
```

Requirements
Python 3.x
NumPy
Pandas
TensorFlow
Gradio

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Comment-Toxicity-Prediction-using-Deep-Learning.git

```

2. Navigate to the project directory:
   cd Comment-Toxicity-Prediction-using-Deep-Learning

3. Download the dataset from Kaggle and place it in the project folder

4. Open and run the Jupyter Notebook:
   jupyter notebook Toxicity.ipynb

## Model

The model used in this project is a Bidirectional LSTM. The data is preprocessed using text vectorization techniques and trained using deep learning layers. Key steps include:

## Data Preprocessing

- Text Vectorization: Converting text into sequences using TensorFlow’s TextVectorization layer.
- Train-Test Split: Splitting the dataset into training validation and testing sets for model evaluation.

## Model Training

Bidirectional LSTM: A sequential deep learning model with an embedding layer, Bidirectional LSTM layers, and dense layers to predict multiple toxicity classes.

### Evaluation

The model is evaluated using the following metric:

- Categorical Accuracy: Measures the percentage of correct predictions across multiple classes.
- Precision: Measures the model’s ability to correctly identify toxic comments.
- Recall: Measures the model’s ability to detect all relevant toxic comments.
