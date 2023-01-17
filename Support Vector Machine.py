# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:16:56 2023

@author: SamuelOkachi
"""
#filename: OKACHI SAMUEL NYECHE (21081467)
#MSC. DATA SCIENCE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from statsmodels.tools.eval_measures import mse
from sklearn import metrics
import seaborn as sns

breast_cancer = datasets.load_breast_cancer()

print(breast_cancer)

# covert the dataset into a DataFrame
# Make use pf pd.DataFrame to convert data
# Now dataset can be more readbable and understandable
df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
print(df.head())

# Giving shape to our data
breast_cancer.data.shape
print(breast_cancer.data.shape)

print(df)

# mathematical description of our dataset
df.describe()
print(df.describe())

# Transpose the rows and columns of our data
df.describe().T
print(df.describe().T)

# Define our DIAGNOSIS
df['diagnosis'] = breast_cancer.target
print(df.head())

# Here we get the type diagnosis from the dataset
# 1 = Benign
# 0 = Malignant
print(df['diagnosis'].value_counts())

# plotting diagnosis data in form of bar chart
# Here we see the total number of diagnosed cases
y_axis = ["Malignant", "Benign"]
x_axis = [212, 357]
title = "Breast Cancer Cases"
x_label = "Diagnosis"
y_label = "Total Cases"

def create_bar_chart(y_axis, x_axis, title, x_label, y_label):
    """
    Parameters
    ----------
    y_axis : TYPE
        DESCRIPTION. TYPES OF  DIAGNOSIS
    x_axis : TYPE    PATIENTS DIAGNOSED WITH A SPECIFIC BREAST CANCER
        DESCRIPTION.
    title : TYPE
        DESCRIPTION. BREAST CANCER DIAGNOSIS 
    x_label : TYPE
        DESCRIPTION. DIAGNOSIS
    y_label : TYPE
        DESCRIPTION. TOTAL CASES RECORDED

    Returns
    -------
    None.

    """
    plt.figure(dpi=100)
    plt.bar(y_axis,x_axis,color=['blue', 'orange'])
    plt.title(title)
    plt.xlabel("Diagnosis")
    plt.ylabel("Cancer Cases")
    plt.show()
   
create_bar_chart(y_axis, x_axis, title, x_label, y_label)


# Getting Dataframe correlation
df.corr()
print(df.corr())

# using seaborn to plot correlation
# plotting correlation in percentage%
plt.figure(figsize=(20, 20))
sns.heatmap(df.iloc[:, 1:16].corr(), annot=True, fmt='.0%')


# HERE WE ARE GOING TO TRAIN OUR DATA
# IMPLEMENTION OF SUPPORT VECTOR MACHINE
# Training data to determine its functionality and find errors if any
# Split the data into train and test set
# we using 30% of our dataset for testing and 70% for training
x = breast_cancer.data
y = breast_cancer.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.3, random_state=3)

# This means that out of 569rows
# 171 are used for test-data and 398 are used for training-data
y.shape, y_test.shape, y_train.shape
print(y.shape, y_test.shape, y_train.shape)

# Here we create a Support Vector Machine model
# using kernel trick to move from low dimension to high dimension
model = svm.SVC(kernel='linear')
print(model)

# Train the machine model
model.fit(X_train, y_train)
print(model.fit(X_train, y_train))

# Prediction of the model
y_pred = model.predict(X_test)
print(y_pred)

# Overall classification Report on our data
c_report = metrics.classification_report(y_test, y_pred)
print(c_report)

# plotting to get the result from our classifier
# Making sure our algorith can identify the types of Breast cancer data
# Note pairplot take a few second to plot due to runtime
# plot clearly show data classification regards to diagnosis
sns.pairplot(df, hue='diagnosis', 
             vars=['mean radius', 'mean texture', 'mean perimeter', 
                   'mean smoothness', 'mean compactness'])

# FOR REGRESSION IN SUPPORT VECTOR MACHINES
# FIRST WE IMPORT RERGRESSION MODULE
SVM_regression = SVR()

SVM_regression.fit(X_train, y_train)
print(SVM_regression.fit(X_train, y_train))
# Predict Test for Regression
y_pred = SVM_regression.predict(X_test)
print(mse(y_pred, y_test))

# y_test = Actual Value
# y_pred = predicted value
predict = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print(predict.head())
