# AIML Assignment 8
# Binary Classification using Logistic Regression

# Step 1: Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# Step 2: Load Breast Cancer Dataset

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Loaded Successfully!")
print("Dataset Shape:", X.shape)
print("Target Names:", data.target_names)

# Step 3: Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 4: Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Logistic Regression Model

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# Step 6: Make Predictions

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# Step 7: Calculate Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Step 8: Display Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Step 9: Calculate ROC Curve and AUC

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

print("\nAUC Score:", auc)

# Step 10: Plot ROC Curve

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc:.3f})"
)

plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")
plt.legend()
plt.grid()

plt.show()

print("\nAssignment 8 Completed Successfully!")