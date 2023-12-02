import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
#Load dataset
data = pd.read_csv('creditcard.csv')
#Split p1
X = data.drop('Class', axis=1)
y = data['Class']
#Split p2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#DTC
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
#RFC
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
#SVM
svm = SVC(random_state=0, probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
#ACCURACY
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
#ROC-AUC
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
#CV
scores_dt = cross_val_score(dt, X, y, cv=10)
scores_rf = cross_val_score(rf, X, y, cv=10)
scores_svm = cross_val_score(svm, X, y, cv=10)
#CVScores
mean_score_dt = scores_dt.mean()
std_score_dt = scores_dt.std()
mean_score_rf = scores_rf.mean()
std_score_rf = scores_rf.std()
mean_score_svm = scores_svm.mean()
std_score_svm = scores_svm.std()
#Results
print("Decision Tree")
print("Accuracy:", accuracy_dt)
print("ROC AUC:", roc_auc_dt)
print("Cross-validation mean score:", mean_score_dt)
print("Cross-validation standard deviation:", std_score_dt)
print()
print("Random Forest")
print("Accuracy:", accuracy_rf)
print("ROC AUC:", roc_auc_rf)
print("Cross-validation mean score:", mean_score_rf)
print("Cross-validation standard deviation:", std_score_rf)
print()
print("Support Vector Machine")
print("Accuracy:", accuracy_svm)
print("ROC AUC:", roc_auc_svm)
print("Cross-validation mean score:", mean_score_svm)
print("Cross-validation standard deviation:", std_score_svm)