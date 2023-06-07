import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("diabetes.csv")
# result = data.corr()
# plt.figure(figsize=(8, 8))
# sn.histplot(data["Outcome"])
# plt.title("Diabetes distribution")
# plt.savefig("diabetes.jpg")

# Set features and target
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]


# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    "n_estimators": [50, 100, 200, 500],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"],
}
cls = GridSearchCV(RandomForestClassifier(), param_grid=parameters, scoring="accuracy", cv=6, verbose=2, n_jobs=8)
cls.fit(x_train, y_train)
print(cls.best_score_)
print(cls.best_params_)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))
# cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1]))
# confusion = pd.DataFrame(cm, index=["Not Diabetic", "Diabetic"], columns=["Not Diabetic", "Diabetic"])
# sn.heatmap(confusion, annot=True)
# plt.savefig("diabetes_prediction.jpg")