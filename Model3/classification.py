#FULL NAME: Luka Nikolaisvili
#STUDENT NUMBER: 0674677
#DUE DATE: 2023-10-11

#All this lines total 6 which has the import I have used them to import the needed libraries that are needed to set up everything
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

#while working on the boston dataset I faced the ethical issue, after the update seems they have removed the boston dataset from the sklearn database therefor this is the workaround
#manually downloading the boston dataset from the database and cleaning it up...
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Transforming the target variable into 3 classes using the kbinsDiscretizer
n_classes = 3
kbins_discretizer = KBinsDiscretizer(n_bins=n_classes, encode='ordinal')
target_binned = kbins_discretizer.fit_transform(target.reshape(-1, 1)).astype(int)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target_binned, test_size=0.30)
# declaring and assigning the Decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
# Fitting the trainning data
clf.fit(X_train, y_train)
# predicting the class labels 
y_pred = clf.predict(X_test)


# Printing the nicely formatted confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print(f"\n confusion matrix:\n{confusion}")

#Printing the nicely formatted classification report 
report = classification_report(y_test, y_pred)
print(f"\n Classification Report:\n{report}")


