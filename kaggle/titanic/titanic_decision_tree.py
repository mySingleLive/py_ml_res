
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from IPython.display import display


def accuracy(y_true, y_pred):
    if len(y_true) == len(y_pred):
        return 'Accuracy: {:.2f}%.'.format((y_true == y_pred).mean() * 100)
    return 'Number of predictions does not match number of outcomes!'

train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')
survived_data = pd.read_csv('dataset/gender_submission.csv')
# print(train_data.head())

# display(train_data.info())

X_train = train_data[['Pclass', 'Age', 'Sex']]
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
y_train = train_data[['Survived']]

X_test = test_data[['Pclass', 'Age', 'Sex']]
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
y_test = survived_data[['Survived']]

vec = DictVectorizer(sparse=False)

X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.fit_transform(X_test.to_dict(orient='record'))

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

display(dtc.score(X_test, y_test))
print(accuracy(np.reshape(y_test.values, [-1]), y_pred))

print(classification_report(y_true=y_test, y_pred=y_pred, target_names=['Died', 'Survived']))

