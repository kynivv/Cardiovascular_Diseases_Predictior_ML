import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as asc
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Data Import
df = pd.read_csv('CVD.csv')


# Data Preprocessing
print(df.info())

print(df['Weight_(kg)'])
df['Weight_(kg)'] = df['Weight_(kg)'].round()

#print(df['BMI'])
df['BMI'] = df['BMI'].round()

print(df.dtypes)
le = LabelEncoder()

for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = le.fit_transform(df[c])


# Train Test Split
target = df['Heart_Disease']

features = df.drop('Heart_Disease', axis= 1)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, random_state= 42, test_size= 0.20)


# Model Training
models = [DecisionTreeClassifier(), KNeighborsClassifier()]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Training Accuracy : {asc(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy : {asc(Y_test, pred_test)}')