import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
feature_name_df = pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/UCI HAR Dataset (1)/UCI HAR Dataset/features.txt",
                              sep='\s+',
                              header=None,
                              names=['column_index', 'column_name'])

feature_name = feature_name_df.iloc[:, 1].values.tolist()
feature_dup_df = feature_name_df.groupby('column_name').count()
print(feature_dup_df[feature_dup_df['column_index'] > 1].count())
feature_dup_df[feature_dup_df['column_index'] > 1].head()


def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),
                                  columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(),
                                   feature_dup_df,
                                   how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(
        lambda x: x[0] + '_' + str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df


def get_human_dataset():
    feature_name_df = pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/UCI HAR Dataset (1)/UCI HAR Dataset/features.txt", sep='\s+',
                                  header=None, names=['column_index', 'column_name'])

    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    X_train = pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/UCI HAR Dataset (1)/UCI HAR Dataset/train/X_train.txt", sep='\s+',
                          names=feature_name)
    X_test = pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/UCI HAR Dataset (1)/UCI HAR Dataset/test/X_test.txt", sep='\s+',
                         names=feature_name)
    y_train = pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/UCI HAR Dataset (1)/UCI HAR Dataset/train/y_train.txt", sep='\s+',
                          header=None, names=['action'])
    y_test = pd.read_csv("C:/Users/parab/OneDrive/Desktop/p3/UCI HAR Dataset (1)/UCI HAR Dataset/test/y_test.txt", sep='\s+',
                         header=None, names=['action'])

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_human_dataset()
print('## train feature dataset info()')
print(X_train.info())
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(random_state=156)
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('Decision tree prediction accuracy: {0:.4f}'.format(accuracy))
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [6, 8, 10, 12, 16, 20, 24]
}

grid_cv = GridSearchCV(dt, param_grid=params, scoring='accuracy', cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print('GridSearchCV Highest Average Accuracy Number: {0:.4f}'.format(grid_cv.best_score_))
cv_results_df = pd.DataFrame(grid_cv.cv_results_)

cv_results_df[['param_max_depth', 'mean_test_score']]

max_depths = [6, 8, 10, 12, 16, 20, 24]
for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth,
                                random_state=156)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print('max_depth = {0} accuracy: {1:.4f}'.format(depth, accuracy))
    params = {
        'max_depth': [8, 12, 16, 20],
        'min_samples_split': [16, 24],
    }

    grid_cv = GridSearchCV(dt, param_grid=params, scoring='accuracy', cv=5, verbose=1)
    grid_cv.fit(X_train, y_train)
    print('GridSearchCV highest average accuracy figure: {0:.4f}'.format(grid_cv.best_score_))

    best_dt = grid_cv.best_estimator_
    pred1 = best_dt.predict(X_test)
    accuracy = accuracy_score(y_test, pred1)
    print('Decision tree prediction accuracy: {0:.4f}'.format(accuracy))

'''Decision tree prediction accuracy: 0.8717'''