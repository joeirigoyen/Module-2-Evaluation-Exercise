import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from df_generator import DataframeGenerator
from sklearn.linear_model import LogisticRegression


def run_model(x, y, sample, penalty, solver, n_jobs, max_iter):
    # Train logistic regressor
    regressor = LogisticRegression(penalty=penalty, solver=solver, n_jobs=n_jobs, max_iter=max_iter).fit(x, y.ravel())
    # Get predictions
    predictions = regressor.predict(sample)
    return predictions


# Read dataframe
df = DataframeGenerator(os.path.join(os.path.split(sys.path[0])[0], "data", "breast-cancer-wisconsin.data"))
# Split training dataframe
train_df = df.train
x = train_df.drop(10, axis=1)
x_cols = x.columns
x = x.to_numpy()
y = train_df.drop(x_cols, axis=1).to_numpy()
# Split testing dataframe
test_df = df.test
x_test = test_df.drop(10, axis=1)
x_test_cols = x_test.columns
x_test = x_test.to_numpy()
y_test = test_df.drop(x_test_cols, axis=1).to_numpy()
y_test = np.reshape(y_test, (y_test.shape[0],))
# Plot results
solver_penalties = {
    'newton-cg' : ['l2', 'none'], 
    'lbfgs' : ['l2', 'none'],
    'liblinear' : ['l1', 'l2'],
    'sag' : ['l2', 'none'],
    'saga' : ['elasticnet', 'none']
}

solver = 'newton-cg'
penalty = solver_penalties[solver][0]
results = run_model(x, y, x_test, penalty, solver, -1, 100)
print(results.shape)
print(y_test.shape)

""" fig = plt.subplots(1, 2)
plt.show() """