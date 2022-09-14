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
    'saga' : ['l2', 'none']
}

""" fig, axes = plt.subplots(1, 2)
axes = axes.ravel()

x_plot = np.arange(100, 600, 100)
y_plot = np.zeros((x_plot.shape[0]))
solver = 'liblinear'

for i in range(len(solver_penalties[solver])):
    penalty = solver_penalties[solver][i]
    results = run_model(x, y, x_test, penalty, solver, -1, 100)
    equal_data = pd.DataFrame(np.array(results == y_test)).value_counts()
    pie_data = equal_data.apply(lambda val : val * 100 / results.shape[0])
    axes[i].pie(pie_data, labels=['True', 'False'], colors=sns.color_palette("hls", 8), autopct='%.2f%%')
    axes[i].set_title(f"{solver} solver with {penalty} penalty accuracy by iterations")
plt.show() """

results = run_model(x, y, x_test, 'l1', 'liblinear', -1, 1000)
variance = np.var(results)
sse = np.mean((np.mean(results) - y_test) ** 2)
bias = sse - variance

fig = plt.figure(figsize=(10, 10))
plt.scatter(np.arange(len(y_test)), y_test, color='orange')
plt.scatter(np.arange(len(y_test)), results, marker='x', color='black')
plt.legend(['Real Values', 'Predictions'])
plt.show()
