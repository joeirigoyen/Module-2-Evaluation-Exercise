# Module 2 Evaluation Exercise
 This repository contains a neural network made from scratch, using only numpy and pandas for numerical operations and handling of arrays. This project has merely academic purposes and will be evaluated accordingly.
 
 ## What does the neural network predict?
 This particular project uses 10 columns from the Wisconsin Breast Cancer Data Set (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). Where each column represents a characteristic from the nucleus of a cell that's suspected to be malign. In reality, these columns are the result of the image processing tha is applied to a microscopic view of the cell, but for interactive purposes, these characteristics can be manually put into the network. The neural network uses the aforementioned attributes to determine if the cell is benign ([0]) or malign ([1]). 
 
 ## How to add your own data?
 The project will showcase it's accuracy every 1000 epochs as well as the results of a select number of predictions, after that, any user can add their own values by interacting with the console. Bear in mind that only values within the 1-10 range will be accepted, otherwise, the program will automatically cap the input to the minimum/maximum possible value. 
