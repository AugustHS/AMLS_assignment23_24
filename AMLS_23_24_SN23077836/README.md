1. Organization of This Project
The purpose of this project is to solve two machine learning tasks. Task A is a binary classification task based on PneumoniaMNIST dataset, and task B is a multi-class classification task based on PathMNIST dataset. Before I began my experiment, I had reviewed the course content and read the relevant essays. Following, I conduct-ed several code experiments through python in VScode and finally succeeded in using the machine learning model to solve two tasks. The broad steps are importing and prepro-cessing the data, constructing the model, training the model using the training set data, tuning the hyperparameters using the validation set, and testing the model accuracy on the test set. After I got the results, I wrote the report to summarize my work.

2. The Role of Each File
A:  
code for task A: It contains the final code for task A, and you can copy it to main.py to test it.
code for hyperselection selection: It contains the code for selecting the hyperparameter in task A, mainly to show the hyperparameter i chosen and how i tested them.
B:  
code for task B: It contains the final code for task B, and you can copy it to main.py to test it.

Datasets: Please copy-paste the dataset into this folder before you run the code. Pneumoniamnist and PathMNIST should be named as 'pneumoniamnist.npz'and 'pathmnist.npz' respectively
main.py: Please copy-paste the code from folder'A' and 'B' into this file to test them.

3. Required Packages
Task A: 
numpy
scikit-learn
Task B: 
numpy
torch
torchvision
matplotlib