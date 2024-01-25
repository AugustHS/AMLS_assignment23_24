import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

#import the data
data = np.load('../Datasets/pneumoniamnist.npz') 

#split the data
train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

#adjust the dimension
x_train = train_images.reshape(train_images.shape[0], -1)
y_train = train_labels.reshape(-1)
x_test = test_images.reshape(test_images.shape[0], -1)
y_test = test_labels.reshape(-1)

#use SVM model to train 
model = svm.SVC(kernel='poly',degree=3, C=0.1, gamma=0.01)
model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#print the acccuracy 
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
print(f'Training Accuracy: {train_acc}, Test Accuracy: {test_acc}')