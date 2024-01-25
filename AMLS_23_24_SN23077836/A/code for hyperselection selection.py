import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

#import the data
data = np.load('../Datasets/pneumoniamnist.npz') 

#split the data
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']

#adjust the dimension
x_train = train_images.reshape(train_images.shape[0], -1)
y_train = train_labels.reshape(-1)
x_val = val_images.reshape(val_images.shape[0], -1)
y_val = val_labels.reshape(-1)

#find the best match of hyperparameters
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.01, 0.1, 1],
    'degree': [2, 3, 4] 
}

best_score = 0
best_params = {}

for C in param_grid['C']:
    for kernel in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            if kernel == 'poly':
                for degree in param_grid['degree']:
                    svc = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree)
                    svc.fit(x_train, y_train)
                    score = accuracy_score(y_val, svc.predict(x_val))
                    if score > best_score:
                        best_score = score
                        best_params = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
            else:
                svc = svm.SVC(C=C, kernel=kernel, gamma=gamma)
                svc.fit(x_train, y_train)
                score = accuracy_score(y_val, svc.predict(x_val))
                if score > best_score:
                    best_score = score
                    best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}

print("Best Score:", best_score)
print("Best Parameters:", best_params)