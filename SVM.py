from sklearn.datasets import make_blobs
from sklearn          import svm

import matplotlib.pyplot as plt
import numpy             as np
import mySVM

# ----------------------------------------------------------------------------------------- #

X, Y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.9)

for i in range(len(Y)):
    if(Y[i] == 0):
        Y[i] = -1

#Visualize the data
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Weight')
plt.ylabel('Sweetness')
plt.title('Toy Dataset: Apples vs Oranges')
plt.show()

xx, yy = np.meshgrid( np.linspace(min(X[:, 0]), max(X[:, 0]) , 100), 
                      np.linspace(min(X[:, 1]), max(X[:, 1]) , 100) )

clf_1 = svm.SVC(  kernel = 'rbf', C = 10.0, gamma = 3, degree = 6, coef0 = 1.0)
clf_2 = mySVM.SVM(Kernel = 'rbf', C = 10.0, gamma = 3, degree = 6, coef0 = 1.0)

clf_1.fit(X, Y)
clf_2.fit(X, Y)

Z_1 = clf_1.predict(np.c_[xx.ravel(), yy.ravel()])
Z_1 = Z_1.reshape(xx.shape)

Z_2 = clf_2.predict(np.c_[xx.ravel(), yy.ravel()])
Z_2 = Z_2.reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

# Visualize the data and decision boundary
ax1.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
ax1.contourf(xx, yy, Z_1, alpha=0.3)
ax1.set_xlabel('Weight')
ax1.set_ylabel('Sweetness')
ax1.set_title('sklearn-SVM Classification')

# Visualize the data and decision boundary
ax2.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
ax2.contourf(xx, yy, Z_2, alpha=0.3)
ax2.set_xlabel('Weight')
ax2.set_ylabel('Sweetness')
ax2.set_title('mySVM Classification')

plt.show()
